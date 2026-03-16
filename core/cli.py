# Modified to start workers once and consume results from a manager.Queue
# Wait for pending flushes to complete before exiting.

import logging
import multiprocessing
import sys
import time
import queue as _queue
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Tuple

import click
import pyopencl as cl

from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_chosen_devices,
)
from core.searcher import multi_gpu_init, save_result
from core.utils.helpers import check_character, load_kernel_source

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


def _parse_pattern(value: str) -> Tuple[str, Optional[str]]:
    if ":" in value:
        pattern, dir_path = value.split(":", 1)
        return pattern, dir_path
    return value, None


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    default=[],
    help=(
        "Prefix to match. Repeatable for multiple prefixes."
        " Append :DIR to set a per-pattern output directory"
        " (e.g. --starts-with bonk:./bonk-keys)."
    ),
    multiple=True,
)
@click.option(
    "--ends-with",
    type=str,
    default=[],
    help=(
        "Suffix to match. Repeatable for multiple suffixes."
        " Append :DIR to set a per-pattern output directory"
        " (e.g. --ends-with pump:./pump-keys)."
    ),
    multiple=True,
)
@click.option("--count", type=int, default=1, help="Count of pubkeys to generate.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./",
    help="Default output directory.",
)
@click.option(
    "--select-device/--no-select-device",
    default=False,
    help="Select OpenCL device manually",
)
@click.option(
    "--iteration-bits",
    type=int,
    default=DEFAULT_ITERATION_BITS,
    help="Iteration bits (e.g., 24, 26, 28, etc.)",
)
@click.option(
    "--is-case-sensitive", type=bool, default=True, help="Case sensitive search flag."
)
def search_pubkey(
    starts_with,
    ends_with,
    count,
    output_dir,
    select_device,
    iteration_bits,
    is_case_sensitive,
):
    """Search for Solana vanity pubkeys."""
    if not starts_with and not ends_with:
        click.echo("Please provide at least one of --starts-with or --ends-with.")
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        sys.exit(1)

    prefix_entries = [_parse_pattern(s) for s in starts_with]
    suffix_entries = [_parse_pattern(s) for s in ends_with]

    for pattern, _ in prefix_entries:
        if pattern == "":
            raise click.BadParameter(
                "You need to specify a prefix before the colon (e.g. bonk:./out), not just :./out"
            )
    for pattern, _ in suffix_entries:
        if pattern == "":
            raise click.BadParameter(
                "You need to specify a suffix before the colon (e.g. pump:./out), not just :./out"
            )

    prefix_patterns = tuple(p for p, _ in prefix_entries)
    suffix_patterns = tuple(s for s, _ in suffix_entries)

    for prefix in prefix_patterns:
        check_character("starts_with", prefix)
    for suffix in suffix_patterns:
        check_character("ends_with", suffix)

    pattern_dirs: Dict[str, str] = {}
    for pattern, dir_path in prefix_entries:
        if dir_path:
            pattern_dirs[f"prefix:{pattern}"] = dir_path
    for pattern, dir_path in suffix_entries:
        if dir_path:
            pattern_dirs[f"suffix:{pattern}"] = dir_path

    chosen_devices: Optional[Tuple[int, List[int]]] = None
    if select_device:
        chosen_devices = get_chosen_devices()
        gpu_counts = len(chosen_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

    logging.info(
        "Searching Solana pubkey with starts_with=(%s), ends_with=(%s), is_case_sensitive=%s",
        ", ".join(repr(s) for s in prefix_patterns),
        ", ".join(repr(s) for s in suffix_patterns),
        is_case_sensitive,
    )
    logging.info(f"Using {gpu_counts} OpenCL device(s)")

    # settings for flush throttling
    FLUSH_INTERVAL = 5.0  # seconds

    found_count = 0       # number of matches found (from workers)
    saved_total = 0       # number of matches actually saved to disk
    pending_results: List = []
    last_flush = time.time()

    # Adaptive printing: track rate and switch to summary mode when fast
    PRINT_RATE_THRESHOLD = 5.0  # keys/sec threshold for batch printing
    last_print_time = time.time()
    keys_since_last_print = 0
    high_throughput = False  # suppress per-key logging when generating fast

    with multiprocessing.Manager() as manager:
        with Pool(processes=gpu_counts) as pool:
            kernel_source = load_kernel_source(
                prefix_patterns, suffix_patterns, is_case_sensitive
            )
            lock = manager.Lock()
            result_queue = manager.Queue()
            stop_flag = manager.Value("i", 0)

            # start long-running workers once per GPU
            async_results = []
            for x in range(gpu_counts):
                async_results.append(
                    pool.apply_async(
                        multi_gpu_init,
                        (
                            x,
                            HostSetting(kernel_source, iteration_bits),
                            gpu_counts,
                            stop_flag,
                            lock,
                            result_queue,
                            chosen_devices,
                        ),
                    )
                )

            # consume results as they come in; flush to disk at most once per FLUSH_INTERVAL
            while found_count < count:
                try:
                    res = result_queue.get(timeout=1.0)
                except _queue.Empty:
                    res = None
                now = time.time()
                if isinstance(res, (list, tuple, bytearray, bytes)) and len(res) > 0 and res[0]:
                    pending_results.append(list(res))
                    found_count += 1
                    keys_since_last_print += 1

                    # Adaptive printing: compute current rate
                    elapsed_since_print = now - last_print_time
                    if elapsed_since_print >= 1.0:
                        rate = keys_since_last_print / elapsed_since_print
                        high_throughput = rate > PRINT_RATE_THRESHOLD
                        if high_throughput:
                            # High-throughput mode: print summary once per second
                            logging.info(
                                f"Progress: {found_count}/{count} ({rate:.1f} keys/sec)"
                            )
                        else:
                            # Low-throughput mode: per-key logging handled by save_keypair
                            logging.info(f"Progress: {found_count}/{count}")
                        last_print_time = now
                        keys_since_last_print = 0
                    elif found_count == count:
                        # Always print at completion
                        if elapsed_since_print > 0:
                            rate = keys_since_last_print / elapsed_since_print
                        else:
                            rate = 0
                        logging.info(
                            f"Progress: {found_count}/{count} ({rate:.1f} keys/sec)"
                        )

                # flush if enough time passed, or if we've reached the total requested matches
                if (now - last_flush) >= FLUSH_INTERVAL or found_count >= count:
                    if pending_results:
                        saved = save_result(
                            pending_results, output_dir,
                            prefix_patterns, suffix_patterns,
                            pattern_dirs, is_case_sensitive,
                            quiet=high_throughput,
                        )
                        saved_total += saved
                        pending_results.clear()
                    last_flush = now

            # At this point we've collected the requested number of matches.
            # Signal workers to stop and then drain the queue and flush any remaining results before exiting.

            # signal workers to stop
            with lock:
                stop_flag.value = 1

            # Drain remaining results while waiting for workers to exit.
            # We loop until all worker async tasks are done and the queue is empty.
            logging.info("Signaled workers to stop; draining remaining results before exit...")
            while True:
                # Try to pull items from queue; timeout to allow checking worker status
                try:
                    res = result_queue.get(timeout=0.5)
                except _queue.Empty:
                    res = None

                now = time.time()
                if isinstance(res, (list, tuple, bytearray, bytes)) and len(res) > 0 and res[0]:
                    pending_results.append(list(res))

                # Flush periodically during drain
                if (now - last_flush) >= FLUSH_INTERVAL and pending_results:
                    saved = save_result(
                        pending_results, output_dir,
                        prefix_patterns, suffix_patterns,
                        pattern_dirs, is_case_sensitive,
                    )
                    saved_total += saved
                    pending_results.clear()
                    last_flush = now

                # If all workers are finished, also drain any remaining queue items without blocking and break
                all_finished = all(a.ready() for a in async_results)
                if all_finished:
                    # drain any remaining items quickly (non-blocking)
                    while True:
                        try:
                            res = result_queue.get_nowait()
                        except _queue.Empty:
                            break
                        if isinstance(res, (list, tuple, bytearray, bytes)) and len(res) > 0 and res[0]:
                            pending_results.append(list(res))
                    break

            # Final flush of any pending results
            if pending_results:
                saved = save_result(
                    pending_results, output_dir,
                    prefix_patterns, suffix_patterns,
                    pattern_dirs, is_case_sensitive,
                    quiet=high_throughput,
                )
                saved_total += saved
                logging.info(f"Final flush saved {saved} results (total saved: {saved_total})")
                pending_results.clear()

            # Now wait for worker futures to finish (get their results / exceptions)
            for a in async_results:
                try:
                    a.get(timeout=10)
                except Exception:
                    # ignore timeouts / exceptions here; pool termination will handle lingering processes
                    pass

    logging.info(f"Search finished. Total matches found: {found_count}, total saved: {saved_total}")
