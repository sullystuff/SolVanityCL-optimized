# Modified to start workers once and consume results from a manager.Queue
# Wait for pending flushes to complete before exiting.

import logging
import multiprocessing
import sys
import time
import queue as _queue
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple

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


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    default=[],
    help="Public key starts with the indicated prefix. Provide multiple arguments to search for multiple prefixes.",
    multiple=True,
)
@click.option(
    "--ends-with",
    type=str,
    default="",
    help="Public key ends with the indicated suffix.",
)
@click.option("--count", type=int, default=1, help="Count of pubkeys to generate.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./",
    help="Output directory.",
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

    for prefix in starts_with:
        check_character("starts_with", prefix)
    check_character("ends_with", ends_with)

    chosen_devices: Optional[Tuple[int, List[int]]] = None
    if select_device:
        chosen_devices = get_chosen_devices()
        gpu_counts = len(chosen_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

    logging.info(
        "Searching Solana pubkey with starts_with=(%s), ends_with=%s, is_case_sensitive=%s",
        ", ".join(repr(s) for s in starts_with),
        repr(ends_with),
        is_case_sensitive,
    )
    logging.info(f"Using {gpu_counts} OpenCL device(s)")

    # settings for flush throttling
    FLUSH_INTERVAL = 5.0  # seconds

    found_count = 0       # number of matches found (from workers)
    saved_total = 0       # number of matches actually saved to disk
    pending_results: List = []
    last_flush = time.time()

    with multiprocessing.Manager() as manager:
        with Pool(processes=gpu_counts) as pool:
            kernel_source = load_kernel_source(
                starts_with, ends_with, is_case_sensitive
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
                    logging.info(f"Found {found_count}/{count} matches (pending save: {len(pending_results)})")

                # flush if enough time passed, or if we've reached the total requested matches
                if (now - last_flush) >= FLUSH_INTERVAL or found_count >= count:
                    if pending_results:
                        saved = save_result(pending_results, output_dir)
                        saved_total += saved
                        logging.info(f"Flushed {saved} results to disk (total saved: {saved_total})")
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
                    logging.info(f"Draining: collected extra pending result (pending save: {len(pending_results)})")

                # Flush periodically during drain
                if (now - last_flush) >= FLUSH_INTERVAL and pending_results:
                    saved = save_result(pending_results, output_dir)
                    saved_total += saved
                    logging.info(f"Draining flush: saved {saved} results (total saved: {saved_total})")
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
                            logging.info(f"Draining final: collected extra pending result (pending save: {len(pending_results)})")
                    break

            # Final flush of any pending results
            if pending_results:
                saved = save_result(pending_results, output_dir)
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
