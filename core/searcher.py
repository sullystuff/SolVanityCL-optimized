# Modified to remove NVIDIA sleep and to push results to a result_queue instead of returning immediately
import logging
import time
from typing import Dict, List, Optional, Tuple

import pyopencl as cl

from core.config import HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_selected_gpu_devices,
)


class Searcher:
    def __init__(
        self,
        kernel_source: str,
        index: int,
        setting: HostSetting,
        chosen_devices: Optional[Tuple[int, List[int]]] = None,
    ):
        if chosen_devices is None:
            devices = get_all_gpu_devices()
        else:
            devices = get_selected_gpu_devices(*chosen_devices)
        enabled_device = devices[index]
        self.context = cl.Context([enabled_device])
        self.gpu_chunks = len(devices)
        self.command_queue = cl.CommandQueue(self.context)
        self.setting = setting
        self.index = index
        self.display_index = (
            index if chosen_devices is None else chosen_devices[1][index]
        )
        self.prev_time = None
        self.is_nvidia = "NVIDIA" in enabled_device.platform.name.upper()

        program = cl.Program(self.context, kernel_source).build()
        self.kernel = cl.Kernel(program, "generate_pubkey")
        self.memobj_key32 = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            len(self.setting.key32),
            hostbuf=self.setting.key32,
        )
        self.memobj_output = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE, 33
        )
        self.memobj_occupied_bytes = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=bytearray([self.setting.iteration_bytes]),
        )
        self.memobj_group_offset = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=bytearray([self.index]),
        )
        self.output = bytearray(33)
        self.kernel.set_arg(0, self.memobj_key32)
        self.kernel.set_arg(1, self.memobj_output)
        self.kernel.set_arg(2, self.memobj_occupied_bytes)
        self.kernel.set_arg(3, self.memobj_group_offset)

    def find(self, log_stats: bool = True) -> bytearray:
        start_time = time.time()
        cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)
        global_work_size = self.setting.global_work_size // self.gpu_chunks
        local_size = self.setting.local_work_size
        global_size = ((global_work_size + local_size - 1) // local_size) * local_size  # align global size and local size
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_size,),
            (local_size,),
        )
        self.command_queue.flush()
        self.setting.increase_key32()
        # Removed the NVIDIA throttle sleep to avoid artificial throttling on quick matches
        cl.enqueue_copy(self.command_queue, self.output, self.memobj_output).wait()
        self.prev_time = time.time() - start_time
        if log_stats:
            logging.info(
                f"GPU {self.display_index} Speed: {global_work_size / ((time.time() - start_time) * 1e6):.2f} MH/s"
            )

        # If a match was found, clear the GPU output buffer so we don't report it again
        if self.output[0]:
            result = bytearray(self.output)  # Make a copy to return
            self.output[:] = bytearray(33)   # Clear local buffer
            # Clear GPU buffer too
            cl.enqueue_copy(self.command_queue, self.memobj_output, self.output).wait()
            return result

        return self.output


def multi_gpu_init(
    index: int,
    setting: HostSetting,
    gpu_counts: int,
    stop_flag,
    lock,
    result_queue,
    chosen_devices: Optional[Tuple[int, List[int]]] = None,
) -> None:
    """
    Long-running worker for a single GPU. Pushes matches into result_queue
    and exits when stop_flag.value is set.
    """
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            chosen_devices=chosen_devices,
        )
        i = 0
        st = time.time()
        while True:
            result = searcher.find(i == 0)
            if result[0]:
                # push result to shared queue
                try:
                    result_queue.put(list(result))
                except Exception:
                    logging.exception("Failed to put result into queue")
            if time.time() - st > max(gpu_counts, 1):
                i = 0
                st = time.time()
                with lock:
                    if stop_flag.value:
                        break
            else:
                i += 1
            # check stop flag between iterations
            if stop_flag.value:
                break
    except Exception as e:
        logging.exception(e)
    # worker returns (pool will collect this), but main communication happens via result_queue
    return


def _resolve_output_dir(
    pubkey: str,
    default_dir: str,
    starts_with: Tuple[str, ...],
    ends_with: Tuple[str, ...],
    pattern_dirs: Dict[str, str],
    is_case_sensitive: bool,
) -> str:
    if not pattern_dirs:
        return default_dir

    def _cmp(a: str, b: str) -> bool:
        if is_case_sensitive:
            return a == b
        return a.lower() == b.lower()

    for prefix in starts_with:
        key = f"prefix:{prefix}"
        if key in pattern_dirs and _cmp(pubkey[: len(prefix)], prefix):
            return pattern_dirs[key]

    for suffix in ends_with:
        key = f"suffix:{suffix}"
        if key in pattern_dirs and _cmp(pubkey[-len(suffix) :], suffix):
            return pattern_dirs[key]

    return default_dir


def save_result(
    outputs: List,
    output_dir: str,
    starts_with: Tuple[str, ...] = (),
    ends_with: Tuple[str, ...] = (),
    pattern_dirs: Optional[Dict[str, str]] = None,
    is_case_sensitive: bool = True,
    quiet: bool = False,
) -> int:
    """
    Save results to disk. Returns count of NEW unique keys saved.
    Deduplication via in-memory set in save_keypair.
    """
    from core.utils.crypto import get_public_key_from_private_bytes, save_keypair, _seen_pubkeys

    before_count = len(_seen_pubkeys)
    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        target_dir = output_dir
        if pattern_dirs:
            pubkey = get_public_key_from_private_bytes(pv_bytes)
            target_dir = _resolve_output_dir(
                pubkey, output_dir, starts_with, ends_with,
                pattern_dirs, is_case_sensitive,
            )
        save_keypair(pv_bytes, target_dir, quiet=quiet)

    # Return actual NEW unique keys saved, not total processed
    new_unique = len(_seen_pubkeys) - before_count
    return new_unique
