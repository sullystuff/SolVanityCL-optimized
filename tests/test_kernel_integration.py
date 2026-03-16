import unittest

from base58 import b58encode
from nacl.signing import SigningKey
import pyopencl as cl

from core.config import HostSetting
from core.searcher import Searcher
from core.utils.helpers import load_kernel_source


class TestKernelIntegration(unittest.TestCase):
    def _get_first_gpu_selection(self):
        try:
            platforms = cl.get_platforms()
        except Exception:
            return None

        for p_index, platform in enumerate(platforms):
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
            except Exception:
                continue
            if devices:
                return p_index, [0]
        return None

    def test_kernel_output_matches_pynacl(self) -> None:
        selection = self._get_first_gpu_selection()
        if selection is None:
            self.skipTest("No OpenCL GPU devices available")

        seed = bytes(range(1, 33))
        pubkey = b58encode(bytes(SigningKey(seed).verify_key)).decode()
        kernel_source = load_kernel_source((pubkey[:2],), (pubkey[-2:],), True)

        setting = HostSetting(kernel_source, iteration_bits=0)
        setting.local_work_size = 1
        setting.key32 = bytearray(seed)

        searcher = Searcher(
            kernel_source=kernel_source,
            index=0,
            setting=setting,
            chosen_devices=selection,
        )
        cl.enqueue_copy(searcher.command_queue, searcher.memobj_output, bytearray(33)).wait()
        result = searcher.find(log_stats=False)

        self.assertEqual(result[0], len(pubkey))
        self.assertEqual(bytes(result[1:33]), seed)


if __name__ == "__main__":
    unittest.main()
