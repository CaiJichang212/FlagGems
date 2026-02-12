import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES, BenchLevel
from benchmark.performance_utils import Benchmark, GenericBenchmark, vendor_name


class SVDBenchmark(Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        svd_shapes = [
            (8, 8),
            (16, 32),
            (32, 16),
            (64, 64),
            (128, 256),
            (256, 128),
            (512, 512),
            (1024, 1024),
        ]
        self.shapes = svd_shapes

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            M, N = shape
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.svd
def test_perf_svd():
    bench = SVDBenchmark(
        op_name="svd",
        torch_op=torch.svd,
        dtypes=[torch.float32],
    )
    bench.run()


class SVDNoUVBenchmark(Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        svd_shapes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
        ]
        self.shapes = svd_shapes

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            M, N = shape
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A, True, False)


@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.svd_no_uv
def test_perf_svd_no_uv():
    def svd_no_uv_wrapper(A, some=True, compute_uv=False):
        return torch.svd(A, some=some, compute_uv=compute_uv)
    
    bench = SVDNoUVBenchmark(
        op_name="svd_no_uv",
        torch_op=svd_no_uv_wrapper,
        dtypes=[torch.float32],
    )
    bench.run()
