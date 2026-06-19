from __future__ import annotations

from experiments.ten_son_cpu_gpu_benchmark import run_gpu_steady
from experiments.ten_son_training_common import benchmark_configuration, generate_host_batches


def test_timed_gpu_mode_has_no_host_readback_and_resets():
    _,_,config=benchmark_configuration("development")
    batches=generate_host_batches(config,5,778)
    result,_,_,valid,_=run_gpu_steady("development",batches,steps=1,repetitions=1)
    assert result["device_to_host_bytes_per_step"]==0
    assert result["host_to_device_bytes_per_step"]==0
    assert result["kernel_launches_per_step"]>0
    assert valid
