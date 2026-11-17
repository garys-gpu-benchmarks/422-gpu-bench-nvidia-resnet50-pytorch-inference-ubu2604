# Results: 422 ResNet-50 BF16 Inference Throughput Sweep

**Main goal:** Measure ResNet-50 BF16 inference throughput  
**Tools:** Run torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_145 · recorded 2026-11-17). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 NVL (95830 MiB) · AMD EPYC 9374F 32-Core Processor · Ubuntu 26.04.1 LTS · kernel 6.8.0-138-generic · driver 595.112.07
- **Run:** 2026-11-17T11:18:36Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:10 · benchmark 00:02 · whole run 01:23
- **Source:** run 422_20261117_111836
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype bf16 --image-size 64 --batch-size 1,2 --warmup-iters 1 --num-iterations 3 --use-channels-last true`

| # | Metric | Result |
|---|---|---|
| 1 | Largest-batch inference throughput, images/s (images_per_sec) | 624.033652 |
| 2 | Largest-batch per-image latency, ms (per_image_latency_msec) | 1.602478 |
| 3 | Largest-batch latency max, ms (batch_latency_max_msec) | 5.987328 |
| 4 | Largest-batch GPU utilization (gpu_utilization_percent) | 39.75 |
| 5 | Peak GPU memory at largest batch, GB (peak_gpu_memory_gb) | 0.092193 |
<!-- /profile:smoke -->
