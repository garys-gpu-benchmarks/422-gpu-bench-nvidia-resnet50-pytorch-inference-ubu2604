# Results: 422 ResNet-50 BF16 Inference Throughput Sweep

**Main goal:** Measure ResNet-50 BF16 inference throughput  
**Tools:** Run torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_140 · recorded 2026-11-13). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 PCIe (81559 MiB) · Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz · Ubuntu 26.04.1 LTS · kernel 6.8.0-138-generic · driver 595.91.07
- **Run:** 2026-11-13T21:01:11Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 02:04 · benchmark 00:09 · whole run 02:34
- **Source:** run 422_20261113_210111
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype bf16 --image-size 64 --batch-size 1,2 --warmup-iters 1 --num-iterations 3 --use-channels-last true`

| # | Metric | Result |
|---|---|---|
| 1 | Largest-batch inference throughput, images/s (images_per_sec) | 295.078675 |
| 2 | Largest-batch per-image latency, ms (per_image_latency_msec) | 3.388927 |
| 3 | Largest-batch latency max, ms (batch_latency_max_msec) | 9.566944 |
| 4 | Largest-batch GPU utilization (gpu_utilization_percent) | 16.857143 |
| 5 | Peak GPU memory at largest batch, GB (peak_gpu_memory_gb) | 0.092193 |
<!-- /profile:smoke -->
