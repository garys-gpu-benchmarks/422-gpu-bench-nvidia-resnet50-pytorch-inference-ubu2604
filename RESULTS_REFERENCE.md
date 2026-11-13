# Results: 422 ResNet-50 BF16 Inference Throughput Sweep

**Main goal:** Measure ResNet-50 BF16 inference throughput  
**Tools:** Run torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_138 · recorded 2026-11-12). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 80GB HBM3 (81559 MiB) · Intel(R) Xeon(R) Platinum 8470 · Ubuntu 26.04.1 LTS · kernel 6.8.0-147-generic · driver 595.91.07
- **Run:** 2026-11-12T23:21:02Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:33 · benchmark 00:06 · whole run 01:53
- **Source:** run 422_20261112_232102
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype bf16 --image-size 64 --batch-size 1,2 --warmup-iters 1 --num-iterations 3 --use-channels-last true`

| # | Metric | Result |
|---|---|---|
| 1 | Largest-batch inference throughput, images/s (images_per_sec) | 535.596562 |
| 2 | Largest-batch per-image latency, ms (per_image_latency_msec) | 1.867077 |
| 3 | Largest-batch latency max, ms (batch_latency_max_msec) | 4.762848 |
| 4 | Largest-batch GPU utilization (gpu_utilization_percent) | 29.5 |
| 5 | Peak GPU memory at largest batch, GB (peak_gpu_memory_gb) | 0.092193 |
<!-- /profile:smoke -->
