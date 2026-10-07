# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

Runs gpu-bench-resnet50-pytorch-inference.py with random ResNet-50 weights, synthetic images, and bf16 under torch.no_grad(). batch_size: comma-separated sweep. image_size, dtype/precision bf16, warmup_iters, and num_iterations come from yaml. This is a local inference sweep, not a five-minute server and not TorchBench Sweep dimensions: device_id, num_gpus, parallel_strategy, weights, dtype, precision, image_source, image_size.

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| device_id | `--device-id` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| num_gpus | `--num-gpus` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| parallel_strategy | `--parallel-strategy` | TBD — confirm against tool documentation | TBD — confirm against tool documentation | From Parameter list; see Execution Description With Parameters. |
| weights | `--weights` | smoke=random, baseline=random, extended=random | random | From Parameter list; see Execution Description With Parameters. |
| dtype | `--dtype` | smoke=bf16, baseline=bf16, extended=bf16 | bf16 | From Parameter list; see Execution Description With Parameters. |
| precision | `--precision` | smoke=bf16, baseline=bf16, extended=bf16 | bf16 | From Parameter list; see Execution Description With Parameters. |
| image_source | `--image-source` | smoke=synthetic, baseline=synthetic, extended=synthetic | synthetic | From Parameter list; see Execution Description With Parameters. |
| image_size | `--image-size` | smoke=64, baseline=224, extended=224 | 224 | From Parameter list; see Execution Description With Parameters. |
| use_channels_last | `--use-channels-last` | smoke=true, baseline=true, extended=true | true | From Parameter list; see Execution Description With Parameters. |
| batch_size | `--batch-size` | smoke=1,2, baseline=1,2,4,8,16,32, extended=1,2,4,8,16,32,64,128 | 1,2,4,8,16,32 | From Parameter list; see Execution Description With Parameters. |
| num_workers | `--num-workers` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| warmup_iters | `--warmup-iters` | smoke=1, baseline=5, extended=10 | 5 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=3, baseline=13350, extended=16200 | 13350 | From Parameter list; see Execution Description With Parameters. |

## Invocation

```bash
Run gpu-bench-resnet50-pytorch-inference.py via collect_resnet50_infer.py; nvidia-smi is sampled on a background thread during the timed loop for GPU utilization, and peak memory comes from torch.cuda.max_memory_allocated
```

## Raw Output Format

raw_results.csv with one row per batch size and a summary row. The summary copies the largest batch and also stores case_* copies

check,batch_size,headline_batch_size,dtype,status,images_per_sec,per_image_latency_msec,batch_p99_latency_msec,gpu_utilization_percent,peak_gpu_memory_gb
summary,32,32,bf16,ok,900,1.1,40,95,8

## Metrics

- **#1: Largest-batch inference throughput, images/s** — stored as `images_per_sec`.
- **#2: Largest-batch per-image latency, ms** — stored as `per_image_latency_msec`.
- **#3: Largest-batch p99 latency, ms** — stored as `batch_p99_latency_msec`.
- **#4: Largest-batch GPU utilization** — stored as `gpu_utilization_percent`.
- **#5: Peak GPU memory at largest batch, GB** — stored as `peak_gpu_memory_gb`.

## Framework

Runs gpu-bench-resnet50-pytorch-inference.py with random ResNet-50 weights, synthetic images, and bf16 under torch.no_grad(). batch_size: comma-separated sweep. image_size, dtype/precision bf16, warmup_iters, and num_iterations come from yaml.

## Installation and Execution Summary

Run gpu-bench-resnet50-pytorch-inference.py once over the yaml batch_size list using torchvision ResNet-50 with random weights, synthetic bf16 images, and torch.no_grad(), emit one RESULT line per batch plus a summary, to measure inference images/s and latency. This is not a five-minute server loop

## Platform Portability

- **AMD (primary):** ```bash
Run gpu-bench-resnet50-pytorch-inference.py via collect_resnet50_infer.py; nvidia-smi is sampled on a background thread during the timed loop for GPU utilization, and peak memory comes from torch.cuda.max_memory_allocated
```
- **NVIDIA:** Native NVIDIA CUDA workload. Execute on the stated Ubuntu release with the host NVIDIA driver and CUDA userspace. ROCm porting notes do not apply.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

raw_results.csv with one row per batch size and a summary row. The summary copies the largest batch and also stores case_* copies

check,batch_size,headline_batch_size,dtype,status,images_per_sec,per_image_latency_msec,batch_p99_latency_msec,gpu_utilization_percent,peak_gpu_memory_gb
summary,32,32,bf16,ok,900,1.1,40,95,8

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Runs gpu-bench-resnet50-pytorch-inference.py with random ResNet-50 weights, synthetic images, and bf16 under torch.no_grad(). batch_size: comma-separated sweep. image_size, dtype/precision bf16, warmup_iters, and num_iterations come from yaml.
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Runs gpu-bench-resnet50-pytorch-inference.py with random ResNet-50 weights, synthetic images, and bf16 under torch.no_grad(). batch_size: comma-separated sweep. image_size, dtype/precision bf16, warmup_iters, and num_iterations come from yaml.

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
