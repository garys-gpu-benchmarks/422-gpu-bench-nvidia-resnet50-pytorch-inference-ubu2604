# ResNet-50 BF16 Inference Throughput Sweep Benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/garys-gpu-benchmarks/422-gpu-bench-nvidia-resnet50-pytorch-inference-ubu2604/actions/workflows/ci.yml/badge.svg)](https://github.com/garys-gpu-benchmarks/422-gpu-bench-nvidia-resnet50-pytorch-inference-ubu2604/actions/workflows/ci.yml)

Target: Ubuntu 26.04 · NVIDIA · see Hardware Requirements. This is a host benchmark, not a laptop `pip install` project.

## Quick Start

```bash
git clone https://github.com/garys-gpu-benchmarks/422-gpu-bench-nvidia-resnet50-pytorch-inference-ubu2604.git
cd 422-gpu-bench-nvidia-resnet50-pytorch-inference-ubu2604
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --profile smoke --validate
```
Results are written to `results/benchmark.db` and `results/summary.json`.

This workload is executed on the validation host after the repository is copied there. `setup.sh` and `run_benchmark.sh` do not open an outbound SSH session.

Prerequisites: Ubuntu 26.04; NVIDIA; Python 3.14.4; root or sudo for `setup.sh`. Framework: Bash, SQLite, Python, PyYAML, CUDA Runtime, PyTorch-CUDA, torchvision, ResNet-50. This is a host benchmark, not a laptop `pip install` project.

```mermaid
flowchart LR
  setup.sh --> run_benchmark.sh --> parse_results.py --> results/benchmark.db
```

## Download as a Bundle

This repository is one of the 32 workloads in the NVIDIA Ubuntu 26.04 bundle, [`bundle-nvidia-ubuntu-2604`](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604), which holds them as git submodules pinned to a release. To put every workload for this platform on a GPU host in one step, run this as root (or put `sudo` in front of the `apt-get` and `git clone` commands):

```bash
apt-get update && apt-get install -y git
git clone --recurse-submodules https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604 /opt/benchmarks \
  && cd /opt/benchmarks \
  && ./run.sh list \
  && cd /opt/benchmarks/422-gpu-bench-nvidia-resnet50-pytorch-inference-ubu2604 \
  && bash run_benchmark.sh --profile smoke --validate \
  && cd /opt/benchmarks \
  && echo "Done: 32 workloads in /opt/benchmarks; 422-gpu-bench-nvidia-resnet50-pytorch-inference-ubu2604 smoke run passed. Next: ./run_benchmark_suite.sh --help"
```

Every line in `./run.sh list` should show `ready`. The commands are joined with `&&`, so if the clone or the smoke run fails, the chain stops there, and the closing message appears only when everything succeeded. The chain ends in `/opt/benchmarks`, where the suite runner is. For more extensive testing:

```bash
./run_benchmark_suite.sh -p smoke                 # smoke test of all 32 workloads
./run_benchmark_suite.sh -w 407 -p baseline       # workload 407, baseline profile
./run_benchmark_suite.sh --help                   # all options
```

A workload that is not set up yet sets itself up on its first run (its `run_benchmark.sh` calls `scripts/ensure_setup.sh`). See [Suite Tools](#suite-tools) for more.

To move to the newest published release later: `cd /opt/benchmarks && git pull && git submodule update --init --recursive`.

| Bundle repository | Platform | Workloads |
|---|---|---|
| [`bundle-amd-ubuntu-2404`](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2404) | AMD · Ubuntu 24.04 | 101–132 |
| [`bundle-nvidia-ubuntu-2404`](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2404) | NVIDIA · Ubuntu 24.04 | 201–232 |
| [`bundle-amd-ubuntu-2604`](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2604) | AMD · Ubuntu 26.04 | 301–332 |
| [`bundle-nvidia-ubuntu-2604`](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604) | NVIDIA · Ubuntu 26.04 | 401–432 |

## Suite Tools

Three operator scripts from [`garys-gpu-benchmarks/gpu-bench-suite`](https://github.com/garys-gpu-benchmarks/gpu-bench-suite) work across the whole bundle. `run_benchmark_suite.sh` and `show_metrics.sh` are also at the top of every bundle (and the generator's remote copy installs `show_metrics.sh` at `/opt/benchmarks`); each script prints full usage with `--help`.

- **`run_benchmark_suite.sh`** (on the GPU host) — runs the installed workloads one after another for each profile, writes one log to `/opt/benchmarks/benchmark_suite_log/`, and ends with a pass/fail summary.
  ```bash
  cd /opt/benchmarks
  ./run_benchmark_suite.sh --dry-run                    # list what would run
  ./run_benchmark_suite.sh -w 422 -p smoke,baseline   # just this workload
  ./run_benchmark_suite.sh -p smoke                     # every workload, smoke profile
  ./run_benchmark_suite.sh -r 20                        # soak: everything, 20 times
  ```
- **`get_remote_info.sh`** (on your laptop; Git Bash on Windows) — over one SSH connection, copies the runtime ledger and every workload's `results/raw`, `results/parsed` and suite logs into `RUNTIME_LEDGERS/` and `DIRECTORIES/`.
  ```bash
  git clone https://github.com/garys-gpu-benchmarks/gpu-bench-suite && cd gpu-bench-suite
  ./scripts/get_remote_info.sh 203.0.113.10                     # root@host, port 22
  ./scripts/get_remote_info.sh -i ~/.ssh/id_ed25519 -o ~/results ubuntu@203.0.113.10 2222
  ```
- **`show_metrics.sh`** (laptop or GPU host) — prints one workload's runs from the runtime ledger in fixed-width metric columns with per-profile means. On the GPU host it is `/opt/benchmarks/show_metrics.sh`.
  ```bash
  cd /opt/benchmarks && ./show_metrics.sh 422             # on the GPU host
  ./scripts/show_metrics.sh 422 --host 203.0.113.10            # over SSH, all profiles
  ./scripts/show_metrics.sh 422 --baseline --file RUNTIME_LEDGERS/<ledger>.csv
  ```

## 1. Overview

Runs gpu-bench-resnet50-pytorch-inference.py with random ResNet-50 weights, synthetic images, and bf16 under torch.no_grad(). batch_size: comma-separated sweep. image_size, dtype/precision bf16, warmup_iters, and num_iterations come from yaml. This is a local inference sweep, not a five-minute server and not TorchBench Sweep dimensions: device_id, num_gpus, parallel_strategy, weights, dtype, precision, image_source, image_size.

## 2. What It Validates

- Validates one inference sweep over yaml batch sizes with random weights and synthetic images
- #1: Largest-batch inference throughput, images/s (images_per_sec); is present and physically sensible.
- #2: Largest-batch per-image latency, ms (per_image_latency_msec); is present and physically sensible.
- #3: Largest-batch p99 latency, ms (batch_p99_latency_msec); is present and physically sensible.
- #4: Largest-batch GPU utilization (gpu_utilization_percent); is present and physically sensible.
- #5: Peak GPU memory at largest batch, GB (peak_gpu_memory_gb) is present and physically sensible.

## 3. Metrics Captured

- **#1: Largest-batch inference throughput, images/s** — stored as `images_per_sec`.
- **#2: Largest-batch per-image latency, ms** — stored as `per_image_latency_msec`.
- **#3: Largest-batch p99 latency, ms** — stored as `batch_p99_latency_msec`.
- **#4: Largest-batch GPU utilization** — stored as `gpu_utilization_percent`.
- **#5: Peak GPU memory at largest batch, GB** — stored as `peak_gpu_memory_gb`.

## 4. Hardware Requirements

### Supported environment

- OS: Ubuntu 26.04
- GPU vendor: NVIDIA
- Framework family: Bash, SQLite, Python, PyYAML, CUDA Runtime, PyTorch-CUDA, torchvision, ResNet-50
- Python: Python 3.14.4

### Reference validation environment

The tables below describe the machine used to generate the reference results. They are not a requirement that every user buy that exact cloud instance.

### System

Runs gpu-bench-resnet50-pytorch-inference.py with random ResNet-50 weights, synthetic images, and bf16 under torch.no_grad(). batch_size: comma-separated sweep. image_size, dtype/precision bf16, warmup_iters, and num_iterations come from yaml.

### GPU

Ubuntu 26.04 / NVIDIA / Bash, SQLite, Python, PyYAML, CUDA Runtime, PyTorch-CUDA, torchvision, ResNet-50

## 5. Software Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 |
| Kernel | kernel 7.0.0 |
| Python | Python 3.14.4 |
| ROCm | CUDA 13.3 |
| rocBLAS | N/A - rocBLAS not used |

Runs gpu-bench-resnet50-pytorch-inference.py with random ResNet-50 weights, synthetic images, and bf16 under torch.no_grad(). batch_size: comma-separated sweep. image_size, dtype/precision bf16, warmup_iters, and num_iterations come from yaml.

## 6. Installation

```bash
Run gpu-bench-resnet50-pytorch-inference.py via collect_resnet50_infer.py; nvidia-smi is sampled on a background thread during the timed loop for GPU utilization, and peak memory comes from torch.cuda.max_memory_allocated
```

## 7. Running the Benchmark

```bash
Run gpu-bench-resnet50-pytorch-inference.py via collect_resnet50_infer.py; nvidia-smi is sampled on a background thread during the timed loop for GPU utilization, and peak memory comes from torch.cuda.max_memory_allocated
```

**Validating results separately:**

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional
python3 -m venv .venv
source ".venv/bin/activate"
".venv/bin/python" scripts/validate_results.py
```

## 8. Output

### `results/benchmark.db` (SQLite)

raw_results.csv with one row per batch size and a summary row. The summary copies the largest batch and also stores case_* copies

check,batch_size,headline_batch_size,dtype,status,images_per_sec,per_image_latency_msec,batch_p99_latency_msec,gpu_utilization_percent,peak_gpu_memory_gb
summary,32,32,bf16,ok,900,1.1,40,95,8

```bash
Run gpu-bench-resnet50-pytorch-inference.py via collect_resnet50_infer.py; nvidia-smi is sampled on a background thread during the timed loop for GPU utilization, and peak memory comes from torch.cuda.max_memory_allocated
```

### `results/summary.json`

Consolidated metrics from the most recent run — suitable for CI artifact upload or dashboard ingestion.

### `results/raw/<timestamp>.txt`

raw_results.csv with one row per batch size and a summary row. The summary copies the largest batch and also stores case_* copies

check,batch_size,headline_batch_size,dtype,status,images_per_sec,per_image_latency_msec,batch_p99_latency_msec,gpu_utilization_percent,peak_gpu_memory_gb
summary,32,32,bf16,ok,900,1.1,40,95,8

## 9. Baselines / Thresholds

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

## 10. Troubleshooting

**`setup.sh` missing collector**
Create cannot finish without `scripts/collect_workload.py`.

**`self_check` overlay rewritten**
Do not overwrite files listed in `results/overlay_lock.json`.

**Remote SSH drop during setup**
Reconnect and resume `bash setup.sh --assume-yes`. Do not wipe `.venv` or `.cache`.

## 11. NVIDIA H100 Coding Differences

Native NVIDIA CUDA workload. Execute on the stated Ubuntu release with the host NVIDIA driver and CUDA userspace. ROCm porting notes do not apply.

## Repository layout

```text
.
├── setup.sh
├── run_benchmark.sh
├── benchmark_specification.json
├── .github/workflows/      # thin CI callers (see Continuous Integration)
├── config/
├── scripts/
├── src/
├── tests/
├── docs/
├── results/
└── LICENSE
```

## Continuous Integration

| Workflow | Runs on | When | What it does |
|---|---|---|---|
| [CI](.github/workflows/ci.yml) | GitHub-hosted runner | every pull request, and every push to `main` | shellcheck, ruff, `bash -n`, `compileall`, `run_benchmark.sh --help`, specification schema, the results validator on a seeded fixture, required files, and actionlint. No GPU and no benchmark run. |
| [GPU Smoke Benchmark](.github/workflows/gpu-smoke.yml) | self-hosted runner labeled `gpu`, `nvidia`, `ubu2604` | only when started by hand: **Actions → GPU Smoke Benchmark → Run workflow** (choose `smoke`, `baseline` or `extended`) | Verifies the pre-provisioned GPU stack, records `results/environment.json` (driver, runtime, kernel, GPU), runs the profile with `--validate`, shows headline metrics on the run page, and uploads the results. |

Both files are short callers. The steps themselves live once, for every workload in the suite, in [`garys-gpu-benchmarks/shared-workflows`](https://github.com/garys-gpu-benchmarks/shared-workflows), pinned at `@v1`. The GPU workflow is never triggered by pull requests, so code from a fork cannot run on the GPU host.

The GPU Smoke Benchmark workflow runs one workload. To install and run the whole NVIDIA Ubuntu 26.04 bundle by hand on the same machines, see [Download as a Bundle](#download-as-a-bundle) and [Suite Tools](#suite-tools).
