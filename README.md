# Assignment 5: Optimising Attention on a CPU

CS6886: Systems for Deep Learning, IIT Madras. Anurag Dhiman.

This repository speeds up a naive C implementation of single-head attention on an Apple M4,
`O = softmax(Q Kᵀ / √d) V`. At N = 4096 and d = 64, the time drops from 1,084 ms to 6.56 ms on
one thread (165× faster), and to 2.61 ms with OpenMP (416× faster). Every variant passes
`make check`. No BLAS is used: the only libraries linked are the C library and, for `omp`, the
OpenMP runtime.

The full answers to Q1–Q5, with the measurements behind every claim, are in
[`report.pdf`](report.pdf).

## Variants

Times are at N = 4096, d = 64, from [`results.csv`](results.csv). Speedup is relative to naive.

| Variant | File | Main technique | Time (ms) | GFLOP/s | Speedup | What limits it |
|---|---|---|---:|---:|---:|---|
| naive | [`attn_naive.c`](src/attn_naive.c) | The given code, unchanged | 1083.5 | 3.96 | 1× | Chains of dependent floating-point operations |
| loops | [`attn_loops.c`](src/attn_loops.c) | Loop reordering so that clang vectorises; K is transposed once | 168.4 | 25.5 | 6.4× | L2 bandwidth; accumulators kept in memory |
| tiled | [`attn_tiled.c`](src/attn_tiled.c) | NEON register tiles of 16 accumulators, plus cache tiling | 67.4 | 63.7 | 16.1× | Compute; the softmax, using libm `expf`, takes 30% of the time |
| fused | [`attn_fused.c`](src/attn_fused.c) | Row by row: the N × N score matrix is never stored | 116.3 | 36.9 | 9.3× | L2 → L1 bandwidth, since K and V are re-read for every row |
| best | [`attn_best.c`](src/attn_best.c) | SME2 outer products (FMOPA) for both matrix products, NEON softmax with a polynomial `exp`, fused over 16-row blocks | 6.56 | 655 | 165× | SME instruction issue, plus the softmax |
| omp | [`attn_omp.c`](src/attn_omp.c) | `best` parallelised over 16-row blocks, `schedule(dynamic, 1)`, 10 threads | 2.61 | 1648 | 416× | One SME unit per core cluster, shared by its cores |

The driver code (`main`) in every file is identical to the starter's. Only `attention()` and the
functions it calls were changed.

## Requirements

| What | Why |
|---|---|
| A Mac with SME2, such as the Apple M4 | `best` and `omp` use SME2 instructions and do not build or run without them. `tiled` and `fused` need only NEON (any 64-bit Arm CPU), and `naive` and `loops` are plain C. |
| Homebrew LLVM: `brew install llvm` | The [`Makefile`](Makefile) sets `CC = /opt/homebrew/opt/llvm/bin/clang`. Apple's Xcode clang does not support `-fopenmp` on its own; Homebrew's LLVM includes the OpenMP runtime (`libomp`). |
| Python 3 | Used by `make check`. |

These results were measured on a MacBook Air with an Apple M4 (4 performance and 6 efficiency
cores, 16 GB), running macOS 15.5 and Homebrew clang 23.1.4. The full specification is in
[`machine_spec.txt`](machine_spec.txt).

## Reproducing the results

```sh
brew install llvm
git clone https://github.com/Anurag9Dhiman/Systems-Engineering-Assignment-5.git
cd Systems-Engineering-Assignment-5

make all      # build every variant and time it on every input in input/
make check    # compare each variant's output with naive's
```

| Command | What it does |
|---|---|
| `make <variant>` | Builds and times one variant: `naive`, `loops`, `tiled`, `fused`, `best` or `omp`. |
| `make check` | Passes when `max\|O − O_naive\| / max\|O_naive\| ≤ 1e-4`. It needs naive's output, so run `make naive` or `make all` first. |
| `OMP_NUM_THREADS=4 make omp` | Runs `omp` with a given thread count. `results.csv` has 1, 2, 3, 4, 5, 6, 8 and 10 threads. |
| `make clean` | Deletes `build/` and `output/`. |

A run prints one line per input:

```text
$ make best
best               N=1024  d=64   threads=1   time_ms=     0.555  GFLOP/s=  483.67
best               N=2048  d=64   threads=1   time_ms=     1.910  GFLOP/s=  562.17
best               N=4096  d=128  threads=1   time_ms=     9.492  GFLOP/s=  904.97
best               N=4096  d=64   threads=1   time_ms=     6.942  GFLOP/s=  618.69

$ make check
...
qkv_N4096_d64.bin    best         PASS  (error 5.9e-06)
qkv_N4096_d64.bin    omp          PASS  (error 5.9e-06)
```

### How the numbers were measured

- **What each run reports.** The driver calls `attention()` once untimed, then times 5 calls
  and prints the median.
- **The clock warm-up.** macOS takes about 100 ms to raise the clock speed after a process
  starts. To avoid timing that ramp, each optimised variant runs itself for 0.2 s on its first
  call. To turn this off, build with
  `make best CFLAGS="-O3 -march=native -DNO_CLOCK_WARMUP"`. Table 2 of the report compares the
  two.
- **The rows in `results.csv`.** Each row is the median of 5 separate runs of the `make` target.
  The arithmetic intensity column is FLOPs divided by simulated DRAM traffic; the method is in Q5
  of the report.
- **Run-to-run variation.** Times vary by several percent between runs. In the sample above,
  `best` took 6.94 ms at N = 4096, d = 64, against 6.56 ms in `results.csv`.

The microbenchmarks, cache simulator and plotting scripts used for the report are not part of
this repository, since the handout specifies its contents. The report describes how they work: its
Microbenchmarks section covers the microbenchmarks, and Q5 covers the cache simulation.

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src/) | `attn_naive.c` (given) and the five variants above |
| [`input/`](input/) | The given inputs: `qkv_N<N>_d<d>.bin`, a header of two int32 values, N and d, then Q, K and V as float32 |
| [`machine_spec.txt`](machine_spec.txt) | Q1: the machine, its peak FLOP/s, memory bandwidth and ridge points |
| [`results.csv`](results.csv) | Time, GFLOP/s, arithmetic intensity and `make check` result for every variant, input and thread count |
| [`plots/roofline.png`](plots/roofline.png) | Q5: every variant on the roofline |
| [`plots/time_N<N>_d<d>.png`](plots/) | The time of every variant, one plot per input |
| [`report.pdf`](report.pdf) | Answers to Q1–Q5 |
| [`Makefile`](Makefile) | Build, run and check targets |
| [`assignment_handout.pdf`](assignment_handout.pdf) | The assignment |

## Report contents

| Section | Topic |
|---|---|
| Q1 | The machine: specification, measured clock speeds, FMA units and bandwidth, peaks and ridge points |
| Q2 | The naive baseline, and why it reaches only 2.8% of the NEON peak |
| Q3 | The four optimised variants, with phase timings and tile-size sweeps |
| Q4 | OpenMP scaling across the performance and efficiency cores |
| Q5 | The roofline: arithmetic intensity, and what limits each variant |
