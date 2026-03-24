# Parallel Seam Carving using OpenMP

**Authors:** Nejc Krajšek, Rok Mušič  
**Date:** March 2026  
**Repository:** https://github.com/pizdoponi/hpc-hw1

## 1. Implementation Summary

The implementation follows the standard seam-carving pipeline and supports three working execution modes:

- Mode 0: sequential baseline.
- Mode 1: parallel energy computation + parallel seam-copy removal.
- Mode 2: mode 1 + basic per-row parallel cumulative-energy dynamic programming.

The benchmarked configuration uses:

- compiler flags: `-O3 -fopenmp`
- seam count: 128 per run
- repetitions: 5 runs per configuration, averaged
- image sizes: 720x480, 1024x768, 1920x1200, 3840x2160, 7680x4320
- thread counts: 1, 2, 4, 8, 16, 32
- cluster execution with `--hint=nomultithread` and OpenMP pinning (`OMP_PLACES=cores`, `OMP_PROC_BIND=close`)

## 2. Speed-up Definition

Speed-up is computed as: `S = t_s / t_p`.

where:

- t_s is the sequential runtime (mode 0, 1 thread) for the same image size,
- t_p is the runtime of the measured parallel configuration.

## 3. Measured Results

Results were collected in [src/benchmark_results.csv](benchmark_results.csv).

### 3.1 Best observed runtime per image and mode

| Image size | Mode 0 (baseline) | Mode 1 best | Mode 2 best |
|---|---:|---:|---:|
| 720x480 | 0.899 s (1 th) | 0.087 s (32 th), S=10.30 | 0.176 s (16 th), S=5.11 |
| 1024x768 | 2.174 s (1 th) | 0.186 s (32 th), S=11.70 | 0.362 s (16 th), S=6.01 |
| 1920x1200 | 6.624 s (1 th) | 0.639 s (32 th), S=10.37 | 0.890 s (32 th), S=7.44 |
| 3840x2160 | 25.088 s (1 th) | 2.950 s (32 th), S=8.50 | 2.865 s (32 th), S=8.76 |
| 7680x4320 | 102.025 s (1 th) | 12.132 s (32 th), S=8.41 | 10.837 s (32 th), S=9.41 |

- *Image size:* Resolution of the input image (width × height).
- *Mode 0 (baseline):* Sequential reference runtime, runs only on a single thread by design.; serves as t_s in speed-up calculation S = t_s / t_p.
- *Mode 1 best / (32 th):* Fastest runtime for mode 1 across all thread counts; measures energy and seam-copy parallelization impact.
- *Mode 2 best / (32 th):* Fastest runtime for mode 2 across all thread counts; includes cumulative-energy dynamic programming parallelization.
- *S (speed-up):* Relative performance improvement S = t_s / t_p; values > 1 indicate acceleration.

### 3.2 Mode 2 vs Mode 1 (same thread count)

For small and medium images (up to 1920x1200), mode 2 is usually not faster than mode 1 at higher thread counts.

Examples:

- 1024x768 @ 32 threads: mode2/mode1 speed ratio = 0.504 (mode 2 slower).
- 1920x1200 @ 32 threads: mode2/mode1 speed ratio = 0.717 (mode 2 slower).

For larger images, mode 2 starts helping:

- 3840x2160 @ 16 threads: mode2/mode1 speed ratio = 1.081 (mode 2 faster).
- 7680x4320 @ 32 threads: mode2/mode1 speed ratio = 1.119 (mode 2 faster).

## 4. Discussion

### 4.1 What improved performance

**Parallel energy computation is the primary source of speedup.** Energy computation is embarrassingly parallel (each pixel's energy is independent), and scaling measurements demonstrate this clearly. Mode 1 achieves consistent, high speedup across all image sizes:
- 720x480: S=10.30 @ 32 threads
- 1024x768: S=11.70 @ 32 threads (highest overall speedup)
- 1920x1200: S=10.37 @ 32 threads
- 3840x2160: S=8.50 @ 32 threads
- 7680x4320: S=8.41 @ 32 threads

Even on smaller images (720x480), mode 1 achieves over 10× speedup at 32 threads, proving that loop-level parallelization of energy is effective. The scaling remains robust because there are no cross-pixel dependencies in energy calculation.

**Parallel seam-copy removal provides additional benefit within mode 1.** After the seam is identified, removing pixels from each row is also embarrassingly parallel with no cross-row dependencies. This stage alone accounts for a portion of mode 1's speedup. The combined effect (energy + seam-copy parallelism) yields the high observed speedups in mode 1.

**On large images, cumulative-energy dynamic programming parallelization (mode 2) becomes beneficial.** At 3840x2160, mode 2 achieves S=8.76 (faster than mode 1's S=8.50 at the same thread count). At the largest image (7680x4320), mode 2 reaches S=9.41 vs mode 1's S=8.41, a 1.119× improvement. This shows that larger workloads per row amortize the cost of per-row synchronization barriers in the basic parallel dynamic programming approach. The effect is visible in Table 3.2: at 3840x2160 @ 16 threads, mode 2 is 1.081× faster than mode 1, and at 7680x4320 @ 32 threads, mode 2 is 1.119× faster.

### 4.2 What did not improve (or improved little)

**Mode 2 underperforms on small images despite parallelization.** This behavior directly contradicts a naive expectation that "more parallelism = always better." On 1024x768:
- Mode 1 @ 32 threads: 0.186 s
- Mode 2 @ 32 threads: 0.395 s (2.12× slower)

At 720x480 @ 16 threads:
- Mode 1: 0.107 s
- Mode 2: 0.176 s (1.65× slower)

Root cause: On small images (≤ 1920×1200), per-row dynamic programming work is modest. Each row requires only ~2000 operations at 1920×1200. Introducing per-row barriers means synchronization overhead repeats for every row, and this overhead per unit of work is proportionally larger than on big images. For 720×480, this effect is severe, dominating the theoretical benefit of parallelizing each row.

**Parallel dynamic programming does not scale linearly.** Mode 2 at 7680×4320 achieves S=9.41 at 32 threads, corresponding to an efficiency of only 29% (9.41/32 = 0.294). This is because:
1. Energy computation and seam-copy are already parallelized in mode 1, capturing most of the easily available parallelism.
2. Dynamic programming introduces per-row synchronization barriers, creating a sequential bottleneck: threads must synchronize after computing each row.
3. Memory bandwidth becomes a limiting factor at high thread counts; all threads compete for memory access.

**Mode 0 at 1 thread runs slightly slower than expected.** The OpenMP runtime itself introduces small overhead even at 1 thread (thread team initialization, etc.), which explains why parallel code at t=1 is sometimes marginally slower than sequential mode 0. However, this overhead is small (< 5%) and is immediately amortized on larger thread counts.

### 4.3 Bottlenecks and Scalability Limits

**Primary bottleneck: per-row synchronization in dynamic programming.** The basic approach to parallel cumulative-energy calculation parallelizes work within a single row, but rows depend on the previous row's results. This requires a barrier after each row—yielding 7680 barriers for a 7680-pixel-wide image. At 32 threads, this synchronization cost becomes visible; mode 2 gains only 1.119× over mode 1 on the largest image, despite adding a new parallelization layer.

Measured evidence shows this bottleneck directly: mode 2 loses to mode 1 on all small/medium images (where per-row work is small relative to sync cost) but only wins on large images (where the per-row work grows and eventually absorbs the overhead).

**Secondary bottleneck: memory bandwidth.** Scaling efficiency drops noticeably above 16–32 threads:
- Mode 1 at 1920×1200: best efficiency is typically achieved at 16 threads.
- Mode 1 at 7680×4320: efficiency still improves at 32 threads, but at a slower rate.

This indicates that memory bus saturation limits speedup. With 32 threads, energy reads consume bandwidth proportional to thread count, and the main memory system cannot keep pace with CPU demand.

**Tertiary bottleneck: OpenMP runtime and scheduling overhead.** At 32 threads on small images like 720×480, there is an imbalance: each thread receives only ~90 columns of work (720/32 = 22.5 pixels per thread). At this granularity, thread scheduling overhead and synchronization points become significant relative to useful work per thread. This is why mode 0 (no overhead at all) and mode 1 (minimal barrier overhead) achieve good speedup on small images, but introducing more synchronization (mode 2) hurts performance.

### 4.4 Expected vs. Unexpected Behaviour

**Expected:** Embarrassingly parallel stages (energy, seam-copy) scale well with threads on all image sizes (observed S ≈ 10 at 32 threads). This aligns with Amdahl's law for a problem with large parallel fraction.

**Expected:** Synchronization-heavy stages (row-wise dynamic programming) show diminishing returns at high thread counts, especially on small images. The per-row barrier cost is unavoidable in the basic approach.

**Expected:** Efficiency drops above 16–32 threads due to memory bandwidth saturation. Modern CPUs have limited memory bandwidth per core; with 32 threads competing, bandwidth per thread becomes a limiting resource.

**Possibly unexpected:** Mode 2 actually outperforms mode 1 on the largest images (7680×4320: 1.119× faster), showing that for sufficiently large workloads, row-parallel dynamic programming can overcome its synchronization overhead. This is good news for very large image sizes and motivates the triangle-based approach suggested in the assignment as a future improvement.

## 5. Conclusion

The implementation achieves substantial acceleration over sequential execution.

- Best overall speed-up observed: S=11.70 (1024x768, mode 1, 32 threads).
- For the largest image, the best assignment-relevant parallel mode (mode 2) achieved S=9.41 at 32 threads.

Behavior matches expectations:

- embarrassingly parallel stages scale best,
- dynamic programming stage with row dependency has limited scalability in the basic parallel approach,
- larger images benefit more from additional parallelism than smaller ones.