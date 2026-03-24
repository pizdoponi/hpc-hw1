# Parallel Seam Carving using OpenMP

**Authors:** Nejc Krajšek, Rok Mušič  
**Date:** March 2026  
**Repository:** https://github.com/pizdoponi/hpc-hw1

## 1. Implementation Summary

The implementation follows the standard seam carving pipeline and we implemented three execution modes:

- Mode 0: sequential baseline.
- Mode 1: parallel energy computation + parallel image copying (during seam removal).
- Mode 2: mode 1 + basic per row parallel cumulative energy dynamic programming.

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
- *Mode 1 best / (32 th):* Fastest runtime for mode 1 across all thread counts; measures energy and seam copy parallelization impact.
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

**What improved performance.** Parallel energy computation is the primary source of speedup. Mode 1 achieves consistent S ≈ 10 across all image sizes (10.30–11.70 @ 32 threads), proving embarrassingly parallel stages scale well. Parallel seam copy removal adds further benefit. Mode 2's row wise dynamic programming parallelization becomes beneficial only on large images: at 7680×4320, mode 2 reaches S=9.41 vs mode 1's S=8.41 (1.119× faster), amortizing per row synchronization barriers.

**What did not improve and bottlenecks.** Mode 2 underperforms on small images despite more parallelism. At 1024×768 @ 32 threads, mode 2 runs 2.12× slower than mode 1 (0.395 s vs 0.186 s). Root cause: per row barriers repeat for every row; on small images, synchronization overhead dominates the work per row (~2000 operations at 1920×1200). Mode 2 at 7680×4320 achieves only 29% efficiency (9.41/32 = 0.294), indicating:

1. **Per row synchronization bottleneck:** The basic approach requires a barrier after each row (7680 barriers for 7680-pixel width). Mode 2 gains only 1.119× over mode 1 on the largest image despite this overhead.

2. **Memory bandwidth saturation:** Efficiency significantly drops above 16–32 threads. With 32 threads competing for memory, the bus cannot sustain CPU demand for energy reads.

3. **Threading overhead vs. work granularity:** At 720×480 @ 32 threads, each thread receives only 22.5 pixels per row—too small to overcome scheduling overhead. Mode 1 (minimal barriers) scales well; mode 2 (many barriers) does not.

**Expected behavior.** Embarrassingly parallel stages (energy, seam copy) scale consistently with Amdahl's law (S ≈ 10 at 32 threads). Synchronization heavy dynamic programming shows diminishing returns, especially on small images. Memory bandwidth saturation above 16–32 threads is expected on modern CPUs. Mode 2 outperforming mode 1 on the largest image is noteworthy: it shows row parallel dynamic programming can overcome synchronization overhead at sufficient workload size, motivating the triangle based approach for future optimization.

## 5. Conclusion

The implementation achieves substantial acceleration over sequential execution.

- Best overall speed-up observed: S=11.70 (1024x768, mode 1, 32 threads).
- For the largest image, the best assignment relevant parallel mode (mode 2) achieved S=9.41 at 32 threads.

Behavior matches expectations:

- embarrassingly parallel stages scale best,
- dynamic programming stage with row dependency has limited scalability in the basic parallel approach,
- larger images benefit more from additional parallelism than smaller ones.

