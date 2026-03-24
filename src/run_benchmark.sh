#!/bin/bash

#SBATCH --job-name=seam_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=seam_bench_%j.log
#SBATCH --hint=nomultithread
# SBATCH --reservation=fri

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
fi

# Keep thread placement fixed for better reproducibility.
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module load numactl

echo "Compiling with -O3 and OpenMP"
g++ -O3 -fopenmp main.cpp seam_dp.cpp image_energy.cpp -lm -lnuma -o main

SIZES=(720x480 1024x768 1920x1200 3840x2160 7680x4320)
THREADS=(1 2 4 8 16 32)
REPEATS=5
SEAMS=128
RESULTS_CSV="benchmark_results.csv"
OUTPUT_DIR="bench_outputs"
RESUME_LOG="${RESUME_LOG:-}"
CSV_HEADER="image_size,mode,threads,repeats,seams_requested,seams_reported,avg_runtime_s,speedup_vs_mode0_t1,speedup_mode2_vs_mode1_same_threads"

mkdir -p "$OUTPUT_DIR"
if [[ -f "$RESULTS_CSV" ]] && [[ "$(head -n 1 "$RESULTS_CSV" 2>/dev/null || true)" == "$CSV_HEADER" ]]; then
    echo "Resuming with existing $RESULTS_CSV"
else
    echo "$CSV_HEADER" > "$RESULTS_CSV"
fi

echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "OMP_PLACES=${OMP_PLACES}"
echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
echo "Repeats=${REPEATS}, seams per run=${SEAMS}"
if [[ -n "$RESUME_LOG" ]]; then
    echo "Resume log: $RESUME_LOG"
fi

declare -A mode1_avg_by_size_thread
declare -A baseline_by_size
declare -A completed_cfg
declare -A resume_count
declare -A resume_sum

# Load already completed configurations from the CSV to avoid reruns.
while IFS=, read -r size mode threads repeats seams_req seams_rep avg_runtime speedup0 speedup21; do
    if [[ "$size" == "image_size" ]] || [[ -z "$size" ]]; then
        continue
    fi
    key="${size}|${mode}|${threads}"
    completed_cfg["$key"]=1
    if [[ "$mode" == "0" && "$threads" == "1" ]]; then
        baseline_by_size["$size"]="$avg_runtime"
    fi
    if [[ "$mode" == "1" ]]; then
        mode1_avg_by_size_thread["${size}|${threads}"]="$avg_runtime"
    fi
done < "$RESULTS_CSV"

# Optionally parse a prior log and continue partially finished configurations.
if [[ -n "$RESUME_LOG" ]]; then
    if [[ ! -f "$RESUME_LOG" ]]; then
        echo "ERROR: RESUME_LOG file not found: $RESUME_LOG" >&2
        exit 1
    fi

    while IFS=$'\t' read -r key count sum; do
        if [[ -n "$key" ]]; then
            resume_count["$key"]="$count"
            resume_sum["$key"]="$sum"
        fi
    done < <(
        awk '
            match($0, /^size=([^ ]+) mode=([0-9]+) threads=([0-9]+) run=([0-9]+)\/[0-9]+ runtime=([0-9.]+)s$/, m) {
                key = m[1] "|" m[2] "|" m[3]
                run = m[4] + 0
                seen_key = key SUBSEP run
                if (!(seen_key in seen)) {
                    seen[seen_key] = 1
                    count[key] += 1
                    sum[key] += m[5]
                }
            }
            END {
                for (k in count) {
                    printf "%s\t%d\t%.9f\n", k, count[k], sum[k]
                }
            }
        ' "$RESUME_LOG"
    )
fi

detect_seams_reported() {
    local run_output="$1"
    awk '
        match($0, /Time to remove ([0-9]+) seams:/, m) {
            print m[1]
        }
    ' <<< "$run_output" | tail -n 1
}

detect_runtime() {
    local run_output="$1"
    awk '
        match($0, /Time to remove [0-9]+ seams: ([0-9.]+) s/, m) {
            print m[1]
        }
    ' <<< "$run_output" | tail -n 1
}

run_once() {
    local input_image="$1"
    local output_image="$2"
    local seams="$3"
    local mode="$4"

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --hint=nomultithread \
            ./main "$input_image" "$output_image" "$seams" "$mode"
    else
        ./main "$input_image" "$output_image" "$seams" "$mode"
    fi
}

for size in "${SIZES[@]}"; do
    input_image="test_images/${size}.png"
    if [[ ! -f "$input_image" ]]; then
        echo "Skipping missing image: $input_image"
        continue
    fi

    echo
    echo "=== ${size} ==="
    baseline_avg="${baseline_by_size["$size"]:-}"

    for mode in 0 1 2; do
        if [[ "$mode" -eq 0 ]]; then
            mode_threads=(1)
        else
            mode_threads=("${THREADS[@]}")
        fi

        for threads in "${mode_threads[@]}"; do
            if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && (( threads > SLURM_CPUS_PER_TASK )); then
                echo "Skipping mode=${mode} threads=${threads}, exceeds SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
                continue
            fi

            cfg_key="${size}|${mode}|${threads}"
            if [[ -n "${completed_cfg["$cfg_key"]:-}" ]]; then
                echo "Skipping already completed config: size=${size} mode=${mode} threads=${threads}"
                continue
            fi

            export OMP_NUM_THREADS="$threads"
            resumed_runs="${resume_count["$cfg_key"]:-0}"
            sum_runtime="${resume_sum["$cfg_key"]:-0}"
            start_run=$((resumed_runs + 1))
            seams_reported=""
            if (( resumed_runs > 0 )); then
                echo "Resuming size=${size} mode=${mode} threads=${threads} from run ${start_run}/${REPEATS}"
            fi

            for run_id in $(seq "$start_run" "$REPEATS"); do
                output_image="${OUTPUT_DIR}/out_${size}_m${mode}_t${threads}_r${run_id}.png"
                run_output="$(run_once "$input_image" "$output_image" "$SEAMS" "$mode")"

                runtime="$(detect_runtime "$run_output")"
                reported="$(detect_seams_reported "$run_output")"

                if [[ -z "$runtime" ]]; then
                    echo "Failed to parse runtime for size=${size}, mode=${mode}, threads=${threads}, run=${run_id}" >&2
                    echo "$run_output" >&2
                    exit 1
                fi

                if [[ -z "$seams_reported" ]]; then
                    seams_reported="$reported"
                    if [[ "$seams_reported" != "$SEAMS" ]]; then
                        echo "WARNING: binary reports removing ${seams_reported} seams, requested ${SEAMS}."
                        echo "WARNING: update main.cpp to accept and use argv[3] for exact assignment-compliant runs."
                    fi
                fi

                sum_runtime="$(awk -v a="$sum_runtime" -v b="$runtime" 'BEGIN {printf "%.9f", a + b}')"
                echo "size=${size} mode=${mode} threads=${threads} run=${run_id}/${REPEATS} runtime=${runtime}s"
            done

            if [[ -z "$seams_reported" ]]; then
                seams_reported="$SEAMS"
            fi

            avg_runtime="$(awk -v s="$sum_runtime" -v n="$REPEATS" 'BEGIN {printf "%.9f", s / n}')"

            if [[ "$mode" -eq 0 && "$threads" -eq 1 ]]; then
                baseline_avg="$avg_runtime"
            fi

            speedup_vs_mode0=""
            if [[ -n "$baseline_avg" ]]; then
                speedup_vs_mode0="$(awk -v ts="$baseline_avg" -v tp="$avg_runtime" 'BEGIN {if (tp == 0) print "inf"; else printf "%.6f", ts / tp}')"
            fi

            speedup_mode2_vs_mode1=""
            if [[ "$mode" -eq 1 ]]; then
                mode1_avg_by_size_thread["${size}|${threads}"]="$avg_runtime"
            elif [[ "$mode" -eq 2 ]]; then
                mode1_ref="${mode1_avg_by_size_thread["${size}|${threads}"]:-}"
                if [[ -n "$mode1_ref" ]]; then
                    speedup_mode2_vs_mode1="$(awk -v t1="$mode1_ref" -v t2="$avg_runtime" 'BEGIN {if (t2 == 0) print "inf"; else printf "%.6f", t1 / t2}')"
                fi
            fi

            echo "${size},${mode},${threads},${REPEATS},${SEAMS},${seams_reported},${avg_runtime},${speedup_vs_mode0},${speedup_mode2_vs_mode1}" >> "$RESULTS_CSV"
            echo "avg(size=${size}, mode=${mode}, threads=${threads})=${avg_runtime}s, speedup_vs_mode0_t1=${speedup_vs_mode0}, speedup_mode2_vs_mode1=${speedup_mode2_vs_mode1}"

            completed_cfg["$cfg_key"]=1
            if [[ "$mode" -eq 0 && "$threads" -eq 1 ]]; then
                baseline_by_size["$size"]="$avg_runtime"
            fi
        done
    done
done

echo
echo "Done. Results saved to ${RESULTS_CSV}."
