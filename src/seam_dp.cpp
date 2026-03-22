#include "seam_dp.h"
#include <algorithm>
#include <vector>

std::vector<float> compute_cumulative_energy_bottom_up(
    const std::vector<float>& energy, int width, int height) {
    std::vector<float> cumulative(static_cast<size_t>(width) * height, 0.0f);

    for (int j = 0; j < width; ++j) {
        cumulative[static_cast<size_t>(height - 1) * width + j] =
            energy[static_cast<size_t>(height - 1) * width + j];
    }

    for (int i = height - 2; i >= 0; --i) {
        for (int j = 0; j < width; ++j) {
            float best = cumulative[static_cast<size_t>(i + 1) * width + j];
            if (j > 0) {
                best = std::min(best, cumulative[static_cast<size_t>(i + 1) * width + (j - 1)]);
            }
            if (j + 1 < width) {
                best = std::min(best, cumulative[static_cast<size_t>(i + 1) * width + (j + 1)]);
            }
            cumulative[static_cast<size_t>(i) * width + j] = energy[static_cast<size_t>(i) * width + j] + best;
        }
    }

    return cumulative;
}

std::vector<int> find_vertical_seam_top_down(
    const std::vector<float>& cumulative, int width, int height) {
    std::vector<int> seam(height, 0);

    int best_j = 0;
    float best_val = cumulative[0];
    for (int j = 1; j < width; ++j) {
        const float v = cumulative[j];
        if (v < best_val) {
            best_val = v;
            best_j = j;
        }
    }
    seam[0] = best_j;

    for (int i = 1; i < height; ++i) {
        const int prev_j = seam[i - 1];

        int next_j = prev_j;
        float next_v = cumulative[static_cast<size_t>(i) * width + prev_j];

        if (prev_j > 0) {
            const float left_v = cumulative[static_cast<size_t>(i) * width + (prev_j - 1)];
            if (left_v < next_v) {
                next_v = left_v;
                next_j = prev_j - 1;
            }
        }

        if (prev_j + 1 < width) {
            const float right_v = cumulative[static_cast<size_t>(i) * width + (prev_j + 1)];
            if (right_v < next_v) {
                next_j = prev_j + 1;
            }
        }

        seam[i] = next_j;
    }

    return seam;
}
