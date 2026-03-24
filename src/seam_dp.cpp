#include "seam_dp.h"
#include <algorithm>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <omp.h>

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

std::vector<float> compute_cumulative_energy_bottom_up_parallel(
    const std::vector<float>& energy, int width, int height) {
    std::vector<float> cumulative(static_cast<size_t>(width) * height, 0.0f);

    // Copy the energy into the cumulative buffer.
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < width; ++j) {
        cumulative[static_cast<size_t>(height - 1) * width + j] =
            energy[static_cast<size_t>(height - 1) * width + j];
    }

    #pragma omp parallel
    for (int i = height - 2; i >= 0; --i) {
        #pragma omp for schedule(static)
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

void remove_seam(float*& image, int& width, int& height, int cpp,
    const std::vector<int>& seam, SeamDirection direction) {
    if (image == nullptr || width <= 1 || height <= 1 || cpp <= 0) {
        return;
    }

    if (direction == SeamDirection::Vertical) {
        if (static_cast<int>(seam.size()) != height) {
            std::printf("[debug] remove_seam: seam size %zu does not match height %d\n", seam.size(), height);
            return;
        }
        const int new_width = width - 1;
        const std::size_t new_count = static_cast<std::size_t>(new_width) * height * cpp;
        float *new_image = static_cast<float *>(std::malloc(new_count * sizeof(float)));
        if (new_image == nullptr) {
            return;
        }

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++) {
            const int seam_x = seam[static_cast<std::size_t>(y)];
            if (seam_x < 0 || seam_x >= width) {
                std::printf("[debug] remove_seam: seam_x out of bounds y=%d seam_x=%d width=%d\n", y, seam_x, width);
                continue;
            }
            for (int x = 0; x < new_width; x++) {
                const int src_x = (x < seam_x) ? x : x + 1;
                const std::size_t dst_base = (static_cast<std::size_t>(y) * new_width + x) * cpp;
                const std::size_t src_base = (static_cast<std::size_t>(y) * width + src_x) * cpp;
                for (int c = 0; c < cpp; ++c) {
                    new_image[dst_base + c] = image[src_base + c];
                }
            }
        }

        std::free(image);
        image = new_image;
        width = new_width;
        return;
    }

    // else direction == SeamDirection::Horizontal
    if (static_cast<int>(seam.size()) != width) {
        std::printf("[debug] remove_seam: seam size %zu does not match width %d\n", seam.size(), width);
        return;
    }
    const int new_height = height - 1;
    const std::size_t new_count = static_cast<std::size_t>(width) * new_height * cpp;
    float *new_image = static_cast<float *>(std::malloc(new_count * sizeof(float)));
    if (new_image == nullptr) {
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int x = 0; x < width; x++) {
        const int seam_y = seam[static_cast<std::size_t>(x)];
        if (seam_y < 0 || seam_y >= height) {
            std::printf("[debug] remove_seam: seam_y out of bounds x=%d seam_y=%d height=%d\n", x, seam_y, height);
            continue;
        }
        for (int y = 0; y < new_height; y++) {
            const int src_y = (y < seam_y) ? y : y + 1;
            const std::size_t dst_base = (static_cast<std::size_t>(y) * width + x) * cpp;
            const std::size_t src_base = (static_cast<std::size_t>(src_y) * width + x) * cpp;
            for (int c = 0; c < cpp; ++c) {
                new_image[dst_base + c] = image[src_base + c];
            }
        }
    }

    std::free(image);
    image = new_image;
    height = new_height;
}
