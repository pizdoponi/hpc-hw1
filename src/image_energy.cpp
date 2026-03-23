#include "image_energy.h"

#include <cstddef>
#include <cmath>
#include <omp.h>

float compute_energy_pp(const float* image, int height, int width, int cpp, int x, int y)
{
    // [[tl, tc, tr],
    //  [ml, mc, mr],
    //  [bl, bc, br]]
    //  tl = top left, tc = top center, mr = middle right, etc.
    //  Convetion used for naming neighbouring cells.
    //
    //  xminus1 et. al. clamps to the closest pixel value (replicate edge).
    int xminus1 = (x == 0) ? 0 : x - 1; // If we are in the first column (i.e. x == 0), we cannot move further left.
    int xplus1 = (x == width - 1) ? width - 1 : x + 1;
    int yminus1 = (y == 0) ? 0 : y - 1;
    int yplus1 = (y == height - 1) ? height - 1 : y + 1;

    int mc_idx = (y * width + x) * cpp;
    int ml_idx = (y * width + xminus1) * cpp;
    int mr_idx = (y * width + xplus1) * cpp;
    int tc_idx = (yminus1 * width + x) * cpp;
    int tl_idx = (yminus1 * width + xminus1) * cpp;
    int tr_idx = (yminus1 * width + xplus1) * cpp;
    int bc_idx = (yplus1 * width + x) * cpp;
    int bl_idx = (yplus1 * width + xminus1) * cpp;
    int br_idx = (yplus1 * width + xplus1) * cpp;

    float energy_sum = 0.0f;

    for (int c = 0; c < cpp; ++c) {
        float tl = image[tl_idx + c];
        float tc = image[tc_idx + c];
        float tr = image[tr_idx + c];
        float ml = image[ml_idx + c];
        float mc = image[mc_idx + c];
        float mr = image[mr_idx + c];
        float bl = image[bl_idx + c];
        float bc = image[bc_idx + c];
        float br = image[br_idx + c];

        float G_x = -tl - 2.0f * ml - bl + tr + 2.0f * mr + br;
        float G_y =  tl + 2.0f * tc + tr - bl - 2.0f * bc - br;

        float E = std::sqrt(G_x * G_x + G_y * G_y);
        energy_sum += E;
    }

    return energy_sum / static_cast<float>(cpp);
}

std::vector<float> compute_energy(const float* image, int height, int width, int cpp)
{
    std::vector<float> energies(static_cast<std::size_t>(height) * width, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            energies[idx] = compute_energy_pp(image, height, width, cpp, x, y);
        }
    }

    return energies;
}
