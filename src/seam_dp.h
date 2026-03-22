#ifndef SEAM_DP_H
#define SEAM_DP_H

#include <vector>

enum class SeamDirection {
    Vertical,
    Horizontal
};

// Compute cumulative energy from bottom to top using dynamic programming.
// Returns a vector of cumulative energies; use as M[i * width + j].
std::vector<float> compute_cumulative_energy_bottom_up(
    const std::vector<float>& energy, int width, int height);

// Find the vertical seam (path from top to bottom) with minimum cumulative energy.
// Returns a vector of column indices, indexed by row.
std::vector<int> find_vertical_seam_top_down(
    const std::vector<float>& cumulative, int width, int height);

// Remove a seam from the image by allocating a new (temporary) buffer.
// The seam is a vector obtained with one of the methods above,
// for SeamDirection::Vertical, size=height (column per row);
// for Horizontal, size=width (row per column).
// Updates image pointer and dimensions; frees old buffer.
void remove_seam(float*& image, int& width, int& height, int cpp,
    const std::vector<int>& seam, SeamDirection direction);

#endif  // SEAM_DP_H
