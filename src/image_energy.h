#ifndef IMAGE_ENERGY_H
#define IMAGE_ENERGY_H

#include <vector>

// Computes pixel energy at (x, y) using Sobel filter.
float compute_energy_pp(const float* image, int height, int width, int cpp, int x, int y);

// Computes energy for the entire image; result is indexed as [y * width + x].
std::vector<float> compute_energy(const float* image, int height, int width, int cpp);

#endif  // IMAGE_ENERGY_H
