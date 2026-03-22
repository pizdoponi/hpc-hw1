#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cmath>
#include <omp.h>
#include <sched.h>
#include <numa.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

void copy_image(float *image_out, const float *image_in, const std::size_t size)
{
    
    #pragma omp parallel
    {
        // Print thread, CPU, and NUMA node information
        #pragma omp single
        std::printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);

        #pragma omp critical
        std::printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);

        // Copy the image data in parallel
        #pragma omp for
        for (size_t i = 0; i < size; ++i)
        {
            image_out[i] = image_in[i];
        }
    }
    
}

/**
 * Computes pixel energy per pixel at pixel (x, y).
 *
 * @param image A pointer to an array of floats ordered in HWC
 * @param height height
 * @param width width
 * ...
*/
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

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::printf("USAGE: main input_image output_image\n");
        std::exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];


    std::snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    std::snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp; // cpp = channels per pixel
    float *image_in = stbi_loadf(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        std::printf("Error reading loading image %s!\n", image_in_name);
        std::exit(EXIT_FAILURE);
    }
    std::printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, width, height, cpp);

    // Allocate space for the output image.
    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(cpp);
    float *image_out = static_cast<float *>(std::malloc(pixel_count * sizeof(float)));
    if (image_out == NULL) {
        std::printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        std::exit(EXIT_FAILURE);
    }

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    // TODO: replace this with the actual computation
    copy_image(image_out, image_in, pixel_count);
    double stop = omp_get_wtime();
    std::printf("Time to copy: %f s\n", stop - start);
    
    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    std::strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char *file_type = std::strrchr(image_out_name, '.');
    if (file_type == NULL) {
        std::printf("Error: No file extension found!\n");
        stbi_image_free(image_in);
        std::free(image_out);
        std::exit(EXIT_FAILURE);
    }
    file_type++; // skip the dot

    if (!std::strcmp(file_type, "hdr")) {
        stbi_write_hdr(image_out_name, width, height, cpp, image_out);
    } else if (!std::strcmp(file_type, "png") || !std::strcmp(file_type, "jpg") || !std::strcmp(file_type, "bmp")) {
        unsigned char *image_out_u8 = static_cast<unsigned char *>(std::malloc(pixel_count * sizeof(unsigned char)));
        if (image_out_u8 == NULL) {
            std::printf("Error: Failed to allocate memory for 8-bit output image!\n");
            stbi_image_free(image_in);
            std::free(image_out);
            std::exit(EXIT_FAILURE);
        }
        for (std::size_t i = 0; i < pixel_count; ++i) {
            float v = image_out[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            image_out_u8[i] = static_cast<unsigned char>(v * 255.0f + 0.5f);
        }

        if (!std::strcmp(file_type, "png"))
            stbi_write_png(image_out_name, width, height, cpp, image_out_u8, width * cpp);
        else if (!std::strcmp(file_type, "jpg"))
            stbi_write_jpg(image_out_name, width, height, cpp, image_out_u8, 100);
        else
            stbi_write_bmp(image_out_name, width, height, cpp, image_out_u8);

        std::free(image_out_u8);
    } else {
        std::printf("Error: Unknown image format %s! Only png, jpg, bmp, or hdr supported.\n", file_type);
    }

    // Release the memory
    stbi_image_free(image_in);
    std::free(image_out);

    return 0;
}
