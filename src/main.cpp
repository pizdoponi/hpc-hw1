#include <vector>
#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <omp.h>
#include <sched.h>
#include <numa.h> // NOLINT

#include "seam_dp.h"
#include "image_energy.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255


int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::printf("USAGE: main input_image output_image [<n_seams_to_remove> = 128] [<parallel_level> = 0]\n");
        std::exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];


    std::snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    std::snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    // Set the number of seams to remove.
    int n_seams_to_remove = 128;
    if (argc >= 4) {
        n_seams_to_remove = std::atoi(argv[3]);
    }

    // Determine the level of parallelism.
    // 0 = no parallelism
    // 1 = parallel energy computation
    // 2 = parallel energy computation + per row parallel cumulative energy computation
    // 3 = parallel energy computation + pyramid method parallel cumulative energy computation
    int parallel = 0;
    if (argc >= 5) {
        parallel = std::atoi(argv[4]);
    }


    // Load image from file and allocate space for the output image
    int width, height, cpp; // cpp = channels per pixel
    float *image_in = stbi_loadf(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        std::printf("Error reading loading image %s!\n", image_in_name);
        std::exit(EXIT_FAILURE);
    }
    std::printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, width, height, cpp);

    double start = omp_get_wtime();

    // ── do the work ─────────────────────────────────────────────────────
    for (int i = 0; i < n_seams_to_remove; i++) {
        std::vector<float> image_energy =
            compute_energy(image_in, height, width, cpp, parallel > 0);

        std::vector<float> cumulative_energy;
        if (parallel == 3) {
            exit(1); // Not yet implemented.
        } else {
            cumulative_energy = compute_cumulative_energy_bottom_up(
                image_energy, width, height, parallel == 2);
        }

        std::vector<int> seam_to_remove = find_vertical_seam_top_down(cumulative_energy, width, height);
        remove_seam(image_in, width, height, cpp, seam_to_remove, SeamDirection::Vertical, parallel > 0);
    }

    // Because we modify the image in place, the output is the last step.
    float *image_out = image_in;
    // ──────────────────────────────────────────────────────────────────────

    double stop = omp_get_wtime();

    std::printf("Time to remove %d seams: %f s\n", n_seams_to_remove, stop - start);

    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(cpp);
    
    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    std::strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char *file_type = std::strrchr(image_out_name, '.');
    if (file_type == NULL) {
        std::printf("Error: No file extension found!\n");
        std::free(image_in);
        std::exit(EXIT_FAILURE);
    }
    file_type++; // skip the dot

    if (!std::strcmp(file_type, "hdr")) {
        stbi_write_hdr(image_out_name, width, height, cpp, image_out);
    } else if (!std::strcmp(file_type, "png") || !std::strcmp(file_type, "jpg") || !std::strcmp(file_type, "bmp")) {
        unsigned char *image_out_u8 = static_cast<unsigned char *>(std::malloc(pixel_count * sizeof(unsigned char)));
        if (image_out_u8 == NULL) {
            std::printf("Error: Failed to allocate memory for 8-bit output image!\n");
            std::free(image_in);
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
    std::free(image_in);

    return 0;
}
