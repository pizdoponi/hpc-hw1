#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
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

void copy_image(unsigned char *image_out, const unsigned char *image_in, const std::size_t size)
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

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        std::printf("USAGE: sample input_image output_image\n");
        std::exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    std::snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    std::snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        std::printf("Error reading loading image %s!\n", image_in_name);
        std::exit(EXIT_FAILURE);
    }
    std::printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, width, height, cpp);
    const std::size_t datasize = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(cpp) * sizeof(unsigned char);
    unsigned char *image_out = static_cast<unsigned char *>(std::malloc(datasize));
    if (image_out == NULL) {
        std::printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        std::exit(EXIT_FAILURE);
    }

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    copy_image(image_out, image_in, datasize);
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

    if (!std::strcmp(file_type, "png"))
        stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);
    else if (!std::strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!std::strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
    else
        std::printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);
    std::free(image_out);

    return 0;
}
