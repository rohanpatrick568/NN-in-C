#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>

/* File Format Handling */
// MNIST files are in big-endian format
// This function converts to little-endian if necessary
uint32_t swap_uint32(uint32_t x) {
    return ((x >> 24) & 0xff) | ((x << 8) & 0xff0000) | ((x >> 8) & 0xff00) | ((x << 24) & 0xff000000);
}

/* MNIST Image Loading */
// Loads MNIST image file and converts to normalized double array
// Format: [magic number][number of images][number of rows][number of columns][pixel data]
double **load_mnist_images(const char *filename, int *num_images) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }

    uint32_t magic;
    fread(&magic, sizeof(magic), 1, file);
    magic = swap_uint32(magic);

    uint32_t n;
    fread(&n, sizeof(n), 1, file);
    *num_images = swap_uint32(n);

    uint32_t rows, cols;
    fread(&rows, sizeof(rows), 1, file);
    rows = swap_uint32(rows);
    fread(&cols, sizeof(cols), 1, file);
    cols = swap_uint32(cols);

    int image_size = rows * cols;
    double **images = (double **)malloc(*num_images * sizeof(double *));
    
    for (int i = 0; i < *num_images; i++) {
        images[i] = (double *)malloc(image_size * sizeof(double));
        unsigned char *pixels = (unsigned char *)malloc(image_size);
        fread(pixels, 1, image_size, file);
        for (int j = 0; j < image_size; j++) {
            images[i][j] = pixels[j] / 255.0;
        }
        free(pixels);
    }
    fclose(file);
    return images;
}

/* MNIST Label Loading */
// Loads MNIST label file
// Format: [magic number][number of labels][label data]
uint8_t *load_mnist_labels(const char *filename, int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }

    uint32_t magic;
    fread(&magic, sizeof(magic), 1, file);
    magic = swap_uint32(magic);

    uint32_t n;
    fread(&n, sizeof(n), 1, file);
    *num_labels = swap_uint32(n);

    uint8_t *labels = (uint8_t *)malloc(*num_labels);
    fread(labels, 1, *num_labels, file);
    fclose(file);
    return labels;
}

/* Data Shuffling */
// Fisher-Yates shuffle algorithm implementation
// Used to randomize training data order each epoch
void shuffle(int *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() % (n - i);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}