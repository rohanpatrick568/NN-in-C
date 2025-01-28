#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdint.h>
#include <stddef.h>

/* MNIST Data Loading Functions */
// Load and normalize MNIST image data
// Returns: Array of image data with pixel values normalized to [0,1]
// num_images: Output parameter for number of images loaded
double **load_mnist_images(const char *filename, int *num_images);

// Load MNIST label data
// Returns: Array of label values (0-9)
// num_labels: Output parameter for number of labels loaded
uint8_t *load_mnist_labels(const char *filename, int *num_labels);

/* Utility Functions */
// Convert big-endian to little-endian for MNIST file format
uint32_t swap_uint32(uint32_t x);

// Random shuffle implementation for training data
// array: Array to shuffle
// n: Length of array
void shuffle(int *array, size_t n);

#endif