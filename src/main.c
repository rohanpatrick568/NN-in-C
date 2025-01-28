#include "neural_net.h"
#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
    srand(time(NULL));

    int train_samples, test_samples;
    double **train_images = load_mnist_images("data/train-images.idx3-ubyte", &train_samples);
    uint8_t *train_labels = load_mnist_labels("data/train-labels.idx1-ubyte", &train_samples);
    double **test_images = load_mnist_images("data/t10k-images.idx3-ubyte", &test_samples);
    uint8_t *test_labels = load_mnist_labels("data/t10k-labels.idx1-ubyte", &test_samples);

    double *hidden_weights = (double *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double *hidden_bias = (double *)malloc(HIDDEN_SIZE * sizeof(double));
    double *output_weights = (double *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
    double *output_bias = (double *)malloc(OUTPUT_SIZE * sizeof(double));

    initialize_network(hidden_weights, hidden_bias, output_weights, output_bias);

    int *indices = (int *)malloc(TRAIN_SAMPLES * sizeof(int));
    for (int i = 0; i < TRAIN_SAMPLES; i++) indices[i] = i;

    double *hidden = (double *)malloc(HIDDEN_SIZE * sizeof(double));
    double *output = (double *)malloc(OUTPUT_SIZE * sizeof(double));

    double *batch_input = (double *)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(double));
    double *batch_hidden = (double *)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
    double *batch_output = (double *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    uint8_t *batch_labels = (uint8_t *)malloc(BATCH_SIZE * sizeof(uint8_t));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle(indices, TRAIN_SAMPLES);
        double total_loss = 0.0;

        for (int i = 0; i < TRAIN_SAMPLES; i += BATCH_SIZE) {
            int current_batch_size = MIN(BATCH_SIZE, TRAIN_SAMPLES - i);
            
            // Prepare batch
            for (int b = 0; b < current_batch_size; b++) {
                int idx = indices[i + b];
                memcpy(&batch_input[b * INPUT_SIZE], train_images[idx], INPUT_SIZE * sizeof(double));
                batch_labels[b] = train_labels[idx];
            }

            forward_pass_batch(batch_input, batch_hidden, batch_output,
                hidden_weights, hidden_bias, output_weights, output_bias, 
                current_batch_size);

            // Calculate batch loss
            for (int b = 0; b < current_batch_size; b++) {
                total_loss += cross_entropy_loss(&batch_output[b * OUTPUT_SIZE], batch_labels[b]);
            }

            backward_pass_batch(batch_input, batch_labels, batch_hidden, batch_output,
                hidden_weights, hidden_bias, output_weights, output_bias,
                current_batch_size);
        }

        double accuracy = calculate_accuracy(test_images, test_labels, TEST_SAMPLES, hidden_weights, hidden_bias, output_weights, output_bias);
        printf("Epoch %d, Loss: %.4f, Accuracy: %.2f%%\n", epoch + 1, total_loss / TRAIN_SAMPLES, accuracy * 100);
    }

    // Cleanup
    free(hidden_weights); free(hidden_bias); free(output_weights); free(output_bias);
    free(hidden); free(output); free(indices);
    free(batch_input); free(batch_hidden); free(batch_output); free(batch_labels);
    for (int i = 0; i < TRAIN_SAMPLES; i++) free(train_images[i]);
    for (int i = 0; i < TEST_SAMPLES; i++) free(test_images[i]);
    free(train_images); free(test_images); free(train_labels); free(test_labels);

    return 0;
}