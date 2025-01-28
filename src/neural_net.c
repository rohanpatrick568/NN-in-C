#include "neural_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Initialize network weights and biases
void initialize_network(double *hidden_weights, double *hidden_bias, double *output_weights, double *output_bias) {
    srand(time(NULL));
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_weights[i * HIDDEN_SIZE + j] = (rand() / (double)RAND_MAX) * sqrt(2.0 / INPUT_SIZE);
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) hidden_bias[j] = 0.0;

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            output_weights[j * OUTPUT_SIZE + k] = (rand() / (double)RAND_MAX) * sqrt(2.0 / HIDDEN_SIZE);
        }
    }
    for (int k = 0; k < OUTPUT_SIZE; k++) output_bias[k] = 0.0;
}

// Forward pass through the network
void forward_pass(double *input, double *hidden, double *output, double *hidden_weights, double *hidden_bias, double *output_weights, double *output_bias) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double activation = hidden_bias[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            activation += input[i] * hidden_weights[i * HIDDEN_SIZE + j];
        }
        hidden[j] = relu(activation);
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output[k] = output_bias[k];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[k] += hidden[j] * output_weights[j * OUTPUT_SIZE + k];
        }
    }
    softmax(output, OUTPUT_SIZE);
}

// Forward pass through the network for a batch
void forward_pass_batch(double *input_batch, double *hidden_batch, double *output_batch,
    const double *hidden_weights, const double *hidden_bias,
    const double *output_weights, const double *output_bias, int batch_size) {
    
    for (int b = 0; b < batch_size; b++) {
        // Hidden layer
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double activation = hidden_bias[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                activation += input_batch[b * INPUT_SIZE + i] * hidden_weights[i * HIDDEN_SIZE + j];
            }
            hidden_batch[b * HIDDEN_SIZE + j] = relu(activation);
        }

        // Output layer
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            double sum = output_bias[k];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += hidden_batch[b * HIDDEN_SIZE + j] * output_weights[j * OUTPUT_SIZE + k];
            }
            output_batch[b * OUTPUT_SIZE + k] = sum;
        }
        
        softmax(&output_batch[b * OUTPUT_SIZE], OUTPUT_SIZE);
    }
}

// Backward pass (gradient descent)
void backward_pass(double *input, uint8_t label, double *hidden, double *output, double *hidden_weights, double *hidden_bias, double *output_weights, double *output_bias) {
    double delta_output[OUTPUT_SIZE];
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        delta_output[k] = output[k] - (k == label ? 1.0 : 0.0);
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            output_weights[j * OUTPUT_SIZE + k] -= LEARNING_RATE * delta_output[k] * hidden[j];
        }
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output_bias[k] -= LEARNING_RATE * delta_output[k];
    }

    double delta_hidden[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double error = 0.0;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            error += delta_output[k] * output_weights[j * OUTPUT_SIZE + k];
        }
        delta_hidden[j] = error * relu_derivative(hidden[j]);
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_weights[i * HIDDEN_SIZE + j] -= LEARNING_RATE * delta_hidden[j] * input[i];
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        hidden_bias[j] -= LEARNING_RATE * delta_hidden[j];
    }
}

// Backward pass (gradient descent) for a batch
void backward_pass_batch(double *input_batch, uint8_t *labels_batch,
    double *hidden_batch, double *output_batch,
    double *hidden_weights, double *hidden_bias,
    double *output_weights, double *output_bias, int batch_size) {
    
    double *delta_output = (double *)malloc(batch_size * OUTPUT_SIZE * sizeof(double));
    double *delta_hidden = (double *)malloc(batch_size * HIDDEN_SIZE * sizeof(double));

    // Calculate output deltas
    for (int b = 0; b < batch_size; b++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            delta_output[b * OUTPUT_SIZE + k] = output_batch[b * OUTPUT_SIZE + k] - 
                (k == labels_batch[b] ? 1.0 : 0.0);
        }
    }

    // Update output weights and biases
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            double weight_update = 0.0;
            for (int b = 0; b < batch_size; b++) {
                weight_update += delta_output[b * OUTPUT_SIZE + k] * 
                    hidden_batch[b * HIDDEN_SIZE + j];
            }
            output_weights[j * OUTPUT_SIZE + k] -= (LEARNING_RATE / batch_size) * weight_update;
        }
    }

    // Update output biases
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        double bias_update = 0.0;
        for (int b = 0; b < batch_size; b++) {
            bias_update += delta_output[b * OUTPUT_SIZE + k];
        }
        output_bias[k] -= (LEARNING_RATE / batch_size) * bias_update;
    }

    // Calculate hidden deltas
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double error = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                error += delta_output[b * OUTPUT_SIZE + k] * output_weights[j * OUTPUT_SIZE + k];
            }
            delta_hidden[b * HIDDEN_SIZE + j] = error * relu_derivative(hidden_batch[b * HIDDEN_SIZE + j]);
        }
    }

    // Update hidden weights and biases
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double weight_update = 0.0;
            for (int b = 0; b < batch_size; b++) {
                weight_update += delta_hidden[b * HIDDEN_SIZE + j] * input_batch[b * INPUT_SIZE + i];
            }
            hidden_weights[i * HIDDEN_SIZE + j] -= (LEARNING_RATE / batch_size) * weight_update;
        }
    }

    // Update hidden biases
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double bias_update = 0.0;
        for (int b = 0; b < batch_size; b++) {
            bias_update += delta_hidden[b * HIDDEN_SIZE + j];
        }
        hidden_bias[j] -= (LEARNING_RATE / batch_size) * bias_update;
    }

    free(delta_output);
    free(delta_hidden);
}

// Cross-entropy loss
double cross_entropy_loss(double *output, uint8_t label) {
    return -log(output[label]);
}

// Predict the class with the highest probability
int predict(double *output) {
    int max_idx = 0;
    double max_val = output[0];
    for (int k = 1; k < OUTPUT_SIZE; k++) {
        if (output[k] > max_val) {
            max_val = output[k];
            max_idx = k;
        }
    }
    return max_idx;
}

// Calculate accuracy on test data
double calculate_accuracy(double **test_images, uint8_t *test_labels, int num_samples, double *hidden_weights, double *hidden_bias, double *output_weights, double *output_bias) {
    int correct = 0;
    double *hidden = (double *)malloc(HIDDEN_SIZE * sizeof(double));
    double *output = (double *)malloc(OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < num_samples; i++) {
        forward_pass(test_images[i], hidden, output, hidden_weights, hidden_bias, output_weights, output_bias);
        if (predict(output) == test_labels[i]) correct++;
    }

    free(hidden);
    free(output);
    return (double)correct / num_samples;
}

// ReLU activation
double relu(double x) {
    return x > 0 ? x : 0;
}

// ReLU derivative
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Softmax activation
void softmax(double *x, int size) {
    double max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}