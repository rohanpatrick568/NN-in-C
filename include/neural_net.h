#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Network Architecture Parameters */
#define INPUT_SIZE 784    // 28x28 input image
#define HIDDEN_SIZE 128   // Number of neurons in hidden layer
#define OUTPUT_SIZE 10    // Number of possible digits (0-9)

/* Training Hyperparameters */
#define LEARNING_RATE 0.01  // Step size for gradient descent
#define EPOCHS 100          // Number of complete passes through the training data
#define TRAIN_SAMPLES 60000 // Size of MNIST training set
#define TEST_SAMPLES 10000  // Size of MNIST test set
#define BATCH_SIZE 64       // Number of samples per mini-batch
#define CACHE_LINE 64       // CPU cache line size for memory alignment

/* Utility Macros */
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* Core Neural Network Functions */
// Initialize network parameters using He initialization
void initialize_network(double *hidden_weights, double *hidden_bias, double *output_weights, double *output_bias);

// Forward propagation for single sample
void forward_pass(double *input, double *hidden, double *output, 
    double *hidden_weights, double *hidden_bias, 
    double *output_weights, double *output_bias);

// Forward propagation for mini-batch
void forward_pass_batch(double *input_batch, double *hidden_batch, double *output_batch, 
    const double *hidden_weights, const double *hidden_bias, 
    const double *output_weights, const double *output_bias, int batch_size);

// Backward propagation for single sample
void backward_pass(double *input, uint8_t label, double *hidden, double *output,
    double *hidden_weights, double *hidden_bias,
    double *output_weights, double *output_bias);

// Backward propagation for mini-batch
void backward_pass_batch(double *input_batch, uint8_t *labels_batch, 
    double *hidden_batch, double *output_batch,
    double *hidden_weights, double *hidden_bias, 
    double *output_weights, double *output_bias, int batch_size);

/* Loss and Evaluation Functions */
// Calculate cross-entropy loss for single prediction
double cross_entropy_loss(double *output, uint8_t label);

// Get the predicted digit (0-9)
int predict(double *output);

// Calculate classification accuracy on test set
double calculate_accuracy(double **test_images, uint8_t *test_labels,
    int num_samples, double *hidden_weights, double *hidden_bias,
    double *output_weights, double *output_bias);

/* Activation Functions */
// Rectified Linear Unit activation
double relu(double x);

// Derivative of ReLU for backpropagation
double relu_derivative(double x);

// Softmax activation for output layer
void softmax(double *x, int size);

#endif