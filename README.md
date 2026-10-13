# Neural Network MNIST Classifier in C

A from-scratch implementation of a feedforward neural network to classify handwritten digits from the MNIST dataset.

## Project Structure

```
NN-in-C/
│
├── bin/                    # Compiled executable (created by build.bat)
├── data/                   # MNIST dataset files (download separately)
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
│
├── include/                # Header files
│   ├── neural_net.h       # Neural network definitions
│   └── data_loader.h      # MNIST data loading functions
│
└── src/                    # Source files
    ├── main.c             # Main program
    ├── neural_net.c       # Neural network implementation
    └── data_loader.c      # MNIST data loading implementation
```

## Features

- Two-layer feedforward neural network
- ReLU activation for hidden layer
- Softmax activation for output layer
- Mini-batch gradient descent
- Cross-entropy loss function
- MNIST dataset support

## Technical Specifications

- Input layer: 784 neurons (28x28 pixels)
- Hidden layer: 64 neurons
- Output layer: 10 neurons (digits 0-9)
- Batch size: 64
- Learning rate: 0.01
- Training epochs: 100
- Training samples: 60,000
- Test samples: 10,000

## Build Instructions

1. Ensure you have GCC installed
2. Place MNIST dataset files in the `data/` directory (see Dataset Setup below)
3. Run the build script:
   ```bash
   build.bat
   ```
4. The executable will be created in the `bin/` directory

## MNIST Dataset Format

- Images are 28x28 pixels in size
- Pixel values are normalized to [0,1]
- Labels are single digits (0-9)
- Files use idx format (big-endian)
  - Images: 4 bytes magic + 4 bytes count + 4 bytes rows + 4 bytes cols + pixels
  - Labels: 4 bytes magic + 4 bytes count + labels

## Implementation Details

### Neural Network (neural_net.c)
- Weight initialization using He initialization
- Forward propagation with batched processing
- Backward propagation with gradient descent
- ReLU and Softmax activation functions
- Cross-entropy loss calculation
- Accuracy evaluation

### Data Loading (data_loader.c)
- MNIST format parsing
- Endianness handling
- Data normalization
- Dataset shuffling

### Main Program (main.c)
- Dataset loading
- Training loop implementation
- Batch processing
- Progress monitoring
- Memory management
- Interactive testing interface
  - ASCII visualization of test digits
  - Real-time predictions
  - Confidence scores for all digits
  - User-selected test samples

## Usage

The program first trains the network, printing loss and accuracy after each epoch:

```
Epoch 1, Loss: ..., Accuracy: ...%
```

It then switches to interactive testing: enter a test-set index (0-9999) to see that digit drawn in ASCII, the network's prediction and its confidence for each digit, or `q` to quit.

## Dependencies

- Standard C libraries
- Math library (-lm)
- MinGW GCC compiler (Windows)

## Performance

The network typically achieves:
- Training time: ~2-3 minutes
- Final accuracy: ~95-96% on test set
- Memory usage: ~100MB

## Memory Management

The program handles memory allocation for:
- Network weights and biases
- Input/hidden/output layers
- Batch processing buffers
- Dataset storage

All allocations are properly freed at program termination.

## Getting Started

### Prerequisites
- GCC compiler (MinGW for Windows)
- Git
- ~100MB free disk space
- 1GB RAM minimum

### Dataset Setup
1. Create a `data/` directory in the project root
2. Download the four MNIST files (training and test images and labels) from any MNIST mirror
3. Extract the `.gz` files into `data/` so the names match: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`

### Quick Start
```bash
# Clone the repository
git clone https://github.com/rohanpatrick568/NN-in-C.git

# Navigate to project directory
cd NN-in-C

# Download and prepare the MNIST dataset (see Dataset Setup above)

# Build the project
build.bat

# Run the program
bin/mnist_nn.exe
```

## Contributing
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors
* **Rohan Patrick**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
* Yann LeCun for the MNIST dataset

