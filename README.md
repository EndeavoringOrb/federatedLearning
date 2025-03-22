# Distributed Language Model Training

This repository contains code for training a text generation Recurrent Neural Network (RNN) using multiple clients. The training is done using [evolution strategies](https://openai.com/index/evolution-strategies/).

## Overview

The system consists of a server and multiple clients that work together to train a language model. The server distributes training data and model weights to clients, while clients perform computation and send back rewards (loss values) that are used to update the model's parameters.

### Key Features

- **Distributed Training**: Leverages multiple machines to accelerate training
- **Simple RNN Language Model**: Implements a character-level language model with RNN architecture
- **Efficient Communication**: Custom protocol for sending data between server and clients
- **Gradient-based Optimization**: Uses Adam optimizer for efficient parameter updates

## Repository Structure

- `utilities/`
  - `communication.py`: Handles network communication between server and clients
  - `data.py`: Data loading and preprocessing functions
  - `model.py`: Definition of the language model and optimizer
  - `misc.py`: Miscellaneous utility functions

- `basicServer.py`: Server implementation that coordinates training
- `basicClient.py`: Client implementation that performs computation

## How It Works

1. The server initializes model weights and starts listening for client connections
2. Clients connect to the server and receive initial weights and configuration
3. For each training step:
   - (Optional, can happen every N steps) Server distributes tokenized text data to clients
   - Clients compute loss values using the current model weights with added noise
   - Clients send loss values back to the server
   - Server normalizes the rewards and sends them back to clients
   - Clients update their local copy of the model weights based on these normalized rewards
   - Periodically, the server requests updated weights from a client to save checkpoints

## Technical Details

### Model Architecture

The system implements a recurrent neural network (RNN) language model with the following components:

- Multiple RNN layers with tanh activation functions
- Hidden state that captures context information
- Output layer that predicts probabilities for the next token
- Support for beam search during generation

### Optimization

The model is trained using:

- Distributed gradient estimation
- Adam optimizer for parameter updates
- Configurable learning rate and other hyperparameters

### Communication Protocol

The system uses a custom communication protocol with:

- Header-based messages specifying data length
- Support for different data types (text, numpy arrays, pickled objects)
- Chunked data transfer for large messages

## Usage

### Install Requirements

Clone github
```bash
git clone https://github.com/EndeavoringOrb/federatedLearning
cd federatedLearning
```
Set up python environment
```bash
python -m venv pyEnv
source pyEnv/bin/activate
pip install -r requirements
```

### Starting the Server

```bash
python basicServer.py
```

The server will start listening for client connections and save training checkpoints to the `trainingRuns/` directory.

### Starting a Client

```bash
python basicClient.py
```

The client will connect to the server and start participating in the training process.  
default address: `130.215.13.29:55551`  
Change server_ip as needed at bottom of basicClient.py.

### Configuration

The training configuration is defined in `basicServer.py` and includes:

- `timePerStep`: How long (in seconds) clients can run trials for any one training step
- `learningRate`: Learning rate for optimization
- `sigma`: Exploration parameter for gradient estimation. Controls the std dev of the noise added to weights for loss calculation
- `hiddenSize`: Size of hidden layers
- `vocabSize`: The number of tokens in the model's vocabulary
- `nLayers`: Number of RNN layers
- `optimizer`: Controls the optimizer type `sgd` or `adam`
- `beta1`: Only used if `optimizer` is set to `adam`. Controls the exponential decay rate of the *first* moment estimate of the gradient
- `beta2`: Only used if `optimizer` is set to `adam`. Controls the exponential decay rate of the *second* moment estimate of the gradient
- `stepNum`: Keeps track of the current training step number
- `checkPointTime`: Interval (in seconds) for saving model checkpoints
- `newTokensInterval`: Interval (in training steps) for sending new tokens to clients
- `batchSize`: Batch size for clients to use when evaluating loss

## Future Work
- Optimize training
  - Port to C++
  - Implement sequence packing for higher average batch sizes
- Support different model architectures
- Support different datasets