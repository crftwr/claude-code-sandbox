# hello-pytorch

A minimal PyTorch example demonstrating a supervised learning training loop on synthetic data, running on CPU.

## What it does

`train.py` generates a synthetic dataset and trains a neural network to recover the hidden relationship in that data.

### Synthetic data

1000 samples are generated where the target value is:

```
y = 3·x₁ + 2·x₂ − 1·x₃ + noise
```

The inputs `x₁, x₂, x₃` are random numbers. The goal of training is for the model to discover the true weights `[3.0, 2.0, -1.0]` from data alone, without being told the formula.

### Model

A single `nn.Linear(3 → 1)` layer — essentially a learned weighted sum with a bias term:

```
ŷ = w₁·x₁ + w₂·x₂ + w₃·x₃ + b
```

### Loss function: MSE

Mean Squared Error measures how far the model's predictions are from the true values:

```
MSE = mean((ŷ - y)²)
```

Lower is better. As training progresses, MSE should drop toward ~0.01 (the irreducible noise floor).

### Optimizer: Adam

Adam (Adaptive Moment Estimation) is an algorithm that updates the model's weights after each mini-batch of data. It improves on plain gradient descent by:

- **Momentum** — accumulates a moving average of past gradients to smooth out noisy updates and avoid oscillating
- **Adaptive learning rates** — each weight gets its own learning rate, scaled by how large its recent gradients have been, so rarely-updated weights can still learn quickly

In practice, Adam converges faster and is more robust to hyperparameter choices than basic SGD, making it the default optimizer for most deep learning tasks.

### Training loop

Each epoch passes all 1000 samples through the model in mini-batches of 64:

1. Forward pass — compute predictions
2. Compute MSE loss
3. Backward pass — compute gradients via backpropagation
4. Adam step — update weights

After 50 epochs the learned weights converge close to the true values `[3.0, 2.0, -1.0]`.

## Usage

```bash
make setup   # create .venv and install dependencies
make train   # run training
make clean   # remove .venv
```
