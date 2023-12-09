"""Training ResNet on CIFAR-10 dataset."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np

import mnist as mnist
from layers import upsample_nearest
import resnet

NUMBER_OF_CLASSES = 10
NUMBER_OF_EPOCHS = 10
BATCH_SIZE = 256
UPSCALE = 2


def convert_to_3_channels(x):
    return mx.concatenate([x, x, x], axis=3)


def batch_iterate(batch_size, X, y,shuffle=True):
    perm = mx.array(np.random.permutation(y.size)) if shuffle else mx.arange(y.size)
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield convert_to_3_channels(X[ids]), y[ids]


def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def train():
    # Load the data
    train_images, train_labels, test_images, test_labels = map(
        mx.array, mnist.mnist()
    )

    model = resnet.resnet18(num_classes=NUMBER_OF_CLASSES)
    mx.eval(model.parameters())

    # Create the gradient function and the optimizer
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=1e-3)

    for e in range(NUMBER_OF_EPOCHS):
        iter = 0
        # Training
        for X, y in batch_iterate(BATCH_SIZE, train_images, train_labels):
            if UPSCALE > 1:
                X = upsample_nearest(X, UPSCALE)
            loss, grads = loss_and_grad_fn(model, X, y)

            # Update the model with the gradients. So far no computation has happened.
            optimizer.update(model, grads)

            # Compute the new parameters but also the optimizer state.
            mx.eval(model.parameters(), optimizer.state)

            if iter % 50 == 0:
                print(f"Epoch: {e} | Iter: {iter} | Loss: {loss.item():.3f}")
            iter += 1

        # Evaluation
        accuracy = 0
        for X, y in batch_iterate(BATCH_SIZE, test_images, test_labels, shuffle=False):
            if UPSCALE > 1:
                X = upsample_nearest(X, UPSCALE)
            accuracy += mx.sum(mx.argmax(model(X), axis=1) == y)
        accuracy /= test_labels.shape[0]
        print(f"Epoch {e}: Test accuracy {accuracy.item():.3f}")


if __name__ == "__main__":
    train()