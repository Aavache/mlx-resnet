<h1 style="text-align: center;">🍏 MLX - ResNet 🍏</h1>

ResNet implementation with the [MLX](https://github.com/ml-explore/mlx), Apple's deep learning framework.

Drop a ⭐️!!

# MNIST with ResNet18

Here the logs when training with ResNet18 (no tuning at all) for couple epochs:

```sh
Epoch: 0 | Iter: 0 | Loss: 2.550
Epoch: 0 | Iter: 50 | Loss: 1.813
Epoch: 0 | Iter: 100 | Loss: 1.241
Epoch: 0 | Iter: 150 | Loss: 0.965
Epoch: 0 | Iter: 200 | Loss: 0.741
Epoch 0: Test accuracy 0.671
Epoch: 1 | Iter: 0 | Loss: 1.039
Epoch: 1 | Iter: 50 | Loss: 0.491
Epoch: 1 | Iter: 100 | Loss: 0.471
Epoch: 1 | Iter: 150 | Loss: 0.497
Epoch: 1 | Iter: 200 | Loss: 0.377
Epoch 1: Test accuracy 0.876
Epoch: 2 | Iter: 0 | Loss: 0.425
Epoch: 2 | Iter: 50 | Loss: 0.367
Epoch: 2 | Iter: 100 | Loss: 0.334
Epoch: 2 | Iter: 150 | Loss: 0.323
Epoch: 2 | Iter: 200 | Loss: 0.346
Epoch 2: Test accuracy 0.916
Epoch: 3 | Iter: 0 | Loss: 0.334
Epoch: 3 | Iter: 50 | Loss: 0.308
Epoch: 3 | Iter: 100 | Loss: 0.282
Epoch: 3 | Iter: 150 | Loss: 0.230
Epoch: 3 | Iter: 200 | Loss: 0.173
Epoch 3: Test accuracy 0.902
Epoch: 4 | Iter: 0 | Loss: 0.282
Epoch: 4 | Iter: 50 | Loss: 0.184
Epoch: 4 | Iter: 100 | Loss: 0.268
Epoch: 4 | Iter: 150 | Loss: 0.205
Epoch: 4 | Iter: 200 | Loss: 0.246
Epoch 4: Test accuracy 0.932
```


# Future updates

* `dilation` and `groups` features in convolutional layers is missing ([issue](https://github.com/ml-explore/mlx/issues/100) created). More recent ResNet requires this parameters.
* Batch norm is also missing.
* No pretrained weights.

# Contribute

Feel free to create issues and PRs :)
