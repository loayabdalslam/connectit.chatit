import numpy as np


def run():
    from connectit.model import random_mlp, layer_forward, act_derivative
    layers = random_mlp(8, 16, 4, 2)
    x = np.random.default_rng(0).normal(0, 1, size=(3, 8)).astype("float32")
    y = layer_forward(layers[0], x)
    assert y.shape == (3, 16)
    d = act_derivative(y, "relu")
    assert d.shape == (3, 16)

