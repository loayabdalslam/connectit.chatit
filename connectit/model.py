from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Layer:
    W: np.ndarray  # (in_dim, out_dim)
    b: np.ndarray  # (out_dim,)
    activation: str  # 'relu' | 'gelu' | 'none'


def act(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(0, x)
    if kind == "gelu":
        # approximate GELU
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return x


def layer_forward(layer: Layer, x: np.ndarray) -> np.ndarray:
    y = x @ layer.W + layer.b
    return act(y, layer.activation)


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    # derivative of approximate GELU used above
    c = np.sqrt(2 / np.pi).astype(np.float32) if isinstance(x, np.ndarray) else np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * (x ** 3)))
    dt = (1 - t ** 2) * c * (1 + 0.134145 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * dt


def act_derivative(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return (z > 0).astype(np.float32)
    if kind == "gelu":
        return gelu_derivative(z)
    return np.ones_like(z, dtype=np.float32)


def random_mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int, seed: int = 42) -> List[Layer]:
    rng = np.random.default_rng(seed)
    dims: List[Tuple[int, int]] = []
    d_in = input_dim
    for i in range(layers - 1):
        dims.append((d_in, hidden_dim))
        d_in = hidden_dim
    dims.append((d_in, output_dim))

    out: List[Layer] = []
    for i, (din, dout) in enumerate(dims):
        W = rng.normal(0, 0.02, size=(din, dout)).astype(np.float32)
        b = np.zeros((dout,), dtype=np.float32)
        activation = "relu" if i < len(dims) - 1 else "none"
        out.append(Layer(W=W, b=b, activation=activation))
    return out


def serialize_layer(layer: Layer) -> dict:
    return {
        "W": layer.W.tolist(),
        "b": layer.b.tolist(),
        "activation": layer.activation,
    }


def deserialize_layer(d: dict) -> Layer:
    return Layer(W=np.array(d["W"], dtype=np.float32), b=np.array(d["b"], dtype=np.float32), activation=d["activation"])

