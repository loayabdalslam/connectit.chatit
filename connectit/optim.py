from __future__ import annotations
from typing import Dict, Any
import numpy as np


class Optimizer:
    def step(self, params: Dict[str, Any], grads: Dict[str, Any]):
        raise NotImplementedError
    def get_state(self) -> Dict[str, Any]:
        return {}
    def load_state(self, state: Dict[str, Any]):
        return self


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = float(lr)

    def step(self, params: Dict[str, Any], grads: Dict[str, Any]):
        for k in list(params.keys()):
            if k in grads:
                params[k] = (np.array(params[k], dtype=np.float32) - self.lr * np.array(grads[k], dtype=np.float32)).tolist()


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.v: Dict[str, Any] = {}

    def step(self, params: Dict[str, Any], grads: Dict[str, Any]):
        for k in list(params.keys()):
            if k in grads:
                g = np.array(grads[k], dtype=np.float32)
                v = self.v.get(k)
                if v is None:
                    v = np.zeros_like(g)
                v = self.momentum * v + g
                self.v[k] = v
                params[k] = (np.array(params[k], dtype=np.float32) - self.lr * v).tolist()
    def get_state(self) -> Dict[str, Any]:
        return {"v": {k: v.tolist() for k, v in self.v.items()}, "lr": self.lr, "momentum": self.momentum}
    def load_state(self, state: Dict[str, Any]):
        v = state.get("v", {})
        self.v = {k: np.array(vv, dtype=np.float32) for k, vv in v.items()}
        self.lr = float(state.get("lr", self.lr))
        self.momentum = float(state.get("momentum", self.momentum))
        return self


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: Dict[str, Any] = {}
        self.v: Dict[str, Any] = {}
        self.t: int = 0

    def step(self, params: Dict[str, Any], grads: Dict[str, Any]):
        self.t += 1
        for k in list(params.keys()):
            if k in grads:
                g = np.array(grads[k], dtype=np.float32)
                m = self.m.get(k)
                v = self.v.get(k)
                if m is None:
                    m = np.zeros_like(g)
                if v is None:
                    v = np.zeros_like(g)
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g * g)
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                params[k] = (np.array(params[k], dtype=np.float32) - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).tolist()
                self.m[k] = m
                self.v[k] = v
    def get_state(self) -> Dict[str, Any]:
        return {
            "m": {k: v.tolist() for k, v in self.m.items()},
            "v": {k: v.tolist() for k, v in self.v.items()},
            "t": self.t,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }
    def load_state(self, state: Dict[str, Any]):
        m = state.get("m", {})
        v = state.get("v", {})
        self.m = {k: np.array(vv, dtype=np.float32) for k, vv in m.items()}
        self.v = {k: np.array(vv, dtype=np.float32) for k, vv in v.items()}
        self.t = int(state.get("t", self.t))
        self.lr = float(state.get("lr", self.lr))
        self.beta1 = float(state.get("beta1", self.beta1))
        self.beta2 = float(state.get("beta2", self.beta2))
        self.eps = float(state.get("eps", self.eps))
        return self


def build_optimizer(name: str, **kwargs) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        return SGD(**kwargs)
    if name in ("momentum", "sgd_momentum"):
        return Momentum(**kwargs)
    if name == "adam":
        return Adam(**kwargs)
    raise ValueError(f"unknown optimizer: {name}")
