import numpy as np


def run():
    from connectit.optim import build_optimizer
    params = {"W": np.ones((2, 2), dtype=np.float32).tolist()}
    grads = {"W": (0.5 * np.ones((2, 2), dtype=np.float32)).tolist()}
    opt = build_optimizer("adam", lr=0.01)
    opt.step(params, grads)
    assert np.all(np.array(params["W"]) < 1.0), "params updated"

