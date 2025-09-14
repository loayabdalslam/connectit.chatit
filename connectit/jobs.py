from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from .utils import data_file, load_json, save_json, new_id, now_ms
from .optim import build_optimizer, Optimizer
import numpy as np


STATE_PATH = data_file("coord_state.json")


@dataclass
class Job:
    job_id: str
    kind: str  # 'mlp_train' | 'hf_infer'
    nodes_order: list[str]
    layers: list[dict]
    optimizer: dict  # {'name': 'adam', 'params': {...}, 'state': {...}}
    step: int
    losses: list[float]
    created_ms: int
    updated_ms: int
    status: str  # 'pending'|'running'|'stopped'
    params: dict  # job-specific config
    outputs: list  # generic outputs for non-loss jobs


def load_state() -> Dict[str, Any]:
    return load_json(STATE_PATH, {"jobs": {}})


def save_state(state: Dict[str, Any]):
    save_json(STATE_PATH, state)


def create_mlp_job(nodes_order: list[str], layers: list[dict], optimizer: dict) -> Job:
    job = Job(
        job_id=new_id("job"),
        kind="mlp_train",
        nodes_order=nodes_order,
        layers=layers,
        optimizer=optimizer,
        step=0,
        losses=[],
        created_ms=now_ms(),
        updated_ms=now_ms(),
        status="pending",
        params={},
        outputs=[],
    )
    state = load_state()
    state["jobs"][job.job_id] = asdict(job)
    save_state(state)
    return job


def job_from_dict(d: dict) -> Job:
    # Backward compatible defaults
    return Job(
        job_id=d.get("job_id"),
        kind=d.get("kind", "mlp_train"),
        nodes_order=d.get("nodes_order", []),
        layers=d.get("layers", []),
        optimizer=d.get("optimizer", {"name": "adam", "params": {}, "state": {}}),
        step=int(d.get("step", 0)),
        losses=d.get("losses", []),
        created_ms=int(d.get("created_ms", now_ms())),
        updated_ms=int(d.get("updated_ms", now_ms())),
        status=d.get("status", "pending"),
        params=d.get("params", {}),
        outputs=d.get("outputs", []),
    )


def get_job(job_id: str) -> Optional[Job]:
    state = load_state()
    j = state.get("jobs", {}).get(job_id)
    if not j:
        return None
    return job_from_dict(j)


def put_job(job: Job):
    state = load_state()
    job.updated_ms = now_ms()
    state["jobs"][job.job_id] = asdict(job)
    save_state(state)


def create_hf_infer_job(node_id: str, model_name: str, dataset: dict, max_new_tokens: int = 32) -> Job:
    job = Job(
        job_id=new_id("job"),
        kind="hf_infer",
        nodes_order=[node_id],
        layers=[],
        optimizer={"name": "none", "params": {}, "state": {}},
        step=0,
        losses=[],
        created_ms=now_ms(),
        updated_ms=now_ms(),
        status="pending",
        params={
            "model_name": model_name,
            "dataset": dataset,  # {name, split, text_field}
            "max_new_tokens": int(max_new_tokens),
            "cursor": int(0),
            "model_id": None,
        },
        outputs=[],
    )
    state = load_state()
    state["jobs"][job.job_id] = asdict(job)
    save_state(state)
    return job

