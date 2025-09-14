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
    kind: str  # 'mlp_train'
    nodes_order: list[str]
    layers: list[dict]
    optimizer: dict  # {'name': 'adam', 'params': {...}}
    step: int
    losses: list[float]
    created_ms: int
    updated_ms: int
    status: str  # 'pending'|'running'|'stopped'


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
    )
    state = load_state()
    state["jobs"][job.job_id] = asdict(job)
    save_state(state)
    return job


def get_job(job_id: str) -> Optional[Job]:
    state = load_state()
    j = state.get("jobs", {}).get(job_id)
    if not j:
        return None
    return Job(**j)


def put_job(job: Job):
    state = load_state()
    job.updated_ms = now_ms()
    state["jobs"][job.job_id] = asdict(job)
    save_state(state)


