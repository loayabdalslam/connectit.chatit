from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# Simple JSON protocol helpers


def msg(type: str, **kwargs) -> Dict[str, Any]:
    d = {"type": type}
    d.update(kwargs)
    return d


# Message types (string constants)

REGISTER = "register"
HEARTBEAT = "heartbeat"
PONG = "pong"
PING = "ping"
TASK = "task"
RESULT = "result"
ERROR = "error"
INFO = "info"
NODE_LIST = "node_list"
LIST_NODES = "list_nodes"
RUN_PIPELINE = "run_pipeline"
RUN_TRAIN_STEP = "run_train_step"
CREATE_JOB = "create_job"
RUN_JOB_STEPS = "run_job_steps"
GET_JOB = "get_job"
STOP_JOB = "stop_job"
FORWARD_TASK = "forward_task"


# Task payloads

TASK_LAYER_FORWARD = "layer_forward"
TASK_LAYER_FORWARD_TRAIN = "layer_forward_train"
TASK_LAYER_BACKWARD = "layer_backward"

# Hugging Face / ONNX tasks (node-side)
HF_LOAD = "hf_load"
HF_UNLOAD = "hf_unload"
HF_INFER = "hf_infer"
ONNX_LOAD = "onnx_load"
ONNX_UNLOAD = "onnx_unload"
ONNX_INFER = "onnx_infer"


def is_message(obj: Any) -> bool:
    return isinstance(obj, dict) and "type" in obj
