from __future__ import annotations
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
import websockets
from websockets.server import WebSocketServerProtocol
from rich.console import Console
from rich.table import Table

from .protocol import (
    msg,
    REGISTER,
    HEARTBEAT,
    TASK,
    RESULT,
    ERROR,
    INFO,
    NODE_LIST,
    LIST_NODES,
    RUN_PIPELINE,
    RUN_TRAIN_STEP,
    CREATE_JOB,
    RUN_JOB_STEPS,
    GET_JOB,
    STOP_JOB,
    FORWARD_TASK,
    TASK_LAYER_FORWARD,
    TASK_LAYER_FORWARD_TRAIN,
    TASK_LAYER_BACKWARD,
)
from .utils import new_id, now_ms
import numpy as np
from .jobs import create_mlp_job, get_job, put_job, Job
from .optim import build_optimizer


console = Console()


class Coordinator:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # node_id -> {ws, resources, price, name, last_seen}
        self._lock = asyncio.Lock()

    async def register_node(self, ws: WebSocketServerProtocol, data: Dict[str, Any]) -> str:
        node_id = data.get("node_id") or new_id("node")
        info = {
            "ws": ws,
            "name": data.get("name", node_id),
            "resources": data.get("resources", {}),
            "price": data.get("price", 0.0),
            "last_seen": now_ms(),
        }
        async with self._lock:
            self.nodes[node_id] = info
        await ws.send(json.dumps(msg(INFO, node_id=node_id)))
        console.log(f"[green]Node registered[/green]: {node_id} {info['resources']}")
        return node_id

    async def heartbeat(self, node_id: str):
        async with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id]["last_seen"] = now_ms()

    async def remove_ws(self, ws: WebSocketServerProtocol):
        async with self._lock:
            for nid, info in list(self.nodes.items()):
                if info.get("ws") is ws:
                    console.log(f"[red]Node disconnected[/red]: {nid}")
                    del self.nodes[nid]

    def list_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        out = []
        for nid, info in self.nodes.items():
            out.append((nid, {k: v for k, v in info.items() if k != "ws"}))
        return out

    async def assign_and_run_pipeline(self, layers: List[dict], nodes_order: List[str], x: List[float]) -> List[float]:
        # Sequential: send layer + input to each node in order
        cur = x
        for i, nid in enumerate(nodes_order):
            info = self.nodes.get(nid)
            if not info:
                raise RuntimeError(f"Node {nid} not available")
            ws: WebSocketServerProtocol = info["ws"]
            task_id = new_id("task")
            payload = {
                "kind": TASK_LAYER_FORWARD,
                "layer": layers[i],
                "x": cur,
            }
            await ws.send(json.dumps(msg(TASK, task_id=task_id, payload=payload)))
            # Wait for result
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("type") == RESULT and data.get("task_id") == task_id:
                    cur = data.get("output")
                    break
                elif data.get("type") == ERROR and data.get("task_id") == task_id:
                    raise RuntimeError(data.get("error", "unknown error"))
                # Ignore other messages (e.g., heartbeats)
        return cur

    async def train_step_pipeline(
        self,
        layers: List[dict],
        nodes_order: List[str],
        x: List[List[float]],
        y_true: List[List[float]],
        lr: float,
    ) -> tuple[List[dict], float]:
        # Forward with caches
        step_id = new_id("step")
        cur = x
        for i, nid in enumerate(nodes_order):
            info = self.nodes.get(nid)
            if not info:
                raise RuntimeError(f"Node {nid} not available")
            ws: WebSocketServerProtocol = info["ws"]
            task_id = new_id("task")
            payload = {
                "kind": TASK_LAYER_FORWARD_TRAIN,
                "layer": layers[i],
                "x": cur,
                "cache_id": f"{step_id}-L{i}",
            }
            await ws.send(json.dumps(msg(TASK, task_id=task_id, payload=payload)))
            # Wait for result
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("type") == RESULT and data.get("task_id") == task_id:
                    cur = data.get("output")
                    break
                elif data.get("type") == ERROR and data.get("task_id") == task_id:
                    raise RuntimeError(data.get("error", "unknown error"))
        y_pred = np.array(cur, dtype=np.float32)
        y_t = np.array(y_true, dtype=np.float32)
        # MSE loss and gradient
        diff = y_pred - y_t
        loss = float(np.mean(diff ** 2))
        upstream = (2.0 / y_pred.shape[0]) * diff

        # Backward pass
        dY = upstream.tolist()
        for i in reversed(range(len(nodes_order))):
            nid = nodes_order[i]
            info = self.nodes.get(nid)
            if not info:
                raise RuntimeError(f"Node {nid} not available")
            ws: WebSocketServerProtocol = info["ws"]
            task_id = new_id("task")
            payload = {
                "kind": TASK_LAYER_BACKWARD,
                "cache_id": f"{step_id}-L{i}",
                "upstream_grad": dY,
            }
            await ws.send(json.dumps(msg(TASK, task_id=task_id, payload=payload)))
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("type") == RESULT and data.get("task_id") == task_id:
                    gW = np.array(data.get("gW"), dtype=np.float32)
                    gb = np.array(data.get("gb"), dtype=np.float32)
                    dX = data.get("dX")
                    # Update layer parameters (SGD)
                    L = layers[i]
                    W = np.array(L["W"], dtype=np.float32) - lr * gW
                    b = np.array(L["b"], dtype=np.float32) - lr * gb
                    L["W"] = W.tolist()
                    L["b"] = b.tolist()
                    dY = dX
                    break
                elif data.get("type") == ERROR and data.get("task_id") == task_id:
                    raise RuntimeError(data.get("error", "unknown error"))
        return layers, loss


async def handle_client(coord: Coordinator, ws: WebSocketServerProtocol):
    node_id: Optional[str] = None
    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except Exception:
                await ws.send(json.dumps(msg(ERROR, error="invalid_json")))
                continue
            t = data.get("type")
            if t == REGISTER:
                node_id = await coord.register_node(ws, data)
            elif t == HEARTBEAT:
                if node_id:
                    await coord.heartbeat(node_id)
            elif t == LIST_NODES:
                nodes = []
                for nid, info in coord.list_nodes():
                    nodes.append({"node_id": nid, **info})
                await ws.send(json.dumps(msg(NODE_LIST, nodes=nodes)))
            elif t == RUN_PIPELINE:
                try:
                    task_id = data.get("task_id")
                    layers = data.get("layers") or []
                    nodes_order = data.get("nodes_order") or []
                    x = data.get("x") or []
                    out = await coord.assign_and_run_pipeline(layers=layers, nodes_order=nodes_order, x=x)
                    await ws.send(json.dumps(msg(RESULT, task_id=task_id, output=out)))
                except Exception as e:
                    await ws.send(json.dumps(msg(ERROR, error=str(e), task_id=data.get("task_id"))))
            elif t == RUN_TRAIN_STEP:
                try:
                    task_id = data.get("task_id")
                    layers = data.get("layers") or []
                    nodes_order = data.get("nodes_order") or []
                    x = data.get("x") or []
                    y = data.get("y") or []
                    lr = float(data.get("lr") or 0.01)
                    updated_layers, loss = await coord.train_step_pipeline(layers=layers, nodes_order=nodes_order, x=x, y_true=y, lr=lr)
                    await ws.send(json.dumps(msg(RESULT, task_id=task_id, layers=updated_layers, loss=loss)))
                except Exception as e:
                    await ws.send(json.dumps(msg(ERROR, error=str(e), task_id=data.get("task_id"))))
            elif t == CREATE_JOB:
                try:
                    cfg = data.get("config") or {}
                    kind = cfg.get("kind", "mlp_train")
                    if kind != "mlp_train":
                        raise RuntimeError("unsupported_job_kind")
                    nodes_order = cfg.get("nodes_order") or []
                    layers = cfg.get("layers") or []
                    optimizer = cfg.get("optimizer") or {"name": "adam", "params": {"lr": 0.001}}
                    job = create_mlp_job(nodes_order, layers, optimizer)
                    await ws.send(json.dumps(msg(RESULT, job_id=job.job_id)))
                except Exception as e:
                    await ws.send(json.dumps(msg(ERROR, error=str(e))))
            elif t == RUN_JOB_STEPS:
                try:
                    job_id = data.get("job_id")
                    steps = int(data.get("steps") or 1)
                    batch = data.get("batch")
                    target = data.get("target")
                    lr = float((data.get("lr") or 0.01))
                    job = get_job(job_id)
                    if not job:
                        raise RuntimeError("job_not_found")
                    if job.kind != "mlp_train":
                        raise RuntimeError("unsupported_job_kind")
                    # Build optimizer stateful per job
                    opt = build_optimizer(job.optimizer.get("name", "adam"), **job.optimizer.get("params", {}))
                    losses = []
                    for _ in range(steps):
                        updated_layers, loss = await coord.train_step_pipeline(layers=job.layers, nodes_order=job.nodes_order, x=batch, y_true=target, lr=lr)
                        job.layers = updated_layers
                        job.step += 1
                        losses.append(loss)
                    put_job(job)
                    await ws.send(json.dumps(msg(RESULT, job_id=job.job_id, losses=losses, step=job.step, layers=job.layers)))
                except Exception as e:
                    await ws.send(json.dumps(msg(ERROR, error=str(e))))
            elif t == GET_JOB:
                j = get_job(data.get("job_id"))
                if not j:
                    await ws.send(json.dumps(msg(ERROR, error="job_not_found")))
                else:
                    await ws.send(json.dumps(msg(RESULT, job=Job(**j.__dict__).__dict__)))
            elif t == STOP_JOB:
                j = get_job(data.get("job_id"))
                if not j:
                    await ws.send(json.dumps(msg(ERROR, error="job_not_found")))
                else:
                    j.status = "stopped"
                    put_job(j)
                    await ws.send(json.dumps(msg(RESULT, job_id=j.job_id, status=j.status)))
            elif t == FORWARD_TASK:
                try:
                    target_nid = data.get("node_id")
                    payload = data.get("payload")
                    info = coord.nodes.get(target_nid)
                    if not info:
                        raise RuntimeError("node_not_found")
                    ws_target: WebSocketServerProtocol = info["ws"]
                    sub_task_id = new_id("task")
                    await ws_target.send(json.dumps(msg(TASK, task_id=sub_task_id, payload=payload)))
                    while True:
                        raw2 = await ws_target.recv()
                        data2 = json.loads(raw2)
                        if data2.get("task_id") == sub_task_id and data2.get("type") in (RESULT, ERROR):
                            await ws.send(json.dumps(data2))
                            break
                except Exception as e:
                    await ws.send(json.dumps(msg(ERROR, error=str(e))))
            else:
                # coordinator currently accepts only register/heartbeat directly
                pass
    except websockets.ConnectionClosed:
        pass
    finally:
        await coord.remove_ws(ws)


async def coordinator_server(host: str, port: int):
    coord = Coordinator()

    async def handler(ws):
        await handle_client(coord, ws)

    console.log(f"Starting coordinator on ws://{host}:{port}")
    async with websockets.serve(handler, host, port, max_size=32 * 1024 * 1024):
        # Periodically print node table
        while True:
            await asyncio.sleep(5)
            tbl = Table(title="Active Nodes")
            tbl.add_column("Node ID")
            tbl.add_column("Name")
            tbl.add_column("OS")
            tbl.add_column("CPU")
            tbl.add_column("GPU")
            tbl.add_column("Mem")
            tbl.add_column("Price")
            for nid, info in coord.list_nodes():
                r = info.get("resources", {})
                tbl.add_row(
                    nid,
                    info.get("name", ""),
                    str(r.get("os", "?")),
                    str(r.get("cpu_count", "?")),
                    str(r.get("gpu", "-")),
                    str(r.get("memory_gb", "?")),
                    str(info.get("price", 0.0)),
                )
            console.print(tbl)


def run_coordinator(host: str, port: int):
    asyncio.run(coordinator_server(host, port))
