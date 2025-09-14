from __future__ import annotations
import asyncio
import json
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
import typer

from .utils import connectit_home, data_file, load_json, save_json, gen_salt, hash_password
from .model import random_mlp, serialize_layer
from .coordinator import Coordinator
from .protocol import NODE_LIST, LIST_NODES, RUN_PIPELINE, RUN_TRAIN_STEP, FORWARD_TASK, HF_LOAD, HF_INFER, HF_UNLOAD, CREATE_JOB, RUN_JOB_STEPS, GET_JOB


console = Console()


USERS_PATH = data_file("users.json")


def ensure_users_store():
    if not USERS_PATH.exists():
        save_json(USERS_PATH, {"users": []})


def find_user(username: str) -> Optional[Dict[str, Any]]:
    data = load_json(USERS_PATH, {"users": []})
    for u in data.get("users", []):
        if u.get("username") == username:
            return u
    return None


def add_user(username: str, password: str):
    data = load_json(USERS_PATH, {"users": []})
    if any(u.get("username") == username for u in data.get("users", [])):
        raise ValueError("username_exists")
    salt = gen_salt()
    hpw = hash_password(password, salt)
    data["users"].append({"username": username, "salt": salt, "hpw": hpw})
    save_json(USERS_PATH, data)


def verify_user(username: str, password: str) -> bool:
    u = find_user(username)
    if not u:
        return False
    return u.get("hpw") == hash_password(password, u.get("salt", ""))


def prompt_login() -> str:
    ensure_users_store()
    console.print("[bold cyan]Welcome to ConnectIT[/bold cyan]")
    while True:
        choice = typer.prompt("Login (l) / Sign up (s) / Quit (q)", default="l")
        if choice.lower() == "q":
            raise typer.Exit()
        if choice.lower() == "s":
            username = typer.prompt("Choose username")
            password = typer.prompt("Choose password", hide_input=True, confirmation_prompt=True)
            try:
                add_user(username, password)
                console.print("[green]User created. Please login.[/green]")
            except ValueError:
                console.print("[red]Username already exists[/red]")
        else:
            username = typer.prompt("Username")
            password = typer.prompt("Password", hide_input=True)
            if verify_user(username, password):
                console.print(f"[green]Logged in as[/green] {username}")
                return username
            else:
                console.print("[red]Invalid credentials[/red]")


async def fetch_nodes_ws(coordinator_url: str) -> List[Dict[str, Any]]:
    import websockets
    async with websockets.connect(coordinator_url) as ws:
        await ws.send(json.dumps({"type": LIST_NODES}))
        raw = await ws.recv()
        data = json.loads(raw)
        if data.get("type") == NODE_LIST:
            return data.get("nodes", [])
        return []


def list_offers(coordinator_url: str) -> List[Dict[str, Any]]:
    console.print("Fetching active nodes from coordinator...")
    try:
        nodes = asyncio.run(fetch_nodes_ws(coordinator_url))
    except Exception as e:
        console.print(f"[red]Failed to fetch nodes:[/red] {e}")
        nodes = []
    if not nodes:
        console.print("[yellow]No nodes found. Start one with `connectit node`.\nYou can still add nodes manually by ID.[/yellow]")
    tbl = Table(title="Active Nodes (Offers)")
    tbl.add_column("#")
    tbl.add_column("Node ID")
    tbl.add_column("Name")
    tbl.add_column("OS")
    tbl.add_column("CPU")
    tbl.add_column("MemGB")
    tbl.add_column("Price")
    for i, n in enumerate(nodes):
        r = n.get("resources", {})
        tbl.add_row(str(i), n.get("node_id", ""), n.get("name", ""), str(r.get("os", "")), str(r.get("cpu_count", "")), str(r.get("memory_gb", "")), str(n.get("price", 0)))
    console.print(tbl)
    # Choose order for a pipeline
    selected: List[Dict[str, Any]] = []
    while True:
        idx = typer.prompt("Pick node index for layer (or 'done')", default="done")
        if idx.lower() == "done":
            break
        try:
            i = int(idx)
            if i < 0 or i >= len(nodes):
                raise ValueError
            selected.append(nodes[i])
            console.print(f"Added: {nodes[i].get('node_id')}")
        except Exception:
            console.print("Invalid index")
    return selected


async def run_demo_inference_ws(coordinator_url: str, node_ids: List[str]):
    import websockets
    import numpy as np
    from .model import random_mlp, serialize_layer
    from .protocol import RESULT

    # Build a tiny 3-layer MLP and input vector
    layers = random_mlp(input_dim=16, hidden_dim=32, output_dim=8, layers=len(node_ids))
    layers_ser = [serialize_layer(l) for l in layers]
    x = np.random.default_rng(0).normal(0, 1, size=(1, 16)).astype("float32").tolist()
    task_id = "demo"
    async with websockets.connect(coordinator_url) as ws:
        await ws.send(json.dumps({
            "type": RUN_PIPELINE,
            "task_id": task_id,
            "layers": layers_ser,
            "nodes_order": [n for n in node_ids],
            "x": x,
        }))
        raw = await ws.recv()
        data = json.loads(raw)
        if data.get("type") == RESULT and data.get("task_id") == task_id:
            return data.get("output")
        raise RuntimeError("Pipeline failed")


def run_console():
    user = prompt_login()
    console.print("\n[bold]Main Menu[/bold]")
    coord_url = typer.prompt("Coordinator WS URL", default="ws://127.0.0.1:8765")
    selected_nodes: List[Dict[str, Any]] = []
    current_job_id: Optional[str] = None

    while True:
        console.print("\nChoose:")
        console.print("1) List/add offers (nodes)")
        console.print("2) Run demo inference across selected nodes")
        console.print("3) Run demo training steps across selected nodes")
        console.print("4) HF quick inference on a node")
        console.print("5) Load dataset with preprocessing (preview)")
        console.print("6) Create persistent MLP training job")
        console.print("7) Run steps on current job")
        console.print("9) Create HF batched inference job")
        console.print("8) Prototype DistilBERT split across 2 nodes")
        console.print("h) Help")
        console.print("q) Quit")
        choice = typer.prompt("Select", default="1")
        if choice == "q":
            break
        elif choice == "1":
            offers = list_offers(coord_url)
            tbl = Table(title="Selected Nodes")
            tbl.add_column("Order")
            tbl.add_column("Node ID")
            for i, o in enumerate(offers):
                tbl.add_row(str(i), o["node_id"])
            console.print(tbl)
            selected_nodes = offers
        elif choice == "2":
            node_ids = [n.get("node_id") for n in selected_nodes]
            if len(node_ids) < 1:
                console.print("[yellow]Select at least one node first.[/yellow]")
                continue
            console.print(f"Running demo across {len(node_ids)} layers/nodes ...")
            try:
                out = asyncio.run(run_demo_inference_ws(coord_url, node_ids))
                console.print("Output:")
                console.print(str(out)[:200] + ("..." if len(str(out)) > 200 else ""))
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        elif choice == "3":
            node_ids = [n.get("node_id") for n in selected_nodes]
            if len(node_ids) < 1:
                console.print("[yellow]Select at least one node first.[/yellow]")
                continue
            import numpy as np
            layers_n = len(node_ids)
            input_dim = int(typer.prompt("Input dim", default="16"))
            hidden_dim = int(typer.prompt("Hidden dim", default="32"))
            output_dim = int(typer.prompt("Output dim", default="8"))
            batch = int(typer.prompt("Batch size", default="4"))
            steps = int(typer.prompt("Steps", default="5"))
            lr = float(typer.prompt("Learning rate", default="0.01"))
            # Build model
            from .model import random_mlp, serialize_layer
            layers = random_mlp(input_dim, hidden_dim, output_dim, layers_n)
            layers_ser = [serialize_layer(l) for l in layers]
            rng = np.random.default_rng(0)
            console.print(f"Training {steps} steps...")
            try:
                for s in range(steps):
                    x = rng.normal(0, 1, size=(batch, input_dim)).astype("float32").tolist()
                    # Random target
                    y = rng.normal(0, 1, size=(batch, output_dim)).astype("float32").tolist()
                    # One step
                    import websockets
                    async def step_once():
                        async with websockets.connect(coord_url) as ws:
                            await ws.send(json.dumps({
                                "type": RUN_TRAIN_STEP,
                                "task_id": f"train-{s}",
                                "layers": layers_ser,
                                "nodes_order": node_ids,
                                "x": x,
                                "y": y,
                                "lr": lr,
                            }))
                            raw = await ws.recv()
                            data = json.loads(raw)
                            if data.get("type") == "result":
                                return data.get("layers"), data.get("loss")
                            raise RuntimeError(data)
                    layers_ser, loss = asyncio.run(step_once())
                    console.print(f"Step {s+1}/{steps} loss={loss:.6f}")
                console.print("[green]Training done.[/green]")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        elif choice == "4":
            nodes = asyncio.run(fetch_nodes_ws(coord_url))
            if not nodes:
                console.print("[yellow]No nodes available.[/yellow]")
                continue
            tbl = Table(title="Nodes")
            tbl.add_column("#")
            tbl.add_column("Node ID")
            for i, n in enumerate(nodes):
                tbl.add_row(str(i), n.get("node_id", ""))
            console.print(tbl)
            idx = int(typer.prompt("Pick node index", default="0"))
            if idx < 0 or idx >= len(nodes):
                console.print("Invalid index")
                continue
            node_id = nodes[idx]["node_id"]
            model_name = typer.prompt("HF model name", default="distilgpt2")
            prompt = typer.prompt("Prompt", default="Hello from ConnectIT")
            max_new = int(typer.prompt("max_new_tokens", default="32"))
            import websockets
            async def hf_once():
                async with websockets.connect(coord_url) as ws:
                    # load
                    await ws.send(json.dumps({"type": FORWARD_TASK, "node_id": node_id, "payload": {"kind": HF_LOAD, "model_name": model_name}}))
                    r = json.loads(await ws.recv())
                    if r.get("type") != "result":
                        raise RuntimeError(r)
                    mid = r.get("model_id")
                    # infer
                    await ws.send(json.dumps({"type": FORWARD_TASK, "node_id": node_id, "payload": {"kind": HF_INFER, "model_id": mid, "prompt": prompt, "max_new_tokens": max_new}}))
                    r2 = json.loads(await ws.recv())
                    txt = r2.get("text") if r2.get("type") == "result" else str(r2)
                    # unload (best-effort)
                    try:
                        await ws.send(json.dumps({"type": FORWARD_TASK, "node_id": node_id, "payload": {"kind": HF_UNLOAD, "model_id": mid}}))
                        await ws.recv()
                    except Exception:
                        pass
                    return txt
            try:
                out = asyncio.run(hf_once())
                console.print("[green]Generated:[/green]")
                console.print(out)
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        elif choice == "5":
            try:
                from .datasets import build_preprocess_config, load_and_preprocess
                ds_name = typer.prompt("Dataset", default="wikitext")
                split = typer.prompt("Split", default="train")
                tok = typer.prompt("Tokenizer", default="distilbert-base-uncased")
                text_field = typer.prompt("Text field", default="text")
                max_len = int(typer.prompt("Max length", default="64"))
                cfg = build_preprocess_config(tok, text_field=text_field, max_length=max_len)
                ds = load_and_preprocess(ds_name, split, cfg)
                # Show 2 samples
                it = iter(ds)
                console.print("Sample 1:")
                console.print(next(it))
                console.print("Sample 2:")
                console.print(next(it))
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        elif choice == "6":
            node_ids = [n.get("node_id") for n in selected_nodes]
            if len(node_ids) < 1:
                console.print("[yellow]Select at least one node first.[/yellow]")
                continue
            from .model import random_mlp, serialize_layer
            layers_n = len(node_ids)
            input_dim = int(typer.prompt("Input dim", default="16"))
            hidden_dim = int(typer.prompt("Hidden dim", default="32"))
            output_dim = int(typer.prompt("Output dim", default="8"))
            layers = random_mlp(input_dim, hidden_dim, output_dim, layers_n)
            layers_ser = [serialize_layer(l) for l in layers]
            opt_name = typer.prompt("Optimizer (sgd/momentum/adam)", default="adam")
            lr = float(typer.prompt("LR", default="0.001"))
            cfg = {"kind": "mlp_train", "nodes_order": node_ids, "layers": layers_ser, "optimizer": {"name": opt_name, "params": {"lr": lr}}}
            import websockets
            async def create_job():
                async with websockets.connect(coord_url) as ws:
                    await ws.send(json.dumps({"type": CREATE_JOB, "config": cfg}))
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("type") == "result":
                        return data.get("job_id")
                    raise RuntimeError(data)
            try:
                current_job_id = asyncio.run(create_job())
                console.print(f"[green]Job created:[/green] {current_job_id}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        elif choice == "7":
            if not current_job_id:
                console.print("[yellow]Create a job first (option 6).[/yellow]")
                continue
            batch = int(typer.prompt("Batch size", default="4"))
            input_dim = int(typer.prompt("Input dim (same as job)", default="16"))
            output_dim = int(typer.prompt("Output dim (same as job)", default="8"))
            steps = int(typer.prompt("Steps", default="5"))
            import numpy as np, websockets
            rng = np.random.default_rng(0)
            x = rng.normal(0, 1, size=(batch, input_dim)).astype("float32").tolist()
            y = rng.normal(0, 1, size=(batch, output_dim)).astype("float32").tolist()
            async def run_steps():
                async with websockets.connect(coord_url) as ws:
                    await ws.send(json.dumps({"type": RUN_JOB_STEPS, "job_id": current_job_id, "steps": steps, "batch": x, "target": y}))
                    data = json.loads(await ws.recv())
                    return data
            try:
                res = asyncio.run(run_steps())
                if res.get("type") == "result":
                    console.print(f"Step now: {res.get('step')} losses={res.get('losses')}")
                else:
                    console.print(res)
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        elif choice == "9":
            nodes = asyncio.run(fetch_nodes_ws(coord_url))
            if not nodes:
                console.print("[yellow]No nodes available.[/yellow]")
                continue
            tbl = Table(title="Nodes")
            tbl.add_column("#")
            tbl.add_column("Node ID")
            for i, n in enumerate(nodes):
                tbl.add_row(str(i), n.get("node_id", ""))
            console.print(tbl)
            idx = int(typer.prompt("Pick node for inference", default="0"))
            if idx < 0 or idx >= len(nodes):
                console.print("Invalid index")
                continue
            node_id = nodes[idx]["node_id"]
            model_name = typer.prompt("HF Causal LM model", default="distilgpt2")
            ds_name = typer.prompt("Dataset name", default="ag_news")
            split = typer.prompt("Split", default="train")
            text_field = typer.prompt("Text field", default="text")
            max_new = int(typer.prompt("max_new_tokens", default="32"))
            cfg = {
                "kind": "hf_infer",
                "nodes_order": [node_id],
                "model_name": model_name,
                "dataset": {"name": ds_name, "split": split, "text_field": text_field},
                "max_new_tokens": max_new,
            }
            import websockets
            async def create_job():
                async with websockets.connect(coord_url) as ws:
                    await ws.send(json.dumps({"type": CREATE_JOB, "config": cfg}))
                    data = json.loads(await ws.recv())
                    if data.get("type") == "result":
                        return data.get("job_id")
                    raise RuntimeError(data)
            try:
                current_job_id = asyncio.run(create_job())
                console.print(f"[green]HF job created:[/green] {current_job_id}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        elif choice == "8":
            nodes = asyncio.run(fetch_nodes_ws(coord_url))
            if len(nodes) < 2:
                console.print("[yellow]Need at least 2 nodes for prototype.[/yellow]")
                continue
            tbl = Table(title="Nodes")
            tbl.add_column("#")
            tbl.add_column("Node ID")
            for i, n in enumerate(nodes):
                tbl.add_row(str(i), n.get("node_id", ""))
            console.print(tbl)
            i0 = int(typer.prompt("Index of node A", default="0"))
            i1 = int(typer.prompt("Index of node B", default="1"))
            split = int(typer.prompt("Split layer (0-6)", default="3"))
            text = typer.prompt("Text", default="Hello from ConnectIT prototype")
            n0 = nodes[i0]["node_id"]
            n1 = nodes[i1]["node_id"]
            import websockets
            async def run_pipe():
                async with websockets.connect(coord_url) as ws:
                    await ws.send(json.dumps({"type": "run_hf_pipeline", "nodes_order": [n0, n1], "model_name": "distilbert-base-uncased", "split_layer": split, "text": text}))
                    d = json.loads(await ws.recv())
                    return d
            try:
                res = asyncio.run(run_pipe())
                if res.get("type") == "result":
                    console.print(f"Hidden shapes: partA={res.get('shape0')} partB={res.get('shape1')}")
                else:
                    console.print(res)
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
        else:
            console.print("- Start coordinator: `connectit coordinator`.")
            console.print("- Start nodes: `connectit node --coordinator ws://HOST:PORT --name mynode`.")
            console.print("- This console helps track nodes and build plans; remote triggering is WIP.")
