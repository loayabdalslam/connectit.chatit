from typing import Optional
import typer
from rich.console import Console

from .console import run_console
from .coordinator import run_coordinator
from .node import run_node
from .hf import has_transformers, has_datasets, load_model_and_tokenizer, export_torchscript, export_onnx

app = typer.Typer(add_completion=False, help="ConnectIT CLI (prototype)")
console = Console()


@app.command()
def console_app():
    """Open the interactive console (login, offers, jobs)."""
    run_console()


@app.command()
def coordinator(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8765, help="Bind port"),
):
    """Start the coordinator service (registry + orchestrator)."""
    run_coordinator(host=host, port=port)


@app.command()
def node(
    coordinator: str = typer.Option("ws://127.0.0.1:8765", help="Coordinator WS URL"),
    name: Optional[str] = typer.Option(None, help="Node name"),
    price: float = typer.Option(0.0, help="Price per unit (demo)"),
):
    """Start a node agent that connects to a coordinator."""
    run_node(coordinator_url=coordinator, node_name=name, price=price)


@app.command()
def export(
    to: str = typer.Option("onnx", help="onnx|torchscript"),
    model: str = typer.Option("distilbert-base-uncased", help="HF model name"),
    output: str = typer.Option("model.onnx", help="Output file for ONNX or directory for TorchScript save"),
    example_text: str = typer.Option("Hello ConnectIT", help="Example text to trace/export"),
):
    """Export a Hugging Face model locally to ONNX or TorchScript."""
    if to.lower() == "onnx":
        if not has_transformers():
            raise typer.Exit(code=1)
        import torch  # type: ignore
        mdl, tok, device = load_model_and_tokenizer(model)
        enc = tok(example_text, return_tensors="pt").to(device)
        export_onnx(mdl, (enc["input_ids"], enc.get("attention_mask")), output)
        console.print(f"Saved ONNX to {output}")
    elif to.lower() == "torchscript":
        import torch, os  # type: ignore
        mdl, tok, device = load_model_and_tokenizer(model)
        enc = tok(example_text, return_tensors="pt").to(device)
        traced = export_torchscript(mdl, (enc["input_ids"], enc.get("attention_mask")))
        out_dir = output
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "model.ts")
        traced.save(path)
        console.print(f"Saved TorchScript to {path}")
    else:
        console.print("Unknown export target")


@app.command()
def test():
    """Run built-in lightweight tests (no pytest required)."""
    import os, sys
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    script = os.path.join(root, "scripts", "test_runner.py")
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    res = subprocess.run([sys.executable, script], cwd=root, env=env)
    raise typer.Exit(code=res.returncode)


if __name__ == "__main__":
    app()
