from typing import Optional, List
import typer
try:
    # Workaround for help formatting issue on some Windows/click combos
    import typer.rich_utils as _typer_rich_utils  # type: ignore
    _typer_rich_utils.USE_RICH = False
except Exception:
    pass
from rich.console import Console

from .console import run_console
from .coordinator import run_coordinator
from .node import run_node
from .hf import has_transformers, has_datasets, load_model_and_tokenizer, export_torchscript, export_onnx
from .p2p import generate_join_link, parse_join_link
from .p2p_runtime import run_p2p_node, P2PNode
import asyncio
from .auth import require_login

app = typer.Typer(add_completion=False, help="ConnectIT CLI (prototype)")
console = Console()


@app.command("console")
def console_app():
    """Open the interactive console (login, offers, jobs)."""
    run_console()


@app.command()
def coordinator(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8765, help="Bind port"),
):
    """Start the coordinator service (registry + orchestrator)."""
    require_login()
    run_coordinator(host=host, port=port)


@app.command()
def node(
    coordinator: str = typer.Option("ws://127.0.0.1:8765", help="Coordinator WS URL"),
    name: Optional[str] = typer.Option(None, help="Node name"),
    price: float = typer.Option(0.0, help="Price per unit (demo)"),
):
    """Start a node agent that connects to a coordinator."""
    require_login()
    run_node(coordinator_url=coordinator, node_name=name, price=price)


@app.command()
def export(
    to: str = typer.Option("onnx", help="onnx|torchscript"),
    model: str = typer.Option("distilbert-base-uncased", help="HF model name"),
    output: str = typer.Option("model.onnx", help="Output file for ONNX or directory for TorchScript save"),
    example_text: str = typer.Option("Hello ConnectIT", help="Example text to trace/export"),
):
    """Export a Hugging Face model locally to ONNX or TorchScript."""
    require_login()
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


@app.command("help")
def help_cmd():
    """Show usage help (workaround for some environments)."""
    console.print("Commands:")
    console.print("- coordinator: Start coordinator server")
    console.print("  python -m connectit coordinator --host 0.0.0.0 --port 8765")
    console.print("- node: Start a node agent")
    console.print("  python -m connectit node --coordinator \"ws://127.0.0.1:8765\" --name node1 --price 0.01")
    console.print("- console: Open interactive console")
    console.print("  python -m connectit console")
    console.print("- export: Export HF model to ONNX/TorchScript")
    console.print("  python -m connectit export --to onnx --model distilbert-base-uncased --output model.onnx")
    console.print("- test: Run built-in tests with logs")
    console.print("  python -m connectit test")
    console.print("- p2p_link: Create a join link")
    console.print("  python -m connectit p2p_link --network llmnet --model demo --hash deadbeef --bootstrap_csv \"/ip4/1.2.3.4/tcp/4001/p2p/QmPeer\"")


@app.command()
def p2p_link(
    network: str = typer.Option("llmnet", help="Network ID"),
    model: str = typer.Option("demo", help="Model identifier"),
    hash_hex: str = typer.Option("deadbeef", help="Content hash hex"),
    bootstrap_csv: str = typer.Option("", help="Comma-separated bootstrap peers (multiaddr or host:port)"),
):
    """Generate or parse a P2P join link."""
    require_login()
    boots = [b for b in (s.strip() for s in bootstrap_csv.split(",")) if b]
    link = generate_join_link(network, model, hash_hex, boots)
    console.print(f"Link: {link}")
    info = parse_join_link(link)
    console.print(f"Parsed: {info}")


@app.command()
def p2p(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(4001, help="Bind port"),
    bootstrap_link: str = typer.Option("", help="p2pnet:// join link or ws://host:port"),
):
    """Run a P2P node (discovery + optional services)."""
    require_login()
    asyncio.run(run_p2p_node(host=host, port=port, bootstrap_link=(bootstrap_link or None)))


@app.command()
def deploy_hf(
    model: str = typer.Option("distilgpt2", help="HF Causal LM model name"),
    price_per_token: float = typer.Option(0.0, help="Price per output token"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(4001, help="Bind port"),
    bootstrap_link: str = typer.Option("", help="p2pnet:// join link or ws://host:port"),
):
    """Deploy a Hugging Face text-generation service on the P2P network."""
    require_login()
    asyncio.run(run_p2p_node(host=host, port=port, bootstrap_link=(bootstrap_link or None), model_name=model, price_per_token=price_per_token))


@app.command()
def p2p_request(
    prompt: str = typer.Argument(..., help="Prompt text"),
    model: str = typer.Option("distilgpt2", help="Model name to request"),
    bootstrap_link: str = typer.Option("", help="Join link or ws:// peer to bootstrap"),
    max_new_tokens: int = typer.Option(32, help="Max new tokens"),
):
    """Join P2P and request a generation from the cheapest/lowest-latency provider."""
    require_login()
    async def _run():
        node = P2PNode(host="127.0.0.1", port=0)
        await node.start()
        if bootstrap_link:
            await node.connect_bootstrap(bootstrap_link)
        # Wait a moment to discover providers
        await asyncio.sleep(2)
        best = node.pick_provider(model)
        if not best:
            console.print("[red]No provider found for model[/red]")
            return
        pid, _ = best
        res = await node.request_generation(pid, prompt, max_new_tokens=max_new_tokens, model_name=model)
        console.print(res)
    asyncio.run(_run())


if __name__ == "__main__":
    app()
