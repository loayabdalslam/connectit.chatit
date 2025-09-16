import click
import asyncio
from rich.console import Console

from .hf import has_transformers, has_datasets, load_model_and_tokenizer, export_torchscript, export_onnx
from .p2p import generate_join_link, parse_join_link
from .p2p_runtime import run_p2p_node, P2PNode

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """ConnectIT CLI (prototype)"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option('--model', default='distilgpt2', help='HF Causal LM model name')
@click.option('--price-per-token', default=0.0, type=float, help='Price per output token')
@click.option('--host', default='0.0.0.0', help='Bind host')
@click.option('--port', default=4001, type=int, help='Bind port')
@click.option('--bootstrap-link', default='', help='p2pnet:// join link or ws://host:port')
def deploy_hf(model, price_per_token, host, port, bootstrap_link):
    """Deploy a Hugging Face text-generation service on the P2P network."""
    asyncio.run(run_p2p_node(
        host=host, 
        port=port, 
        bootstrap_link=(bootstrap_link or None), 
        model_name=model, 
        price_per_token=price_per_token
    ))


@cli.command()
@click.argument('prompt')
@click.option('--model', default='distilgpt2', help='Model name to request')
@click.option('--bootstrap-link', default='', help='Join link or ws:// peer to bootstrap')
@click.option('--max-new-tokens', default=32, type=int, help='Max new tokens')
def p2p_request(prompt, model, bootstrap_link, max_new_tokens):
    """Join P2P and request a generation from the cheapest/lowest-latency provider."""
    async def _run():
        console.print("\n🚀 [bold cyan]ConnectIT P2P Request Client[/bold cyan]")
        console.print(f"📝 Prompt: [yellow]{prompt}[/yellow]")
        console.print(f"🤖 Model: [green]{model}[/green]")
        console.print(f"🔢 Max tokens: [blue]{max_new_tokens}[/blue]")
        
        console.print("\n🔧 [bold]Starting P2P node...[/bold]")
        node = P2PNode(host="127.0.0.1", port=0)
        await node.start()
        console.print(f"✓ [green]P2P node started[/green] - {node.addr}")
        
        if bootstrap_link:
            console.print(f"\n🔗 [bold]Connecting to bootstrap...[/bold]")
            console.print(f"   Link: [cyan]{bootstrap_link}[/cyan]")
            await node.connect_bootstrap(bootstrap_link)
            console.print("✓ [green]Bootstrap connection established[/green]")
        else:
            console.print("\n⚠️  [yellow]No bootstrap link provided - running in isolated mode[/yellow]")
        
        console.print("\n🔍 [bold]Discovering providers...[/bold]")
        console.print(f"🔍 Current peers: {len(node.peers)}")
        
        # Wait longer and check multiple times for service discovery
        for attempt in range(1, 6):
            with console.status(f"[bold green]Searching for providers... (attempt {attempt}/5)", spinner="dots"):
                await asyncio.sleep(2)
            
            providers = [p for p in node.providers.values() if model in p.get('hf', {}).get('models', [])]
            console.print(f"📊 Attempt {attempt}: Found [bold]{len(providers)}[/bold] providers for model '{model}'")
            
            if providers:
                break
            
            if attempt < 5:
                console.print(f"🔄 No providers found yet, retrying... ({5-attempt} attempts left)")
        
        if providers:
            for i, provider in enumerate(providers, 1):
                price = provider.get('price_per_token', 'Unknown')
                console.print(f"   {i}. Provider: [cyan]{provider.get('peer_id', 'Unknown')}[/cyan] - Price: [green]{price}[/green]")
        
        best = node.pick_provider(model)
        if not best:
            console.print("\n❌ [bold red]No provider found for model[/bold red]")
            console.print("💡 [yellow]Tips:[/yellow]")
            console.print("   • Make sure the model name matches exactly")
            console.print("   • Check if the bootstrap link is correct")
            console.print("   • Ensure a provider is running with this model")
            return
        
        pid, provider_info = best
        price = provider_info.get('price_per_token', 'Unknown')
        console.print(f"\n🎯 [bold green]Selected provider:[/bold green] [cyan]{pid}[/cyan] (Price: [green]{price}[/green])")
        
        console.print("\n📡 [bold]Requesting generation...[/bold]")
        with console.status("[bold green]Generating response...", spinner="bouncingBar"):
            res = await node.request_generation(pid, prompt, max_new_tokens=max_new_tokens, model_name=model)
        
        console.print("\n🎉 [bold green]Generation completed![/bold green]")
        console.print("\n📄 [bold cyan]Response:[/bold cyan]")
        console.print(f"[white on blue] {res} [/white on blue]")
        
        await node.stop()
        console.print("\n✓ [green]P2P node stopped[/green]")
    
    asyncio.run(_run())


if __name__ == "__main__":
    cli()
