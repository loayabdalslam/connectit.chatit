#!/usr/bin/env python3
"""
ConnectIT P2P Request Demo Script

This script demonstrates how to use ConnectIT programmatically to request
text generation from the P2P network. It includes examples for:
- Basic text generation
- Batch processing multiple prompts
- Error handling and provider discovery
- Custom configuration options

Usage:
    python p2p_request_demo.py

Prerequisites:
    1. Install ConnectIT: pip install -e .
    2. Have at least one provider running:
       python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.002 --port 4001
"""

import asyncio
import time
from typing import List, Dict, Optional, Any
from connectit.p2p_runtime import P2PNode


class ConnectITP2PClient:
    """A client wrapper for ConnectIT P2P requests."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.node = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.node = P2PNode(host=self.host, port=self.port)
        await self.node.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.node:
            await self.node.stop()
    
    async def connect_to_network(self, bootstrap_link: str, discovery_time: float = 2.0):
        """Connect to the P2P network and wait for provider discovery."""
        if not self.node:
            raise RuntimeError("Client not started. Use async context manager.")
        
        await self.node.connect_bootstrap(bootstrap_link)
        print(f"Connected to network via {bootstrap_link}")
        print(f"Waiting {discovery_time}s for provider discovery...")
        await asyncio.sleep(discovery_time)
    
    async def request_generation(
        self, 
        prompt: str, 
        model_name: str = "distilgpt2", 
        max_new_tokens: int = 32
    ) -> Optional[Dict[str, Any]]:
        """Request text generation from the best available provider."""
        if not self.node:
            raise RuntimeError("Client not started. Use async context manager.")
        
        # Find the best provider for the model
        best = self.node.pick_provider(model_name)
        if not best:
            print(f"❌ No provider found for model: {model_name}")
            return None
        
        provider_id, provider_info = best
        print(f"🎯 Using provider {provider_id} (price: {provider_info.get('price_per_token', 'unknown')})")
        
        # Request generation
        start_time = time.time()
        result = await self.node.request_generation(
            provider_id, 
            prompt, 
            max_new_tokens=max_new_tokens, 
            model_name=model_name
        )
        end_time = time.time()
        
        if result:
            result['request_time'] = end_time - start_time
        
        return result
    
    def list_available_providers(self) -> Dict[str, List[Dict]]:
        """List all available providers by model."""
        if not self.node:
            raise RuntimeError("Client not started. Use async context manager.")
        
        providers_by_model = {}
        for provider_id, provider_info in self.node.providers.items():
            models = provider_info.get('models', [])
            for model in models:
                if model not in providers_by_model:
                    providers_by_model[model] = []
                providers_by_model[model].append({
                    'provider_id': provider_id,
                    'price_per_token': provider_info.get('price_per_token', 'unknown'),
                    'latency_ms': provider_info.get('latency_ms', 'unknown')
                })
        
        return providers_by_model


async def basic_example():
    """Basic example: single text generation request."""
    print("\n=== Basic Example ===")
    
    async with ConnectITP2PClient() as client:
        # Connect to network (replace with your bootstrap link)
        await client.connect_to_network("ws://127.0.0.1:4001")
        
        # Request text generation
        prompt = "The future of artificial intelligence is"
        result = await client.request_generation(
            prompt=prompt,
            model_name="distilgpt2",
            max_new_tokens=50
        )
        
        if result:
            print(f"\n📝 Prompt: {prompt}")
            print(f"🤖 Generated: {result.get('generated_text', 'N/A')}")
            print(f"⏱️  Request time: {result.get('request_time', 0):.2f}s")
            print(f"💰 Cost: {result.get('total_cost', 'N/A')}")
        else:
            print("❌ Generation failed")


async def batch_processing_example():
    """Batch processing example: multiple prompts."""
    print("\n=== Batch Processing Example ===")
    
    prompts = [
        "Hello, how are you?",
        "Tell me a short story about a robot.",
        "What is the meaning of life?",
        "Explain quantum computing in simple terms."
    ]
    
    async with ConnectITP2PClient() as client:
        await client.connect_to_network("ws://127.0.0.1:4001")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🔄 Processing prompt {i}/{len(prompts)}: {prompt[:30]}...")
            
            result = await client.request_generation(
                prompt=prompt,
                model_name="distilgpt2",
                max_new_tokens=32
            )
            
            results.append({
                'prompt': prompt,
                'result': result,
                'success': result is not None
            })
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 Batch Summary: {successful}/{len(results)} successful")
        
        for i, item in enumerate(results, 1):
            status = "✅" if item['success'] else "❌"
            print(f"{status} {i}. {item['prompt'][:40]}...")
            if item['success']:
                generated = item['result'].get('generated_text', 'N/A')
                print(f"    → {generated[:60]}...")


async def provider_discovery_example():
    """Provider discovery example: list available providers."""
    print("\n=== Provider Discovery Example ===")
    
    async with ConnectITP2PClient() as client:
        await client.connect_to_network("ws://127.0.0.1:4001", discovery_time=3.0)
        
        providers = client.list_available_providers()
        
        if not providers:
            print("❌ No providers found on the network")
            return
        
        print(f"🌐 Found providers for {len(providers)} model(s):")
        
        for model, provider_list in providers.items():
            print(f"\n📦 Model: {model}")
            for provider in provider_list:
                print(f"  🔗 Provider: {provider['provider_id'][:12]}...")
                print(f"     💰 Price: {provider['price_per_token']} per token")
                print(f"     ⚡ Latency: {provider['latency_ms']}ms")


async def error_handling_example():
    """Error handling example: dealing with network issues."""
    print("\n=== Error Handling Example ===")
    
    async with ConnectITP2PClient() as client:
        try:
            # Try to connect to a non-existent bootstrap
            print("🔄 Attempting to connect to non-existent bootstrap...")
            await client.connect_to_network("ws://nonexistent:9999", discovery_time=1.0)
            
            # This will likely fail
            result = await client.request_generation(
                "Hello world",
                model_name="nonexistent-model"
            )
            
            if not result:
                print("❌ No providers available for the requested model")
                print("💡 Make sure at least one provider is running:")
                print("   python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.002 --port 4001")
        
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("💡 Check your bootstrap link and network connectivity")


async def custom_configuration_example():
    """Custom configuration example: different models and parameters."""
    print("\n=== Custom Configuration Example ===")
    
    configurations = [
        {"model": "distilgpt2", "max_tokens": 20, "prompt": "Once upon a time"},
        {"model": "gpt2", "max_tokens": 30, "prompt": "The best programming language is"},
        {"model": "microsoft/DialoGPT-medium", "max_tokens": 25, "prompt": "How's the weather today?"}
    ]
    
    async with ConnectITP2PClient() as client:
        await client.connect_to_network("ws://127.0.0.1:4001")
        
        for config in configurations:
            print(f"\n🎯 Testing {config['model']} with {config['max_tokens']} tokens")
            
            result = await client.request_generation(
                prompt=config['prompt'],
                model_name=config['model'],
                max_new_tokens=config['max_tokens']
            )
            
            if result:
                print(f"✅ Success: {result.get('generated_text', 'N/A')[:80]}...")
            else:
                print(f"❌ No provider available for {config['model']}")


async def main():
    """Run all examples."""
    print("🚀 ConnectIT P2P Request Demo")
    print("==============================")
    print("\n💡 Make sure you have a provider running:")
    print("   python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.002 --port 4001")
    print("\n⏳ Starting examples in 3 seconds...")
    await asyncio.sleep(3)
    
    try:
        await basic_example()
        await provider_discovery_example()
        await batch_processing_example()
        await custom_configuration_example()
        await error_handling_example()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())