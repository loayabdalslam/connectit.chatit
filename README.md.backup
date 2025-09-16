ConnectIT
==========

<a href="https://www.producthunt.com/products/connect-it?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-connect&#0045;it" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1016671&theme=neutral&t=1758001359763" alt="Connect&#0032;it&#0032; - Torrent&#0032;Like&#0032;Protocol&#0032;for&#0032;Deployment&#0032;LLM&#0032;Models | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

# ConnectIT

[![PyPI version](https://badge.fury.io/py/connectit.svg)](https://badge.fury.io/py/connectit)
[![Python versions](https://img.shields.io/pypi/pyversions/connectit.svg)](https://pypi.org/project/connectit/)
[![Downloads](https://pepy.tech/badge/connectit)](https://pepy.tech/project/connectit)
[![License](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/connectit/connectit/ci.yml?branch=main)](https://github.com/connectit/connectit/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/connectit/badge/?version=latest)](https://connectit.readthedocs.io/en/latest/?badge=latest)

A peer-to-peer network for deploying and accessing Hugging Face language models. ConnectIT allows you to deploy any Hugging Face model as a service on a decentralized network and request text generation from the cheapest/lowest-latency providers.

## ✨ Features

- 🌐 **Decentralized P2P Network**: No central server required
- 🤖 **Hugging Face Integration**: Deploy any HF model instantly
- 💰 **Cost Optimization**: Automatically selects cheapest providers
- ⚡ **Low Latency**: Smart provider selection based on response time
- 🔒 **Secure**: Custom licensing with commercial protection
- 🚀 **Easy Setup**: One-command deployment and requests

## 📊 Stats & Requirements

![GitHub stars](https://img.shields.io/github/stars/connectit/connectit?style=social)
![GitHub forks](https://img.shields.io/github/forks/connectit/connectit?style=social)
![GitHub issues](https://img.shields.io/github/issues/connectit/connectit)
![GitHub pull requests](https://img.shields.io/github/issues-pr/connectit/connectit)

**Requirements:**
- Python 3.9+
- 2GB+ RAM recommended
- Network connectivity for P2P operations

## Installation

### From PyPI (Recommended)

![PyPI - Status](https://img.shields.io/pypi/status/connectit)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/connectit)
![PyPI - Format](https://img.shields.io/pypi/format/connectit)

```bash
# Basic installation
pip install connectit

# With Hugging Face support
pip install connectit[hf]

# With all optional dependencies
pip install connectit[all]
```

### From Source

![GitHub repo size](https://img.shields.io/github/repo-size/connectit/connectit)
![GitHub code size](https://img.shields.io/github/languages/code-size/connectit/connectit)

```bash
git clone <repository-url>
cd connectit
pip install -e .
```

## 🚀 Quick Start

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Terminal](https://img.shields.io/badge/terminal-bash%20%7C%20powershell%20%7C%20cmd-blue)

**Prerequisites:** Python 3.9+, `pip`

1) Install ConnectIT:

   ```bash
   pip install -e .
   ```

   For full functionality with Hugging Face models:

   ```bash
   pip install -e .[all]
   ```

2) Deploy a Hugging Face model:

   ```bash
   python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.002 --host 127.0.0.1 --port 4334
   ```

3) Request text generation from another terminal:

   ```bash
   python -m connectit p2p-request "Hello world" --bootstrap-link "p2pnet://join?network=connectit&model=distilgpt2&hash=32a0fa785bfb95c97ced872ac200560ffface58c574c775b7fd8304494a4d4e3&bootstrap=d3M6Ly8xMjcuMC4wLjE6NDMzNA=="
   ```

   **Note:** Use the join link displayed by the provider, not the raw WebSocket address.

Commands
--------

### deploy-hf

Deploy a Hugging Face text-generation model as a service on the P2P network.

```bash
python -m connectit deploy-hf --model MODEL_NAME --price-per-token PRICE --host HOST --port PORT
```

**Parameters:**
- `--model`: Hugging Face model name (e.g., `distilgpt2`, `gpt2`, `microsoft/DialoGPT-medium`)
- `--price-per-token`: Price per output token (float, e.g., `0.002`)
- `--host`: Bind host address (default: `0.0.0.0`)
- `--port`: Bind port (default: `4001`)

**Example:**
```bash
python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.002 --host 127.0.0.1 --port 4334
```

The provider will display a join link like:
```
🔗 Join link: p2pnet://join?network=connectit&model=distilgpt2&hash=...&bootstrap=...
```

### p2p-request

Request text generation from providers on the P2P network.

```bash
python -m connectit p2p-request PROMPT [OPTIONS]
```

**Parameters:**
- `PROMPT`: Text prompt for generation (required)
- `--model`: Model name to request (default: `distilgpt2`)
- `--bootstrap-link`: P2P network join link from a provider (required)
- `--max-new-tokens`: Maximum tokens to generate (default: `32`)

**Example:**
```bash
python -m connectit p2p-request "Hello world" --bootstrap-link "p2pnet://join?network=connectit&model=distilgpt2&hash=32a0fa785bfb95c97ced872ac200560ffface58c574c775b7fd8304494a4d4e3&bootstrap=d3M6Ly8xMjcuMC4wLjE6NDMzNA=="
```

**Important:** Always use the complete `p2pnet://` join link provided by the provider, not raw WebSocket addresses.

Troubleshooting
--------------

### "No provider found for model"

**Possible causes:**
- Model name mismatch between request and provider
- Bootstrap link is incorrect or expired
- Provider is not running or unreachable
- Network connectivity issues

**Solutions:**
1. Verify the model name matches exactly (case-sensitive)
2. Copy the complete join link from the provider output
3. Ensure the provider is running and shows "ready to accept connections"
4. Check firewall settings if connecting across networks

### "Failed to retrieve command output"

**Possible causes:**
- Terminal encoding issues
- Long-running process conflicts

**Solutions:**
1. Run commands in separate terminals
2. Ensure proper terminal encoding (UTF-8)
3. Restart terminals if needed

### Connection Issues

**Symptoms:**
- Peer connection failures
- Bootstrap connection timeouts
- Generation request failures

**Solutions:**
1. Verify both provider and client are on the same network
2. Check port availability and firewall rules
3. Try different host/port combinations
4. Ensure provider is fully loaded before making requests

## 📖 Documentation & Support

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://connectit.readthedocs.io/)
[![Wiki](https://img.shields.io/badge/wiki-available-blue)](https://github.com/connectit/connectit/wiki)
[![FAQ](https://img.shields.io/badge/FAQ-available-orange)](https://github.com/connectit/connectit/wiki/FAQ)
[![Community](https://img.shields.io/badge/community-discussions-purple)](https://github.com/connectit/connectit/discussions)

## 🤝 Contributing

[![Contributors](https://img.shields.io/github/contributors/connectit/connectit)](https://github.com/connectit/connectit/graphs/contributors)
[![Good first issues](https://img.shields.io/github/issues/connectit/connectit/good%20first%20issue)](https://github.com/connectit/connectit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
[![Help wanted](https://img.shields.io/github/issues/connectit/connectit/help%20wanted)](https://github.com/connectit/connectit/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under a custom license that permits non-commercial use only. For commercial use, please contact: loaiabdalslam@gmail.com

See the [LICENSE](LICENSE) file for full details.

## Architecture

Request text generation from the P2P network. Automatically selects the cheapest/lowest-latency provider for the specified model.

```bash
python -m connectit p2p-request "PROMPT_TEXT" --model MODEL_NAME --bootstrap-link BOOTSTRAP_LINK
```

**Parameters:**
- `PROMPT_TEXT`: The text prompt for generation (required)
- `--model`: Model name to request (default: `distilgpt2`)
- `--bootstrap-link`: Bootstrap link to join the network (required for discovery)
- `--max-new-tokens`: Maximum new tokens to generate (default: `32`)

**Examples:**

```bash
# Basic text generation
python -m connectit p2p-request "Hello world" --model distilgpt2 --bootstrap-link ws://127.0.0.1:4334

# Longer generation with more tokens
python -m connectit p2p-request "The future of AI is" --model distilgpt2 --max-new-tokens 50 --bootstrap-link ws://127.0.0.1:4334

# Question answering
python -m connectit p2p-request "What is artificial intelligence?" --model distilgpt2 --max-new-tokens 100 --bootstrap-link ws://127.0.0.1:4334

# Creative writing prompt
python -m connectit p2p-request "Once upon a time in a distant galaxy" --model distilgpt2 --max-new-tokens 75 --bootstrap-link ws://127.0.0.1:4334
```

## Real-World Usage Scenarios

### Scenario 1: Local Development and Testing

**Step 1:** Start a local provider in one terminal:
```bash
python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.002 --host 127.0.0.1 --port 4334
```

**Step 2:** Test requests from another terminal:
```bash
# Simple test
python -m connectit p2p-request "Hello, world!" --model distilgpt2 --bootstrap-link ws://127.0.0.1:4334

# Check response quality
python -m connectit p2p-request "Explain machine learning in simple terms" --model distilgpt2 --max-new-tokens 50 --bootstrap-link ws://127.0.0.1:4334
```

### Scenario 2: Multi-Provider Network

**Provider A (Fast, Expensive):**
```bash
python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.005 --host 0.0.0.0 --port 4001
```

**Provider B (Slow, Cheap):**
```bash
python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.001 --host 0.0.0.0 --port 4002 --bootstrap-link ws://localhost:4001
```

**Client requests automatically select the best provider:**
```bash
# Will choose Provider B (cheaper)
python -m connectit p2p-request "Generate a short story" --model distilgpt2 --bootstrap-link ws://localhost:4001
```

### Scenario 3: Different Models for Different Tasks

**Deploy specialized models:**
```bash
# Terminal 1: General text generation
python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.002 --port 4001

# Terminal 2: Conversational AI
python -m connectit deploy-hf --model microsoft/DialoGPT-small --price-per-token 0.003 --port 4002 --bootstrap-link ws://127.0.0.1:4001

# Terminal 3: Code generation
python -m connectit deploy-hf --model microsoft/CodeGPT-small-py --price-per-token 0.004 --port 4003 --bootstrap-link ws://127.0.0.1:4001
```

**Use appropriate model for each task:**
```bash
# General text
python -m connectit p2p-request "Write a product description" --model distilgpt2 --bootstrap-link ws://127.0.0.1:4001

# Conversation
python -m connectit p2p-request "How are you feeling today?" --model microsoft/DialoGPT-small --bootstrap-link ws://127.0.0.1:4001

# Code
python -m connectit p2p-request "def fibonacci(n):" --model microsoft/CodeGPT-small-py --bootstrap-link ws://127.0.0.1:4001
```

Programmatic Usage
------------------

You can use ConnectIT programmatically in your Python scripts:

```python
import asyncio
from connectit.p2p_runtime import P2PNode

async def request_generation(prompt, model_name="distilgpt2", bootstrap_link=None):
    """Request text generation programmatically."""
    node = P2PNode(host="127.0.0.1", port=0)
    await node.start()
    
    if bootstrap_link:
        await node.connect_bootstrap(bootstrap_link)
    
    # Wait for provider discovery
    await asyncio.sleep(2)
    
    # Find the best provider
    best = node.pick_provider(model_name)
    if not best:
        print(f"No provider found for model: {model_name}")
        return None
    
    provider_id, _ = best
    result = await node.request_generation(
        provider_id, 
        prompt, 
        max_new_tokens=32, 
        model_name=model_name
    )
    
    await node.stop()
    return result

# Usage
result = asyncio.run(request_generation(
    "Hello world", 
    model_name="distilgpt2",
    bootstrap_link="ws://127.0.0.1:4001"
))
print(result)
```

### Script Integration Examples

**Batch Processing:**

```python
import asyncio
from connectit.p2p_runtime import P2PNode

async def batch_generate(prompts, model_name="distilgpt2", bootstrap_link=None):
    """Generate text for multiple prompts."""
    node = P2PNode(host="127.0.0.1", port=0)
    await node.start()
    
    if bootstrap_link:
        await node.connect_bootstrap(bootstrap_link)
    await asyncio.sleep(2)  # Discovery time
    
    results = []
    for prompt in prompts:
        best = node.pick_provider(model_name)
        if best:
            provider_id, _ = best
            result = await node.request_generation(provider_id, prompt, model_name=model_name)
            results.append({"prompt": prompt, "result": result})
        else:
            results.append({"prompt": prompt, "result": None})
    
    await node.stop()
    return results

# Usage
prompts = ["Hello", "How are you?", "Tell me a story"]
results = asyncio.run(batch_generate(prompts, bootstrap_link="ws://127.0.0.1:4001"))
for item in results:
    print(f"Prompt: {item['prompt']}")
    print(f"Result: {item['result']}")
    print("---")
```

**Web Service Integration:**

```python
from flask import Flask, request, jsonify
import asyncio
from connectit.p2p_runtime import P2PNode

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt')
    model = data.get('model', 'distilgpt2')
    bootstrap_link = data.get('bootstrap_link')
    
    async def _generate():
        node = P2PNode(host="127.0.0.1", port=0)
        await node.start()
        if bootstrap_link:
            await node.connect_bootstrap(bootstrap_link)
        await asyncio.sleep(2)
        
        best = node.pick_provider(model)
        if not best:
            return None
        
        provider_id, _ = best
        result = await node.request_generation(provider_id, prompt, model_name=model)
        await node.stop()
        return result
    
    result = asyncio.run(_generate())
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
```

Deploying Hugging Face Models
-----------------------------

### Supported Models

ConnectIT supports any Hugging Face Causal Language Model. Popular choices include:

- **GPT-2 family**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **DistilGPT-2**: `distilgpt2` (smaller, faster)
- **DialoGPT**: `microsoft/DialoGPT-small`, `microsoft/DialoGPT-medium`, `microsoft/DialoGPT-large`
- **CodeGPT**: `microsoft/CodeGPT-small-py`
- **GPT-Neo**: `EleutherAI/gpt-neo-125M`, `EleutherAI/gpt-neo-1.3B`
- **Custom models**: Any compatible model from Hugging Face Hub

### Model Deployment Best Practices

1. **Choose appropriate pricing**: Set `--price-per-token` based on model size and computational cost
2. **Resource considerations**: Larger models require more memory and compute time
3. **Network setup**: Ensure your host/port is accessible to other network participants
4. **Model caching**: First deployment will download the model; subsequent runs use cached version

### Advanced Deployment

**Custom model with specific configuration:**

```bash
# Deploy a larger model with higher pricing
python -m connectit deploy-hf \
  --model EleutherAI/gpt-neo-1.3B \
  --price-per-token 0.01 \
  --host 0.0.0.0 \
  --port 4001 \
  --bootstrap-link ws://bootstrap.mynetwork.com:4001
```

**Multiple model deployment:**

You can run multiple instances on different ports to serve different models:

```bash
# Terminal 1: Deploy DistilGPT-2
python -m connectit deploy-hf --model distilgpt2 --price-per-token 0.001 --port 4001

# Terminal 2: Deploy GPT-2 Medium
python -m connectit deploy-hf --model gpt2-medium --price-per-token 0.005 --port 4002

# Terminal 3: Deploy DialoGPT
python -m connectit deploy-hf --model microsoft/DialoGPT-medium --price-per-token 0.003 --port 4003
```

Troubleshooting
---------------

**Common Issues:**

1. **Command not found**: Use `python -m connectit` instead of `connectit` if the command is not in PATH
2. **Model download fails**: Ensure internet connection and sufficient disk space
3. **No providers found**: Check bootstrap-link and ensure at least one provider is running
4. **Port conflicts**: Use different ports for multiple deployments
5. **Memory issues**: Use smaller models like `distilgpt2` for limited resources
6. **Connection timeout**: Wait a few seconds after starting providers before making requests
7. **Concurrency errors**: Fixed in latest version - providers now handle multiple simultaneous requests

**Dependencies:**

- Core functionality: `typer`, `rich`, `websockets`, `numpy`
- Hugging Face models: `transformers`, `torch`
- Full features: Install with `pip install -e .[all]`

**Performance Tips:**

- Use GPU-enabled PyTorch for faster inference on compatible hardware
- Choose model size based on available system resources
- Consider network latency when selecting bootstrap peers
- Monitor system resources during model deployment
- Start with `distilgpt2` for testing - it's fast and lightweight
- Use `--max-new-tokens` to control response length and generation time
- Multiple providers of the same model create automatic load balancing

License
-------

This is a prototype implementation. See license file for details.
