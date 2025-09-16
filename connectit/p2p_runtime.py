from __future__ import annotations
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol
from rich.console import Console

from .p2p import parse_join_link, sha256_hex_bytes
from .utils import new_id
from .pieces import split_pieces, piece_hashes


console = Console()


class P2PNode:
    def __init__(self, host: str = "0.0.0.0", port: int = 4001):
        self.host = host
        self.port = port
        self.peer_id = new_id("peer")
        self.addr = f"ws://{host}:{port}"
        self.server: Optional[websockets.server.Serve] = None
        self.peers: Dict[str, Dict[str, Any]] = {}  # pid -> {ws, addr, last_pong_ms}
        self.services: Dict[str, Dict[str, Any]] = {}  # svc_name -> data
        self.providers: Dict[str, Dict[str, Any]] = {}  # pid -> {'hf': {'models': [...], 'price_per_token': float}, 'latency_ms': float}
        self.pieces: Dict[str, Dict[str, Any]] = {}  # content_hash -> { 'pieces': List[bytes], 'hashes': List[str] }
        self._lock = asyncio.Lock()
        self._pending_requests: Dict[str, asyncio.Future] = {}

    async def start(self):
        async def handler(ws: WebSocketServerProtocol):
            await self._handle(ws)
        console.log(f"[cyan]P2P listening[/cyan] on ws://{self.host}:{self.port}")
        self.server = await websockets.serve(handler, self.host, self.port, max_size=32*1024*1024)

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def connect_bootstrap(self, link_or_addr: str):
        # Accept either p2pnet link or raw ws://host:port
        addrs: List[str]
        if link_or_addr.startswith("p2pnet://"):
            parsed = parse_join_link(link_or_addr)
            addrs = [a for a in parsed.get("bootstrap", [])]
        else:
            addrs = [link_or_addr]
        for addr in addrs:
            try:
                await self._connect_peer(addr)
            except Exception as e:
                console.log(f"[yellow]Bootstrap connect failed[/yellow] {addr}: {e}")

    async def _connect_peer(self, addr: str):
        ws = await websockets.connect(addr, max_size=32*1024*1024)
        pid = new_id("peer")  # temporary until hello
        async with self._lock:
            self.peers[pid] = {"ws": ws, "addr": addr, "last_pong_ms": 0}
        await self._send(ws, {"type": "hello", "peer_id": self.peer_id, "addr": self.addr, "services": self.services})
        # We expect the remote to send us hello too.
        asyncio.create_task(self._peer_reader(ws))

    async def _peer_reader(self, ws: WebSocketClientProtocol | WebSocketServerProtocol):
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    print(f"[DEBUG] _peer_reader processing message: {data.get('type')} from {ws.remote_address}")
                except Exception as e:
                    print(f"[DEBUG] Failed to parse message from {ws.remote_address}: {e}")
                    continue
                await self._on_message(ws, data)
        except Exception as e:
            print(f"[DEBUG] _peer_reader exception from {ws.remote_address}: {e}")
        finally:
            print(f"[DEBUG] _peer_reader ending for {ws.remote_address}")
            await self._on_disconnect(ws)

    async def _on_disconnect(self, ws):
        async with self._lock:
            for pid, info in list(self.peers.items()):
                if info.get("ws") is ws:
                    self.peers.pop(pid, None)
                    self.providers.pop(pid, None)
                    console.log(f"[yellow]Peer disconnected[/yellow]: {pid}")
                    break

    async def _handle(self, ws: WebSocketServerProtocol):
        # Start reader
        console.log(f"[cyan]New connection from {ws.remote_address}[/cyan]")
        await self._peer_reader(ws)

    async def _send(self, ws, obj: Dict[str, Any]):
        await ws.send(json.dumps(obj))

    async def _broadcast(self, obj: Dict[str, Any]):
        async with self._lock:
            peers = [info.get("ws") for info in self.peers.values()]
        for ws in peers:
            try:
                await self._send(ws, obj)
            except Exception:
                pass

    async def _on_message(self, ws, data: Dict[str, Any]):
        t = data.get("type")
        print(f"[DEBUG] Received message type: {t} from {ws.remote_address}")
        if t == "hello":
            pid = data.get("peer_id")
            addr = data.get("addr")
            async with self._lock:
                # Replace placeholder entry for this ws
                found = None
                for k, v in list(self.peers.items()):
                    if v.get("ws") is ws:
                        found = k
                        break
                if found is not None and found != pid:
                    self.peers.pop(found, None)
                self.peers[pid] = {"ws": ws, "addr": addr, "last_pong_ms": 0}
            # Register provider info
            svcs = data.get("services", {})
            if "hf" in svcs:
                self.providers[pid] = {"hf": svcs["hf"], "latency_ms": None}
                print(f"[DEBUG] Registered provider {pid} with services: {svcs['hf']}")
            # Respond with our hello + peers list (send only serializable service info)
            serializable_services = {}
            for svc_name, svc_data in self.services.items():
                if svc_name == "hf":
                    serializable_services[svc_name] = {
                        "models": svc_data.get("models", []),
                        "price_per_token": svc_data.get("price_per_token", 0.0),
                        "max_new_tokens": svc_data.get("max_new_tokens", 32)
                    }
                else:
                    serializable_services[svc_name] = svc_data
            
            hello_msg = {"type": "hello", "peer_id": self.peer_id, "addr": self.addr, "services": serializable_services}
            await self._send(ws, hello_msg)
            print(f"[DEBUG] Sent hello to {pid} with our services: {serializable_services}")
            peer_list_msg = {"type": "peer_list", "peers": [v.get("addr") for v in self.peers.values() if v.get("addr")]}
            await self._send(ws, peer_list_msg)
            print(f"[DEBUG] Sent peer_list to {pid}")
            # Kick off ping
            ping_msg = {"type": "ping", "ts": time.time()}
            await self._send(ws, ping_msg)
            print(f"[DEBUG] Sent ping to {pid}")
        elif t == "peer_list":
            peers = data.get("peers", [])
            for addr in peers:
                # Connect to any new peer addresses
                if addr == self.addr:
                    continue
                if any(addr == v.get("addr") for v in self.peers.values()):
                    continue
                try:
                    await self._connect_peer(addr)
                except Exception:
                    pass
        elif t == "ping":
            await self._send(ws, {"type": "pong", "ts": data.get("ts")})
        elif t == "pong":
            rtt = (time.time() - float(data.get("ts", time.time()))) * 1000.0
            async with self._lock:
                for pid, info in self.peers.items():
                    if info.get("ws") is ws:
                        info["last_pong_ms"] = rtt
                        prov = self.providers.get(pid)
                        if prov is not None:
                            prov["latency_ms"] = rtt
                        break
        elif t == "service_announce":
            svc = data.get("service")
            meta = data.get("meta", {})
            async with self._lock:
                for pid, info in self.peers.items():
                    if info.get("ws") is ws:
                        if pid not in self.providers:
                            self.providers[pid] = {}
                        self.providers[pid][svc] = meta
                        break
        elif t == "gen_result":
            # Handle generation result responses
            rid = data.get("rid")
            if rid in self._pending_requests:
                future = self._pending_requests.pop(rid)
                if not future.cancelled():
                    future.set_result(data)
        elif t == "gen_request":
            # Incoming generation request for our local HF service
            rid = data.get("rid")
            prompt = data.get("prompt")
            max_new = int(data.get("max_new_tokens", 32))
            model = data.get("model")
            svc = self.services.get("hf")
            if not svc:
                await self._send(ws, {"type": "gen_result", "rid": rid, "error": "no_service"})
                return
            try:
                t0 = time.time()
                mdl = svc["model"]
                tok = svc["tokenizer"]
                device = svc["device"]
                from .hf import generate_text
                text = generate_text(mdl, tok, device, prompt, max_new)
                # token accounting: approximate cost by output minus input tokens
                in_tokens = len(tok.encode(prompt))
                out_tokens = len(tok.encode(text))
                new_tokens = max(0, out_tokens - in_tokens)
                latency_ms = int((time.time() - t0) * 1000.0)
                price = float(svc.get("price_per_token", 0.0))
                cost = price * new_tokens
                await self._send(ws, {"type": "gen_result", "rid": rid, "text": text, "tokens": new_tokens, "latency_ms": latency_ms, "price_per_token": price, "cost": cost})
            except Exception as e:
                await self._send(ws, {"type": "gen_result", "rid": rid, "error": str(e)})
        elif t == "piece_request":
            content_hash = data.get("content_hash")
            index = int(data.get("index", -1))
            entry = self.pieces.get(content_hash)
            if not entry or index < 0:
                await self._send(ws, {"type": "piece_data", "content_hash": content_hash, "index": index, "error": "not_found"})
                return
            parts = entry.get("pieces", [])
            if index >= len(parts):
                await self._send(ws, {"type": "piece_data", "content_hash": content_hash, "index": index, "error": "out_of_range"})
                return
            B = parts[index]
            await self._send(ws, {"type": "piece_data", "content_hash": content_hash, "index": index, "data": base64.b64encode(B).decode("ascii")})
        # ignore others

    async def add_hf_service(self, model_name: str, price_per_token: float, max_new_tokens: int = 32):
        # Load HF model locally and announce
        from .hf import load_model_and_tokenizer
        mdl, tok, device = load_model_and_tokenizer(model_name)
        self.services["hf"] = {
            "models": [model_name],
            "price_per_token": float(price_per_token),
            "model": mdl,
            "tokenizer": tok,
            "device": device,
            "max_new_tokens": int(max_new_tokens),
        }
        # Broadcast service announcement with serializable data only
        await self._broadcast({"type": "service_announce", "service": "hf", "meta": {"models": [model_name], "price_per_token": float(price_per_token)}})

    def share_blob(self, data: bytes, piece_size: int = 1024 * 64) -> Dict[str, Any]:
        pieces = split_pieces(data, piece_size)
        hashes = piece_hashes(pieces)
        content_hash = sha256_hex_bytes(b"".join(pieces))
        self.pieces[content_hash] = {"pieces": pieces, "hashes": hashes, "piece_size": piece_size}
        return {"content_hash": content_hash, "hashes": hashes, "piece_size": piece_size, "num_pieces": len(pieces)}

    async def fetch_piece(self, peer_id: str, content_hash: str, index: int) -> Optional[bytes]:
        info = self.peers.get(peer_id)
        if not info:
            return None
        ws = info.get("ws")
        req_id = new_id("piece")
        await self._send(ws, {"type": "piece_request", "req_id": req_id, "content_hash": content_hash, "index": index})
        while True:
            raw = await ws.recv()
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if data.get("type") == "piece_data" and data.get("content_hash") == content_hash and data.get("index") == index:
                if data.get("error"):
                    return None
                import base64
                return base64.b64decode(data.get("data"))

    def list_providers(self) -> List[Dict[str, Any]]:
        out = []
        for pid, info in self.providers.items():
            hf = info.get("hf") or {}
            out.append({
                "peer_id": pid,
                "addr": self.peers.get(pid, {}).get("addr"),
                "latency_ms": info.get("latency_ms"),
                "models": hf.get("models", []),
                "price_per_token": hf.get("price_per_token"),
            })
        return out

    def pick_provider(self, model_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        # Choose by price then latency
        providers = []
        for pid, info in self.providers.items():
            hf = info.get("hf") or {}
            if model_name in (hf.get("models") or []):
                providers.append((pid, hf.get("price_per_token", 0.0), info.get("latency_ms") or 1e9))
        if not providers:
            return None
        providers.sort(key=lambda x: (x[1], x[2]))
        best_id = providers[0][0]
        return best_id, self.providers[best_id]

    async def request_generation(self, provider_peer_id: str, prompt: str, max_new_tokens: int = 32, model_name: Optional[str] = None) -> Dict[str, Any]:
        info = self.peers.get(provider_peer_id)
        if not info:
            raise RuntimeError("provider_not_connected")
        ws = info.get("ws")
        rid = new_id("req")
        
        # Create a future to wait for the response
        future = asyncio.Future()
        self._pending_requests[rid] = future
        
        try:
            await self._send(ws, {"type": "gen_request", "rid": rid, "prompt": prompt, "max_new_tokens": int(max_new_tokens), "model": model_name})
            # Wait for the response with a timeout
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            # Clean up the pending request
            self._pending_requests.pop(rid, None)
            raise RuntimeError("generation_timeout")
        except Exception:
            # Clean up the pending request
            self._pending_requests.pop(rid, None)
            raise


async def run_p2p_node(host: str, port: int, bootstrap_link: Optional[str] = None, model_name: Optional[str] = None, price_per_token: Optional[float] = None):
    from .p2p import generate_join_link
    
    node = P2PNode(host, port)
    await node.start()
    
    # Display node information
    console.print(f"\n[bold green]🚀 ConnectIT P2P Node Started[/bold green]")
    console.print(f"[cyan]Node ID:[/cyan] {node.peer_id}")
    console.print(f"[cyan]Address:[/cyan] {node.addr}")
    
    if bootstrap_link:
        console.print(f"\n[yellow]🔗 Connecting to bootstrap...[/yellow]")
        await node.connect_bootstrap(bootstrap_link)
        console.print(f"[green]✓ Bootstrap connection attempted[/green]")
    
    if model_name and price_per_token is not None:
        console.print(f"\n[yellow]🤖 Loading model '{model_name}'...[/yellow]")
        await node.add_hf_service(model_name, float(price_per_token))
        
        # Generate and display join link
        model_hash = sha256_hex_bytes(model_name.encode())
        join_link = generate_join_link("connectit", model_name, model_hash, [node.addr])
        
        console.print(f"[green]✓ Model loaded successfully[/green]")
        console.print(f"[cyan]Model:[/cyan] {model_name}")
        console.print(f"[cyan]Price per token:[/cyan] {price_per_token}")
        console.print(f"\n[bold yellow]📋 Join Link (share this with peers):[/bold yellow]")
        console.print(f"[blue]{join_link}[/blue]")
        console.print(f"\n[bold yellow]🔗 Direct Connection:[/bold yellow]")
        console.print(f"[blue]{node.addr}[/blue]")
    
    console.print(f"\n[bold green]🌐 Node is running and ready to accept connections![/bold green]")
    console.print(f"[dim]Press Ctrl+C to stop the node[/dim]\n")
    
    # Status monitoring loop
    last_peer_count = 0
    while True:
        await asyncio.sleep(5)
        
        # Display peer status updates
        current_peer_count = len(node.peers)
        if current_peer_count != last_peer_count:
            if current_peer_count > last_peer_count:
                console.print(f"[green]📈 Peers connected: {current_peer_count}[/green]")
            else:
                console.print(f"[yellow]📉 Peers connected: {current_peer_count}[/yellow]")
            
            if current_peer_count > 0:
                console.print(f"[dim]Active peers: {', '.join(node.peers.keys())}[/dim]")
            
            last_peer_count = current_peer_count
