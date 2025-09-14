from __future__ import annotations
from typing import List, Tuple, Dict, Any
import os
from .p2p import sha256_hex_bytes


def split_pieces(data: bytes, piece_size: int) -> List[bytes]:
    return [data[i:i+piece_size] for i in range(0, len(data), piece_size)]


def piece_hashes(pieces: List[bytes]) -> List[str]:
    return [sha256_hex_bytes(p) for p in pieces]


def verify_and_reassemble(pieces: List[bytes], hashes: List[str]) -> bytes:
    if len(pieces) != len(hashes):
        raise ValueError("length_mismatch")
    for i, p in enumerate(pieces):
        if sha256_hex_bytes(p) != hashes[i]:
            raise ValueError(f"hash_mismatch_at_{i}")
    return b"".join(pieces)


def save_pieces(folder: str, content_hash: str, pieces: List[bytes]) -> List[str]:
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, p in enumerate(pieces):
        path = os.path.join(folder, f"{content_hash}_{i:08d}.part")
        with open(path, "wb") as f:
            f.write(p)
        paths.append(path)
    return paths


