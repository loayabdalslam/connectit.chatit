import json
import os
import platform
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict


def connectit_home() -> Path:
    base = os.environ.get("CONNECTIT_HOME")
    if base:
        p = Path(base)
    else:
        p = Path.home() / ".connectit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_file(name: str) -> Path:
    p = connectit_home() / name
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def now_ms() -> int:
    return int(time.time() * 1000)


def os_name() -> str:
    return platform.system()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_password(password: str, salt: str) -> str:
    return sha256_hex(password + ":" + salt)


def gen_salt() -> str:
    return uuid.uuid4().hex


