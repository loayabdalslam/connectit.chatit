from __future__ import annotations
from typing import Optional, Dict, Any
from rich.console import Console
import typer

from .utils import data_file, load_json, save_json, gen_salt, hash_password


console = Console()

USERS_PATH = data_file("users.json")
SESSION_PATH = data_file("session.json")


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


def get_session_user() -> Optional[str]:
    data = load_json(SESSION_PATH, {})
    return data.get("username")


def set_session_user(username: str) -> None:
    save_json(SESSION_PATH, {"username": username})


def clear_session() -> None:
    save_json(SESSION_PATH, {})


def prompt_login() -> str:
    ensure_users_store()
    console.print("[bold cyan]ConnectIT Login[/bold cyan]")
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
                set_session_user(username)
                return username
            else:
                console.print("[red]Invalid credentials[/red]")


def require_login() -> str:
    user = get_session_user()
    if user:
        return user
    return prompt_login()

