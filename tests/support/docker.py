from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.request import urlopen

import pytest


@dataclass
class DockerContainer:
    name: str

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], check=False, capture_output=True, text=True)


def docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "version", "--format", "{{json .}}"], check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return False
    return result.returncode == 0


def reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def require_docker() -> None:
    if not docker_available():
        pytest.skip("Docker is not available; skipping real-container integration test.")


def run_container(*, image: str, ports: dict[int, int], env: dict[str, str] | None = None, command: list[str] | None = None) -> DockerContainer:
    require_docker()
    name = f"llmguidance-test-{uuid.uuid4().hex[:10]}"
    cmd = ["docker", "run", "-d", "--name", name]
    for host_port, container_port in ports.items():
        cmd.extend(["-p", f"127.0.0.1:{host_port}:{container_port}"])
    for key, value in (env or {}).items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.append(image)
    if command:
        cmd.extend(command)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container {image}: {result.stderr or result.stdout}")
    return DockerContainer(name=name)


def wait_for_tcp(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for TCP {host}:{port}")


def wait_for_http_json(url: str, timeout_s: float = 30.0) -> dict | list | str | None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:  # noqa: S310 - test helper only
                body = response.read().decode("utf-8")
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return body
        except Exception as exc:  # pragma: no cover - best effort polling helper
            last_error = exc
            time.sleep(0.3)
    raise TimeoutError(f"Timed out waiting for HTTP readiness on {url}: {last_error}")


@contextlib.contextmanager
def managed_container(*, image: str, ports: dict[int, int], env: dict[str, str] | None = None, command: list[str] | None = None) -> Iterator[DockerContainer]:
    container = run_container(image=image, ports=ports, env=env, command=command)
    try:
        yield container
    finally:
        container.stop()
