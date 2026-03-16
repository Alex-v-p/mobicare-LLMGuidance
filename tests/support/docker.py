from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager

import httpx
import redis.asyncio as redis


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for this test")


def reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


@contextmanager
def managed_container(
    *,
    image: str,
    ports: dict[int, int],
    env: dict[str, str] | None = None,
    command: list[str] | None = None,
):
    require_docker()

    name = f"pytest-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-d", "--name", name]

    for host_port, container_port in ports.items():
        cmd.extend(["-p", f"{host_port}:{container_port}"])

    if env:
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])

    cmd.append(image)

    if command:
        cmd.extend(command)

    subprocess.check_output(cmd, text=True).strip()

    try:
        yield name
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


async def wait_for_redis(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        client = redis.from_url(f"redis://{host}:{port}/0", decode_responses=True)
        try:
            if await client.ping():
                await client.aclose()
                return
        except Exception as exc:
            last_error = exc
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

        await asyncio.sleep(0.5)

    raise RuntimeError(f"Redis container did not become ready in time: {last_error}")


async def wait_for_http_ok(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5)

    raise RuntimeError(f"HTTP endpoint did not become ready in time: {url} ({last_error})")