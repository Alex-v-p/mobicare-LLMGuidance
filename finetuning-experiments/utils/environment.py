
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from adapters.minio import MinioClient


def _run_command(args: list[str], *, cwd: str | None = None) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001
        return None
    output = (completed.stdout or '').strip()
    return output or None


def detect_git_commit(search_from: str | Path | None = None) -> dict[str, Any]:
    cwd = str(Path(search_from or '.').resolve())
    full = _run_command(['git', 'rev-parse', 'HEAD'], cwd=cwd)
    short = _run_command(['git', 'rev-parse', '--short', 'HEAD'], cwd=cwd)
    branch = _run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=cwd)
    dirty = _run_command(['git', 'status', '--porcelain'], cwd=cwd)
    return {
        'commit': full,
        'short_commit': short,
        'branch': branch,
        'dirty': bool(dirty),
    }


def _parse_compose_images(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        services = raw.get('services') or {}
        return {
            str(name): str(service.get('image'))
            for name, service in services.items()
            if isinstance(service, dict) and service.get('image')
        }
    except Exception:  # noqa: BLE001
        images: dict[str, str] = {}
        current_service: str | None = None
        in_services = False
        for line in path.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if stripped == 'services:':
                in_services = True
                continue
            if not in_services or not stripped or stripped.startswith('#'):
                continue
            if line.startswith('  ') and stripped.endswith(':') and not stripped.startswith('image:'):
                current_service = stripped[:-1]
                continue
            if current_service and stripped.startswith('image:'):
                images[current_service] = stripped.split(':', 1)[1].strip().strip("\"'")
        return images


def collect_environment_snapshot(config: Any) -> dict[str, Any]:
    execution = getattr(config, 'execution', None)
    env_cfg = getattr(execution, 'environment', None)
    capture_enabled = bool(getattr(env_cfg, 'capture_enabled', True))
    if not capture_enabled:
        return {
            'capture_enabled': False,
            'models': {
                'llm_model': getattr(getattr(config, 'inference', None), 'llm_model', None),
                'inference_embedding_model': getattr(getattr(config, 'inference', None), 'embedding_model', None),
                'ingestion_embedding_model': getattr(getattr(config, 'ingestion', None), 'embedding_model', None),
            },
        }

    docker_compose_path = Path(getattr(env_cfg, 'docker_compose_path', 'docker-compose.yml'))
    if not docker_compose_path.is_absolute():
        docker_compose_path = Path.cwd() / docker_compose_path

    snapshot: dict[str, Any] = {
        'capture_enabled': True,
        'git': detect_git_commit(docker_compose_path.parent),
        'models': {
            'llm_model': getattr(getattr(config, 'inference', None), 'llm_model', None),
            'inference_embedding_model': getattr(getattr(config, 'inference', None), 'embedding_model', None),
            'ingestion_embedding_model': getattr(getattr(config, 'ingestion', None), 'embedding_model', None),
            'prompt_engineering_label': getattr(getattr(config, 'inference', None), 'prompt_engineering_label', None),
            'pipeline_variant': getattr(getattr(config, 'inference', None), 'pipeline_variant', None),
        },
        'runtime': {
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor() or None,
        },
        'hardware_note': getattr(env_cfg, 'hardware_note', ''),
        'container_versions': _parse_compose_images(docker_compose_path),
    }

    if getattr(env_cfg, 'container_names', None):
        selected = set(getattr(env_cfg, 'container_names') or [])
        snapshot['container_versions'] = {
            name: image
            for name, image in (snapshot.get('container_versions') or {}).items()
            if name in selected
        }

    minio_url = getattr(env_cfg, 'minio_url', None) or os.getenv('MINIO_URL')
    if getattr(env_cfg, 'include_minio', False) and minio_url:
        snapshot['minio'] = MinioClient(minio_url).probe(getattr(env_cfg, 'minio_bucket', None))

    return snapshot
