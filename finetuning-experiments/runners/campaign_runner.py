from __future__ import annotations

from pathlib import Path

from campaigns.executor import execute_campaign
from campaigns.generator import load_campaign_config



def run_campaign(config_path: str | Path) -> Path:
    config = load_campaign_config(config_path)
    return execute_campaign(config)
