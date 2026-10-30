"""Configuration, paths, persistent JSONL and API cost accounting."""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
_LEDGER_LOCKS = defaultdict(threading.RLock)


def config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path(__file__).with_name("config.yaml")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    out = Path(cfg["data_root"])
    cfg["output"] = out if out.is_absolute() else ROOT / out
    cfg["config_path"] = str(p.resolve())
    return cfg


def source(cfg: dict, name: str) -> Path:
    return Path(cfg["sources"][name])


def output(cfg: dict, *parts: str) -> Path:
    p = cfg["output"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_benchmark(cfg: dict) -> Path:
    manifest_path = output(cfg, "benchmark", "v1", "benchmark_manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError("frozen benchmark missing; run freeze first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"queries.jsonl", "qrels.tsv", "answers.jsonl", "evidence.jsonl", "accepted_ranges.tsv", "splits.json"}
    if not required.issubset(manifest.get("hashes", {})):
        raise RuntimeError("benchmark manifest does not hash all required files")
    for name, digest in manifest["hashes"].items():
        path = manifest_path.parent / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen benchmark hash mismatch: {name}")
    return manifest_path.parent


def api_key() -> str:
    key = configured_api_key()
    if not key:
        raise RuntimeError("OpenRouter key missing: set LLM_API_KEY or OPENROUTER_API_KEY in the process environment or repo .env")
    return key


def configured_api_key() -> str | None:
    """Read process environment first, then the repository's existing .env convention."""
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.is_file():
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
        return values.get("LLM_API_KEY") or values.get("OPENROUTER_API_KEY")
    return None


def ssl_verify() -> bool | str | ssl.SSLContext:
    bundle = os.getenv("LLM_CA_BUNDLE")
    if bundle:
        return bundle
    if os.getenv("LLM_SSL_VERIFY", "true").lower() == "false":
        return False
    if os.name == "nt":
        # Honor Windows' trusted enterprise roots without disabling TLS validation.
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return True


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    """Uses returned token counts and configured Qwen3.8 Flash rates, in USD."""

    def __init__(self, cfg: dict):
        self.path = output(cfg, "derived", "api_usage.jsonl")
        self.limit = float(cfg["max_usd"])
        self.lock = _LEDGER_LOCKS[str(self.path)]
        self.total = sum(float(r.get("estimated_usd", 0)) for r in read_jsonl(self.path))

    def check(self) -> None:
        with self.lock:
            self.total = sum(float(r.get("estimated_usd", 0)) for r in read_jsonl(self.path))
            if self.total + 0.10 > self.limit:
                raise BudgetExceeded(f"Estimated API cost ${self.total:.4f} leaves less than $0.10 under ${self.limit:.2f} cap")

    def record(self, stage: str, usage: dict, model: str) -> None:
        inp = int(usage.get("prompt_tokens", 0))
        out = int(usage.get("completion_tokens", 0))
        # OpenRouter may change prices; the manifest identifies this as an estimate.
        usd = inp * 0.15 / 1_000_000 + out * 0.47 / 1_000_000
        if inp == 0 and out == 0:
            usd = 0.02
        with self.lock:
            self.total = sum(float(r.get("estimated_usd", 0)) for r in read_jsonl(self.path)) + usd
            append_jsonl(self.path, {"stage": stage, "model": model, "prompt_tokens": inp,
                                    "completion_tokens": out, "estimated_usd": usd,
                                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                    "usage_source": "token_usage" if inp or out else "missing_usage_allowance",
                                    "cumulative_estimated_usd": self.total})

    def reserve_estimate(self, stage: str, usd: float, model: str) -> None:
        """Conservative allowance for library calls that do not expose token usage."""
        with self.lock:
            self.total = sum(float(r.get("estimated_usd", 0)) for r in read_jsonl(self.path))
            if self.total + usd > self.limit:
                raise BudgetExceeded(f"Estimated API cost would exceed ${self.limit:.2f} cap")
            self.total += usd
            append_jsonl(self.path, {"stage": stage, "model": model, "estimated_usd": usd,
                                    "usage_source": "fixed_allowance_no_token_usage",
                                    "cumulative_estimated_usd": self.total})
