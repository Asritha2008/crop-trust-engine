import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_log_lock = threading.Lock()
MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


def _append_to_file(log_path: Path, line: str) -> None:
    """Rotate if >= 5 MB and append line to log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size >= MAX_LOG_SIZE_BYTES:
        backup = log_path.with_name(f"{log_path.name}.1")
        if backup.exists():
            backup.unlink()
        log_path.rename(backup)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def log_request(inputs: Dict[str, Any], assessment: Dict[str, Any], latency_ms: float) -> None:
    """Log a single prediction request to a JSONL file with rotation and zero personal data.

    Whitelisted fields only:
      - time: ISO 8601 UTC string
      - verdict: recommend | caution | refuse | null
      - ood_percentile: float | null
      - stability: float | null
      - latency_ms: float
      - model_version: string | null
      - N, P, K, temperature, humidity, ph, rainfall: floats
    """
    try:
        ood = assessment.get("ood") if isinstance(assessment.get("ood"), dict) else {}
        ood_percentile = ood.get("percentile") if ood else None

        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "verdict": assessment.get("verdict"),
            "ood_percentile": ood_percentile,
            "stability": assessment.get("stability"),
            "latency_ms": round(float(latency_ms), 3),
            "model_version": assessment.get("model_version"),
            "N": float(inputs["N"]),
            "P": float(inputs["P"]),
            "K": float(inputs["K"]),
            "temperature": float(inputs["temperature"]),
            "humidity": float(inputs["humidity"]),
            "ph": float(inputs["ph"]),
            "rainfall": float(inputs["rainfall"]),
        }

        line = json.dumps(entry) + "\n"
        log_path = Path(os.getenv("REQUEST_LOG_PATH", "logs/requests.jsonl"))

        with _log_lock:
            _append_to_file(log_path, line)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to write request log: {e}\n")
