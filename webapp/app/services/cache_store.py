import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


class JsonCacheStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._data = self._load()

    def _load(self) -> dict:
        if not self.file_path.exists():
            return {}
        try:
            return json.loads(self.file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def get(self, key: str, ttl_hours: int):
        entry = self._data.get(key)
        if not entry:
            return None
        timestamp = entry.get("timestamp")
        if not timestamp:
            return None
        try:
            created_at = datetime.fromisoformat(timestamp)
        except ValueError:
            return None
        if datetime.now(UTC) - created_at > timedelta(hours=ttl_hours):
            return None
        return entry.get("value")

    def set(self, key: str, value: dict) -> None:
        self._data[key] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "value": value,
        }

    def save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
