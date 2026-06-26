from __future__ import annotations
import os
import json
import datetime
from typing import Any, Dict, List, Optional

class QALogger:
    """A structured JSONL/in-memory logger for tracking QA events and executions."""
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.events: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }
        self.events.append(event)
        
        if self.log_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception:
                pass
