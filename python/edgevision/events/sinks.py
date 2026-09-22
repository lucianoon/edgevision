"""Where events go: a JSONL file, stdout, an HTTP webhook, or several at once.

Sinks are deliberately dependency-free (urllib for the webhook) so the events layer installs
with the package; MQTT/Kafka adapters can wrap the same `Sink` protocol later.
"""

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Protocol

from edgevision.events.engine import Event


class Sink(Protocol):
    def emit(self, event: Event) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """One JSON object per line; `path` may be "-" for stdout."""

    def __init__(self, path: str | Path):
        self._own = str(path) != "-"
        self._file: IO[str] = open(path, "w", encoding="utf-8") if self._own else sys.stdout

    def emit(self, event: Event) -> None:
        self._file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.flush()
        if self._own:
            self._file.close()


class WebhookSink:
    """POSTs each event as JSON. Failures are counted, not raised: a slow webhook must never stall
    the pipeline that feeds it."""

    def __init__(self, url: str, timeout_s: float = 2.0):
        self.url = url
        self.timeout_s = timeout_s
        self.sent = 0
        self.failed = 0

    def emit(self, event: Event) -> None:
        body = json.dumps(event.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response.read()
            self.sent += 1
        except (urllib.error.URLError, TimeoutError, OSError):
            self.failed += 1

    def close(self) -> None:
        return None


class MultiSink:
    def __init__(self, sinks: Iterable[Sink]):
        self.sinks = list(sinks)

    def emit(self, event: Event) -> None:
        for sink in self.sinks:
            sink.emit(event)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
