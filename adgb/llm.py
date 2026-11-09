"""Small Ollama client built on the standard library."""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Optional


class OllamaUnavailable(RuntimeError):
    """Ollama cannot serve the request; the run stops and can be resumed."""


@dataclass(frozen=True)
class LLMReply:
    text: str
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    seconds: float


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 2048,
        num_predict: int = 256,
        timeout: float = 300.0,
        attempts: int = 3,
        urlopen: Optional[Callable[..., object]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict}
        self.timeout = timeout
        self.attempts = attempts
        # An empty proxy table keeps localhost traffic away from any system proxy.
        self._urlopen = urlopen or urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    def _request(self, path: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        last_error = "unknown error"
        for attempt in range(1, self.attempts + 1):
            try:
                with self._urlopen(request, timeout=self.timeout) as response:  # type: ignore[attr-defined]
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = "HTTP %d" % exc.code
                if exc.code < 500:
                    break
            except urllib.error.URLError as exc:
                if not isinstance(exc.reason, socket.timeout):
                    raise OllamaUnavailable("Ollama is not reachable at %s. Open the Ollama app and run the same command again." % self.base_url)
                last_error = "timed out"
            except socket.timeout:
                last_error = "timed out"
            if attempt < self.attempts:
                time.sleep(2.0 * attempt)
        raise OllamaUnavailable("Ollama request %s failed: %s. Run the same command again to resume." % (path, last_error))

    def version(self) -> str:
        return str(self._request("/api/version").get("version", "unknown"))

    def installed_models(self) -> Dict[str, str]:
        models = self._request("/api/tags").get("models", [])
        return {str(m["name"]): str(m.get("digest", "")) for m in models}  # type: ignore[union-attr]

    def complete(self, model: str, prompt: str) -> LLMReply:
        started = time.monotonic()
        body = self._request(
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": False, "format": "json", "keep_alive": "10m", "options": self.options},
        )
        total = body.get("total_duration")
        return LLMReply(
            text=str(body.get("response", "")),
            prompt_tokens=body.get("prompt_eval_count"),  # type: ignore[arg-type]
            output_tokens=body.get("eval_count"),  # type: ignore[arg-type]
            seconds=float(total) / 1e9 if total else time.monotonic() - started,  # type: ignore[arg-type]
        )
