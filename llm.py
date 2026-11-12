"""OpenAI-compatible LLM client with an on-disk cache (adapted from E:\\thies\\poc\\pipeline.py).

Every answer is saved under a hash of (model, messages, parameters), so a stopped run resumes without
repeating calls, and a re-run with the same prompts costs nothing. The key is never printed.
"""
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path


class DeadlineReached(Exception):
    pass


class LLM:
    def __init__(self, base_url, model, cache_dir, reasoning_effort="high", json_mode=True,
                 max_tokens=12000, deadline=None, mock=False):
        self.model, self.mock, self.deadline = model, mock, deadline
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.params = {"temperature": 0, "seed": 42, "reasoning_effort": reasoning_effort, "max_tokens": max_tokens}
        if json_mode:
            self.params["response_format"] = {"type": "json_object"}
        self._lock = threading.Lock()
        self.stats = Counter()
        self.log_lines = []
        self.client = None
        if not mock:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
                                 timeout=1800, max_retries=0)

    def _key(self, messages):
        blob = json.dumps({"model": self.model, "messages": messages, "params": self.params},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def call(self, messages, meta):
        """Returns the answer text. meta (role, user, ...) goes to the call log."""
        path = self.cache_dir / f"{self._key(messages)}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            self._log(meta, rec, cached=True)
            return rec["content"]
        if self.deadline and time.time() > self.deadline:
            raise DeadlineReached()
        rec = _mock_answer(messages, meta) if self.mock else self._request(messages)
        rec["model"], rec["params"] = self.model, self.params
        tmp = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(tmp, path)
        except PermissionError:      # two threads wrote the same request at once (Windows); keep the first
            if not path.exists():
                raise
            tmp.unlink(missing_ok=True)
        self._log(meta, rec, cached=False)
        return rec["content"]

    def _request(self, messages):
        import openai
        delay = 5
        for _ in range(12):
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, **self.params)
                usage = r.usage
                details = getattr(usage, "completion_tokens_details", None) if usage else None
                return {"content": r.choices[0].message.content or "",
                        "finish_reason": r.choices[0].finish_reason,
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
                        "seconds": round(time.time() - t0, 2)}
            except openai.BadRequestError as e:
                msg = str(e).lower()
                dropped = [p for p in ("response_format", "reasoning_effort", "seed") if p in msg and p in self.params]
                for p in dropped:
                    print(f"  [warn] backend rejected '{p}', dropping it for the rest of the run", flush=True)
                    self.params.pop(p)
                if not dropped:
                    return {"content": "", "finish_reason": f"error 400: {str(e)[:300]}", "prompt_tokens": 0,
                            "completion_tokens": 0, "reasoning_tokens": 0, "seconds": round(time.time() - t0, 2)}
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                print(f"  [net] {type(e).__name__}, retry in {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
            except openai.APIStatusError as e:
                if e.status_code in (401, 403, 404):
                    raise
                print(f"  [http {e.status_code}] retry in {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("LLM call failed after retries")

    def _log(self, meta, rec, cached):
        with self._lock:
            self.stats["calls"] += 1
            self.stats["cached" if cached else "new"] += 1
            if not cached:
                for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "seconds"):
                    self.stats[k] += rec.get(k, 0) or 0
            self.log_lines.append({**meta, "cached": cached, **{k: rec.get(k) for k in (
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "seconds", "finish_reason")}})


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            return None
    return None


def _mock_answer(messages, meta):
    """Deterministic fake answers for --mock dry runs (no network)."""
    h = int(hashlib.md5(messages[-1]["content"].encode()).hexdigest(), 16)
    if meta["role"] == "extract":
        content = json.dumps({"like": [f"liked thing {h % 7}"], "dislike": [f"disliked thing {h % 5}"] if h % 3 else []})
    elif meta["role"] == "profile":
        content = json.dumps({"like": [f"likes theme {h % 9 + k}" for k in range(5)],
                              "dislike": [f"dislikes theme {h % 4 + k}" for k in range(5)]})
    elif meta["role"] == "item":
        content = json.dumps({"praised": [f"praised theme {h % 9 + k}" for k in range(3)],
                              "criticized": [f"criticized theme {h % 4 + k}" for k in range(2)] if h % 5 else []})
    else:
        content = "{}"
    return {"content": content, "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "seconds": 0.0}
