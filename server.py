#!/usr/bin/env python3
"""Local UI for checking a LiteLLM proxy.

Run:
    python3 server.py

Then open http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
MAX_MODELS = 40
DEFAULT_PORT = 8787


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def clean_token(token: object) -> str:
    if not isinstance(token, str):
        raise ValueError("Token is required")
    value = token.strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if not value:
        raise ValueError("Token is required")
    if len(value) > 2000:
        raise ValueError("Token is too long")
    return value


def normalize_base(endpoint: object) -> str:
    if not isinstance(endpoint, str):
        raise ValueError("Endpoint is required")
    value = endpoint.strip()
    if not value:
        raise ValueError("Endpoint is required")
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Endpoint must be an http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("Put the token in the token field, not in the URL")
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/models", "/models", "/v1/chat/completions", "/chat/completions", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")


def error_message(parsed: object, text: str) -> str:
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            message = err.get("message") or err.get("error")
            if message:
                return str(message)
        if isinstance(err, str) and err.strip():
            return err.strip()
        message = parsed.get("message") or parsed.get("detail")
        if isinstance(message, str) and message.strip():
            return message.strip()
    cleaned = (text or "").strip()
    if cleaned:
        return cleaned[:1500]
    return "The endpoint returned an empty error"


def classify(status: int, message: str) -> str:
    if status == 200:
        return "allowed"
    lowered = (message or "").lower()
    denied_markers = (
        "not allowed",
        "key not allowed",
        "access denied",
        "permission denied",
        "unauthorized",
        "invalid api key",
        "invalid token",
        "authentication",
        "forbidden",
    )
    if status in {401, 403} or any(marker in lowered for marker in denied_markers):
        return "denied"
    return "error"


def parse_models(parsed: object) -> list[dict]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("data"), list):
        raise ValueError("The models endpoint did not return a model list")
    models = []
    seen = set()
    for item in parsed["data"]:
        if isinstance(item, str):
            model_id = item.strip()
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            model_id = item["id"].strip()
        else:
            continue
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append({"id": model_id})
    return models


def request_json(method: str, url: str, token: str | None = None, payload: dict | None = None, timeout: int = 30):
    headers = {
        "Accept": "application/json",
        "User-Agent": "litellm-model-tester/1.0",
    }
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except TimeoutError as exc:
        raise ApiError(504, "The endpoint timed out") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ApiError(502, f"Could not reach the endpoint: {reason}") from exc
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = None
    return status, parsed, text


def list_models(endpoint: object, token: object) -> dict:
    """Return the models visible to this key.

    A master key should see the full catalog. A virtual key should see only
    the models that key is allowed to call.
    """
    base = normalize_base(endpoint)
    secret = clean_token(token)
    token_status, token_parsed, token_text = request_json(
        "GET", f"{base}/v1/models", token=secret, timeout=30
    )
    if token_status != 200:
        status = token_status if token_status in {400, 401, 403, 404} else 502
        if token_status == 404:
            message = "The models endpoint was not found. Use the proxy base URL, for example https://llm.ecda.ai"
        else:
            message = error_message(token_parsed, token_text)
        raise ApiError(status, message)

    models = [{"id": model["id"]} for model in parse_models(token_parsed)]
    truncated = len(models) > MAX_MODELS
    if truncated:
        models = models[:MAX_MODELS]
    attach_model_info(base, secret, models)
    return {
        "endpoint": base,
        "truncated": truncated,
        "models": models,
    }


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def attach_model_info(base: str, token: str, models: list[dict]) -> None:
    parsed = None
    for path in ("/model/info", "/v1/model/info"):
        try:
            status, body, _text = request_json("GET", f"{base}{path}", token=token, timeout=20)
        except ApiError:
            return
        if status == 200 and isinstance(body, dict):
            parsed = body
            break
    if not isinstance(parsed, dict) or not isinstance(parsed.get("data"), list):
        return
    details = {}
    for item in parsed["data"]:
        if not isinstance(item, dict):
            continue
        name = item.get("model_name") or item.get("id")
        if not isinstance(name, str) or not name.strip():
            continue
        info = item.get("model_info") if isinstance(item.get("model_info"), dict) else {}
        params = item.get("litellm_params") if isinstance(item.get("litellm_params"), dict) else {}
        provider = params.get("custom_llm_provider") or info.get("litellm_provider") or ""
        details[name.strip()] = {
            "mode": info.get("mode") if isinstance(info.get("mode"), str) else "",
            "provider": provider if isinstance(provider, str) else "",
            "max_input_tokens": _optional_int(info.get("max_input_tokens")),
            "max_output_tokens": _optional_int(info.get("max_output_tokens") or info.get("max_tokens")),
        }
    for model in models:
        extra = details.get(model["id"])
        if not extra:
            continue
        for key, value in extra.items():
            if value not in (None, ""):
                model[key] = value


def key_info(endpoint: object, master_token: object, key: object) -> dict:
    base = normalize_base(endpoint)
    master = clean_token(master_token)
    target = clean_token(key)
    status, parsed, text = request_json(
        "GET",
        f"{base}/key/info?key={quote(target, safe='')}",
        token=master,
        timeout=20,
    )
    if status != 200 or not isinstance(parsed, dict):
        raise ApiError(status if status in {400, 401, 403, 404} else 502, error_message(parsed, text))
    info = parsed.get("info") if isinstance(parsed.get("info"), dict) else parsed
    if not isinstance(info, dict):
        raise ApiError(502, "The key info endpoint returned an unexpected payload")
    models = info.get("models")
    alias = info.get("key_alias") or info.get("key_name") or ""
    return {
        "spend": info.get("spend") if isinstance(info.get("spend"), (int, float)) and not isinstance(info.get("spend"), bool) else None,
        "max_budget": info.get("max_budget") if isinstance(info.get("max_budget"), (int, float)) and not isinstance(info.get("max_budget"), bool) else None,
        "rpm_limit": _optional_int(info.get("rpm_limit")),
        "tpm_limit": _optional_int(info.get("tpm_limit")),
        "expires": info.get("expires") if isinstance(info.get("expires"), str) else None,
        "alias": alias if isinstance(alias, str) else "",
        "model_count": len(models) if isinstance(models, list) else None,
    }


def is_embedding_model(model: str) -> bool:
    name = model.lower()
    return "embed" in name


def flatten_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(content).strip()


def clip(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def chat_text(parsed: object) -> str | None:
    if not isinstance(parsed, dict):
        return None
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message")
    content = ""
    reasoning = ""
    if isinstance(message, dict):
        content = flatten_content(message.get("content"))
        reasoning = flatten_content(message.get("reasoning_content"))
    if not content:
        content = flatten_content(choice.get("text"))
    if content:
        return content
    if reasoning:
        return reasoning
    return ""


def embedding_summary(parsed: object) -> str | None:
    if not isinstance(parsed, dict):
        return None
    data = parsed.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    vector = data[0].get("embedding")
    if not isinstance(vector, list):
        return None
    return (
        "This model returns embeddings, so the sample prompt was sent to the embeddings endpoint. "
        f"Access works. Vector length: {len(vector)}."
    )


def reported_model(parsed: object) -> str | None:
    if isinstance(parsed, dict) and isinstance(parsed.get("model"), str):
        return parsed["model"]
    return None


def usage_brief(parsed: object) -> dict | None:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("usage"), dict):
        return None
    usage = parsed["usage"]
    brief = {}
    if isinstance(usage.get("prompt_tokens"), int):
        brief["input_tokens"] = usage["prompt_tokens"]
    if isinstance(usage.get("completion_tokens"), int):
        brief["output_tokens"] = usage["completion_tokens"]
    return brief or None


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def result_payload(
    model: str,
    access: str,
    kind: str,
    started: float,
    http_status: int | None = None,
    response: str | None = None,
    error: str | None = None,
    parsed: object = None,
) -> dict:
    return {
        "model": model,
        "access": access,
        "http_status": http_status,
        "latency_ms": elapsed_ms(started),
        "kind": kind,
        "response": clip(response) if isinstance(response, str) else None,
        "error": error,
        "reported_model": reported_model(parsed) if access == "allowed" else None,
        "usage": usage_brief(parsed) if access == "allowed" else None,
    }


def looks_like_embedding_mismatch(message: str) -> bool:
    lowered = message.lower()
    return "embed" in lowered and any(
        phrase in lowered for phrase in ("not support", "does not", "unsupported", "only support", "chat")
    )


def should_switch_token_field(text: str) -> bool:
    lowered = text.lower()
    return "max_tokens" in lowered and any(
        phrase in lowered for phrase in ("max_completion_tokens", "unsupported", "not supported")
    )


def should_drop_temperature(text: str) -> bool:
    lowered = text.lower()
    return "temperature" in lowered and any(
        phrase in lowered for phrase in ("unsupported", "not support", "not supported", "cannot")
    )


def probe_embeddings(base: str, token: str, model: str, prompt: str, started: float) -> dict:
    status, parsed, text = request_json(
        "POST",
        f"{base}/v1/embeddings",
        token=token,
        payload={"model": model, "input": prompt},
        timeout=60,
    )
    message = None if status == 200 else error_message(parsed, text)
    access = classify(status, message or "")
    response = None
    if access == "allowed":
        response = embedding_summary(parsed)
        if response is None:
            access = "error"
            message = "The embeddings endpoint returned an unexpected payload"
    return result_payload(
        model,
        access,
        "embedding",
        started,
        http_status=status,
        response=response,
        error=message,
        parsed=parsed,
    )


def probe_chat(base: str, token: str, model: str, prompt: str, started: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 200,
    }
    status, parsed, text = request_json(
        "POST",
        f"{base}/v1/chat/completions",
        token=token,
        payload=payload,
        timeout=120,
    )
    if status not in {200, 401, 403} and should_switch_token_field(text):
        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = 200
        status, parsed, text = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            token=token,
            payload=payload,
            timeout=120,
        )
    if status not in {200, 401, 403} and should_drop_temperature(text):
        payload.pop("temperature", None)
        status, parsed, text = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            token=token,
            payload=payload,
            timeout=120,
        )
    if status not in {200, 401, 403} and looks_like_embedding_mismatch(error_message(parsed, text)):
        return probe_embeddings(base, token, model, prompt, started)

    message = None if status == 200 else error_message(parsed, text)
    access = classify(status, message or "")
    response = None
    if access == "allowed":
        extracted = chat_text(parsed)
        if extracted is None:
            access = "error"
            message = "The endpoint returned a response that was not a chat completion"
        else:
            response = extracted or "(The model returned an empty message.)"
    return result_payload(
        model,
        access,
        "chat",
        started,
        http_status=status,
        response=response,
        error=message,
        parsed=parsed,
    )


def probe_model(endpoint: object, token: object, model: object, prompt: object) -> dict:
    base = normalize_base(endpoint)
    secret = clean_token(token)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model is required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt is required")
    model_id = model.strip()
    text = prompt.strip()
    if len(model_id) > 300:
        raise ValueError("Model id is too long")
    if len(text) > 8000:
        raise ValueError("Prompt is too long")

    started = time.perf_counter()
    kind = "embedding" if is_embedding_model(model_id) else "chat"
    try:
        if kind == "embedding":
            return probe_embeddings(base, secret, model_id, text, started)
        return probe_chat(base, secret, model_id, text, started)
    except ApiError as exc:
        return result_payload(model_id, "error", kind, started, error=exc.message)


def _stat(values: list[float]) -> dict | None:
    numbers = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not numbers:
        return None
    ordered = sorted(numbers)
    count = len(ordered)
    if count % 2:
        median = ordered[count // 2]
    else:
        median = (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def _round_stat(stat: dict | None, digits: int) -> dict | None:
    if not stat:
        return None
    return {key: round(value, digits) for key, value in stat.items()}


def delta_text(delta: object) -> str:
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        text = flatten_content(content)
        if text:
            return text
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    return ""


def _sample(ok: bool, access: str, started: float, **fields) -> dict:
    sample = {
        "ok": ok,
        "access": access,
        "total_ms": elapsed_ms(started),
        "ttft_ms": None,
        "output_tokens": None,
        "tokens_per_second": None,
        "response": None,
        "error": None,
        "http_status": None,
        "streamed": False,
        "kind": "chat",
        "reported_model": None,
    }
    sample.update(fields)
    output_tokens = sample.get("output_tokens")
    ttft_ms = sample.get("ttft_ms")
    total_ms = sample.get("total_ms")
    if isinstance(output_tokens, int) and isinstance(ttft_ms, int) and isinstance(total_ms, int):
        window_ms = total_ms - ttft_ms
        if window_ms >= 50 and output_tokens > 0:
            sample["tokens_per_second"] = round(output_tokens / (window_ms / 1000), 1)
    return sample


def stream_chat_once(base: str, token: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
        "max_tokens": 200,
    }
    started = time.perf_counter()
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "litellm-model-tester/1.0",
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=120)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed = None
        message = error_message(parsed, text)
        if should_switch_token_field(text):
            payload.pop("max_tokens", None)
            payload["max_completion_tokens"] = 200
            return stream_chat_once_payload(base, token, model, payload, retry=False)
        return _sample(False, classify(exc.code, message), started, error=message, http_status=exc.code)
    except TimeoutError:
        return _sample(False, "error", started, error="The endpoint timed out")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return _sample(False, "error", started, error=f"Could not reach the endpoint: {reason}")

    return read_stream(response, started)


def stream_chat_once_payload(base: str, token: str, model: str, payload: dict, retry: bool = False) -> dict:
    started = time.perf_counter()
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "litellm-model-tester/1.0",
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=120)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed = None
        message = error_message(parsed, text)
        return _sample(False, classify(exc.code, message), started, error=message, http_status=exc.code)
    except TimeoutError:
        return _sample(False, "error", started, error="The endpoint timed out")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return _sample(False, "error", started, error=f"Could not reach the endpoint: {reason}")
    return read_stream(response, started)


def read_stream(response, started: float) -> dict:
    parts: list[str] = []
    ttft_at = None
    usage = None
    reported = None
    try:
      try:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("model"), str):
                reported = chunk["model"]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            piece = delta_text(choices[0].get("delta"))
            if not piece:
                continue
            if ttft_at is None:
                ttft_at = time.perf_counter()
            parts.append(piece)
      except Exception as exc:
          return _sample(False, "error", started, error=f"The stream ended early: {exc}")
    finally:
        response.close()

    output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(output_tokens, int):
        output_tokens = None
    text = "".join(parts).strip()
    ttft_ms = int((ttft_at - started) * 1000) if ttft_at is not None else None
    return _sample(
        True,
        "allowed",
        started,
        ttft_ms=ttft_ms,
        output_tokens=output_tokens,
        response=clip(text) if text else "(The model returned an empty message.)",
        http_status=200,
        streamed=ttft_ms is not None,
        reported_model=reported,
    )


def embedding_once(base: str, token: str, model: str, prompt: str) -> dict:
    started = time.perf_counter()
    try:
        status, parsed, text = request_json(
            "POST",
            f"{base}/v1/embeddings",
            token=token,
            payload={"model": model, "input": prompt},
            timeout=60,
        )
    except ApiError as exc:
        return _sample(False, "error", started, error=exc.message, kind="embedding")
    message = None if status == 200 else error_message(parsed, text)
    access = classify(status, message or "")
    if access != "allowed":
        return _sample(False, access, started, error=message, http_status=status, kind="embedding")
    summary = embedding_summary(parsed)
    if summary is None:
        return _sample(False, "error", started, error="The embeddings endpoint returned an unexpected payload", http_status=status, kind="embedding")
    return _sample(True, "allowed", started, response=summary, http_status=status, kind="embedding", reported_model=reported_model(parsed))


def chat_once(base: str, token: str, model: str, prompt: str) -> dict:
    if is_embedding_model(model):
        return embedding_once(base, token, model, prompt)
    sample = stream_chat_once(base, token, model, prompt)
    if sample["ok"] or sample["access"] == "denied":
        return sample
    message = sample.get("error") or ""
    if "stream" not in message.lower():
        return sample
    started = time.perf_counter()
    try:
        fallback = probe_chat(base, token, model, prompt, started)
    except ApiError as exc:
        return _sample(False, "error", started, error=exc.message)
    return _sample(
        fallback["access"] == "allowed",
        fallback["access"],
        started,
        total_ms=fallback["latency_ms"],
        output_tokens=(fallback.get("usage") or {}).get("output_tokens"),
        response=fallback.get("response"),
        error=fallback.get("error"),
        http_status=fallback.get("http_status"),
        streamed=False,
        reported_model=fallback.get("reported_model"),
    )


def performance_test(endpoint: object, token: object, model: object, prompt: object, runs: object) -> dict:
    base = normalize_base(endpoint)
    secret = clean_token(token)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model is required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt is required")
    try:
        count = int(runs)
    except (TypeError, ValueError) as exc:
        raise ValueError("Runs must be a number from 1 to 5") from exc
    if count < 1 or count > 5:
        raise ValueError("Runs must be a number from 1 to 5")
    model_id = model.strip()
    text = prompt.strip()
    if len(model_id) > 300:
        raise ValueError("Model id is too long")
    if len(text) > 8000:
        raise ValueError("Prompt is too long")

    samples = []
    failure = None
    for _ in range(count):
        sample = chat_once(base, secret, model_id, text)
        samples.append(sample)
        if not sample["ok"]:
            failure = sample
            break

    successes = [sample for sample in samples if sample["ok"]]
    kind = samples[-1]["kind"] if samples else "chat"
    summary = None
    if successes:
        summary = {
            "ttft_ms": _round_stat(_stat([sample["ttft_ms"] for sample in successes if sample["ttft_ms"] is not None]), 0),
            "total_ms": _round_stat(_stat([sample["total_ms"] for sample in successes]), 0),
            "tokens_per_second": _round_stat(
                _stat([sample["tokens_per_second"] for sample in successes if sample["tokens_per_second"] is not None]),
                1,
            ),
            "output_tokens": _round_stat(
                _stat([sample["output_tokens"] for sample in successes if sample["output_tokens"] is not None]),
                0,
            ),
        }
    last_success = successes[-1] if successes else None
    return {
        "model": model_id,
        "access": "allowed" if successes else (failure["access"] if failure else "error"),
        "http_status": None if successes else (failure or {}).get("http_status"),
        "kind": kind,
        "streamed": any(sample["streamed"] for sample in successes),
        "response": last_success["response"] if last_success else None,
        "reported_model": last_success["reported_model"] if last_success else None,
        "error": None if failure is None or successes else failure.get("error"),
        "run_error": failure.get("error") if failure and successes else None,
        "samples": [
            {
                "ttft_ms": sample["ttft_ms"],
                "total_ms": sample["total_ms"],
                "output_tokens": sample["output_tokens"],
                "tokens_per_second": sample["tokens_per_second"],
                "error": sample["error"],
            }
            for sample in samples
        ],
        "summary": summary,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            if not INDEX.is_file():
                self.send_json(500, {"error": "index.html is missing"})
                return
            self.send_bytes(200, "text/html; charset=utf-8", INDEX.read_bytes())
            return
        if path == "/flow.svg":
            chart = ROOT / "flow.svg"
            if not chart.is_file():
                self.send_json(404, {"error": "Not found"})
                return
            self.send_bytes(200, "image/svg+xml", chart.read_bytes())
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            data = self.read_json()
            if path == "/api/models":
                self.send_json(200, list_models(data.get("endpoint", ""), data.get("token", "")))
                return
            if path == "/api/probe":
                self.send_json(
                    200,
                    probe_model(
                        data.get("endpoint", ""),
                        data.get("token", ""),
                        data.get("model", ""),
                        data.get("prompt", ""),
                    ),
                )
                return
            if path == "/api/perf":
                self.send_json(
                    200,
                    performance_test(
                        data.get("endpoint", ""),
                        data.get("token", ""),
                        data.get("model", ""),
                        data.get("prompt", ""),
                        data.get("runs", 3),
                    ),
                )
                return
            if path == "/api/key-info":
                self.send_json(
                    200,
                    key_info(data.get("endpoint", ""), data.get("master_token", ""), data.get("key", "")),
                )
                return
            self.send_json(404, {"error": "Not found"})
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except ApiError as exc:
            status = exc.status if exc.status in {400, 401, 403, 404} else 502
            self.send_json(status, {"error": exc.message})
        except Exception:
            self.send_json(500, {"error": "The local tester hit an unexpected error"})

    def read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Request body must be JSON") from exc
        if length < 0 or length > 1_000_000:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        return data

    def send_json(self, status: int, payload: dict):
        self.send_bytes(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def send_bytes(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, fmt: str, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Local LiteLLM model tester")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"Could not listen on 127.0.0.1:{args.port}: {exc}") from exc
    server.daemon_threads = True
    print(f"LiteLLM model tester running at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
