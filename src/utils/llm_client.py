"""
Single entry point for all LLM calls in Itihas.
No LLM SDK is imported anywhere else in the codebase only here.

Supports two backends:
  hf     — HuggingFace Inference Providers router (default)
  ollama — local Ollama server (Week 3 domain agents)

Usage:
    from src.utils.llm_client import call

    response = call("how was battle of palked fought?")
    response = call("Summarise this.", backend="ollama")
"""

from __future__ import annotations

import time
from typing import Any

import requests

from config.settings import (
    HF_API_TOKEN,
    HF_MODEL,
    HF_API_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_TEMPERATURE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class LLMCallError(Exception):
    """Raised when an LLM call fails after exhausting retries."""
    pass


def _hf_call(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: list[str] | None = None,
) -> str:
    """
    POST to HuggingFace router using the OpenAI-compatible chat completions API.
    prompt is sent as a single user message.
    Returns the assistant reply string only.
    Retries on RateLimitError and 503 (model loading) only. No retry on other 4xx.
    Raises LLMCallError on permanent failure.
    """
    import openai
    from openai import OpenAI

    client = OpenAI(
        base_url=HF_API_BASE_URL,
        api_key=HF_API_TOKEN,
    )

    kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        timeout=60.0,
    )
    if stop_sequences:
        kwargs["stop"] = stop_sequences

    last_exc: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            completion = client.chat.completions.create(**kwargs)

            text = completion.choices[0].message.content
            duration = time.monotonic() - t0
            log.info(
                "HF call success: model=%s max_tokens=%d temp=%.2f duration=%.2fs",
                model, max_tokens, temperature, duration,
            )
            return text.strip() if text else ""

        except openai.RateLimitError as exc:
            wait = LLM_RETRY_DELAY * attempt
            log.warning(
                "HF rate limited: attempt=%d waiting=%ds",
                attempt, wait,
            )
            time.sleep(wait)
            continue

        except openai.AuthenticationError as exc:
            raise LLMCallError(
                "HF API auth failed — check HF_API_TOKEN in .env"
            ) from exc

        except openai.APITimeoutError as exc:
            last_exc = exc
            log.warning("HF timeout: attempt=%d/%d", attempt, LLM_MAX_RETRIES)
            time.sleep(LLM_RETRY_DELAY * attempt)

        except openai.APIStatusError as exc:
            # Retry only on 503 (model loading). All other status errors are permanent.
            if exc.status_code == 503:
                last_exc = exc
                log.warning(
                    "HF model loading (503): attempt=%d/%d",
                    attempt, LLM_MAX_RETRIES,
                )
                time.sleep(LLM_RETRY_DELAY * attempt)
            else:
                raise LLMCallError(
                    f"HF API error: model={model} status={exc.status_code} body={exc.message}"
                ) from exc

        except openai.APIError as exc:
            last_exc = exc
            log.warning(
                "HF API error: attempt=%d/%d error=%s",
                attempt, LLM_MAX_RETRIES, exc,
            )
            time.sleep(LLM_RETRY_DELAY * attempt)

        except Exception as exc:
            last_exc = exc
            log.warning(
                "HF request error: attempt=%d/%d error=%s",
                attempt, LLM_MAX_RETRIES, exc,
            )
            time.sleep(LLM_RETRY_DELAY)

    raise LLMCallError(
        f"HF API failed after {LLM_MAX_RETRIES} attempts: model={model} last_error={last_exc}"
    )


# ---------------------------------------------------------------------------
# Ollama local inference — /api/generate (Week 3 domain agents)
# ---------------------------------------------------------------------------

def _ollama_call(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: list[str] | None = None,
) -> str:
    """
    POST to local Ollama server at OLLAMA_BASE_URL/api/generate.
    Raises LLMCallError if Ollama is not running or returns an error.
    """
    url     = f"{OLLAMA_BASE_URL}/api/generate"
    payload: dict = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    if stop_sequences:
        payload["options"]["stop"] = stop_sequences

    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=120)

        if resp.status_code != 200:
            raise LLMCallError(
                f"Ollama error: model={model} status={resp.status_code} body={resp.text[:200]}"
            )

        text     = resp.json().get("response", "").strip()
        duration = time.monotonic() - t0
        log.info(
            "Ollama call success: model=%s max_tokens=%d duration=%.2fs",
            model, max_tokens, duration,
        )
        return text

    except LLMCallError:
        raise

    except requests.exceptions.ConnectionError as exc:
        raise LLMCallError(
            f"Ollama not reachable at {OLLAMA_BASE_URL} — is 'ollama serve' running?"
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise LLMCallError(
            f"Ollama timed out after 120s: model={model}"
        ) from exc


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def call(
    prompt: str,
    *,
    backend: str = "hf",
    model: str | None = None,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    stop_sequences: list[str] | None = None,
) -> str:
    """
    Make a single LLM completion call. Only function the codebase should call.

    Args:
        prompt:         Full prompt string. Caller formats it.
        backend:        "hf" (HuggingFace router) or "ollama" (local). Default "hf".
        model:          Override model for this call only.
                        Defaults to HF_MODEL for hf, OLLAMA_MODEL for ollama.
        max_tokens:     Maximum tokens to generate.
        temperature:    Use 0.01 for classification tasks (router rejects exactly 0.0).
        stop_sequences: Optional list of strings that stop generation when encountered.

    Returns:
        Generated text string, stripped of whitespace.

    Raises:
        LLMCallError: API failure after retries, or permanent error (auth, bad status).
        ValueError:   Invalid backend.
    """
    if backend == "hf":
        resolved = model or HF_MODEL
        log.info(
            "LLM call: backend=hf model=%s max_tokens=%d temp=%.2f",
            resolved, max_tokens, temperature,
        )
        return _hf_call(prompt, resolved, max_tokens, temperature, stop_sequences)

    elif backend == "ollama":
        resolved = model or OLLAMA_MODEL
        log.info(
            "LLM call: backend=ollama model=%s max_tokens=%d temp=%.2f",
            resolved, max_tokens, temperature,
        )
        return _ollama_call(prompt, resolved, max_tokens, temperature, stop_sequences)

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'hf' or 'ollama'.")


# Convenience wrappers

def call_hf(
    prompt: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    model: str | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    return call(prompt, backend="hf", model=model,
                max_tokens=max_tokens, temperature=temperature,
                stop_sequences=stop_sequences)


def call_ollama(
    prompt: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    model: str | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    """Used by domain agents in Week 3. Requires 'ollama serve' running locally."""
    return call(prompt, backend="ollama", model=model,
                max_tokens=max_tokens, temperature=temperature,
                stop_sequences=stop_sequences)


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_hf_connection() -> bool:
    if not HF_API_TOKEN:
        log.error("HF_API_TOKEN not set in .env")
        return False
    try:
        result = call_hf("Reply with the single word: ready", max_tokens=5)
        log.info("HF connection OK: response=%r", result[:40])
        return True
    except Exception as exc:
        log.error("HF connection failed: %s", exc)
        return False


def check_ollama_connection() -> bool:
    try:
        result = call_ollama("Reply with the single word: ready", max_tokens=5)
        log.info("Ollama connection OK: response=%r", result[:40])
        return True
    except Exception as exc:
        log.error("Ollama connection failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM client")
    parser.add_argument(
        "--backend", choices=["hf", "ollama", "both"], default="hf",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="In one word, what was the INA? Answer:",
    )
    args = parser.parse_args()

    if args.backend in ("hf", "both"):
        print("Testing HF API...")
        if check_hf_connection():
            print(f"HF response: {call_hf(args.prompt, max_tokens=20)!r}")
        else:
            print("HF API unavailable — check HF_API_TOKEN in .env")

    if args.backend in ("ollama", "both"):
        print("Testing Ollama...")
        if check_ollama_connection():
            print(f"Ollama response: {call_ollama(args.prompt, max_tokens=20)!r}")
        else:
            print("Ollama unavailable — run 'ollama serve' first")