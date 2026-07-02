"""
Single entry point for all LLM calls in Itihas.
No LLM SDK is imported anywhere else in the codebase.

Supports three backends:
  hf     — HuggingFace Inference Providers router (OpenAI-compatible API)
  openai — OpenAI API (same SDK, different base_url and credentials)
  ollama — local Ollama server (different protocol)

Both "hf" and "openai" use the openai SDK's chat.completions.create().
They share one private function (_openai_sdk_call) — only the client
initialisation differs. No duplicated retry logic.

Usage:
    from src.utils.llm_client import call

    response = call("Classify this document.")                  # OpenAI default
    response = call("Summarise this.", backend="openai")        # OpenAI
    response = call("Summarise this.", backend="ollama")        # local
"""

from __future__ import annotations
import time
import requests
from config.settings import (
    HF_API_TOKEN,
    HF_MODEL,
    HF_API_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
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


# ---------------------------------------------------------------------------
# Shared OpenAI-SDK call — used by both "hf" and "openai" backends
# ---------------------------------------------------------------------------

def _openai_sdk_call(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: list[str] | None,
    api_key: str,
    base_url: str | None,       # None → standard OpenAI endpoint
    vendor: str,                # "hf" or "openai" — used only in log/error messages
    min_temperature: float,     # HF rejects 0.0 exactly; OpenAI accepts it
    retryable_status_codes: tuple[int, ...],  # e.g. (503,) for HF, (500, 503) for OpenAI
) -> str:
    """
    Shared implementation for any backend that speaks the OpenAI chat completions API.
    Retries on RateLimitError, APITimeoutError, and specified status codes.
    Raises LLMCallError on permanent failures (auth, bad request, exhausted retries).
    """
    import openai
    from openai import OpenAI

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=max(temperature, min_temperature),
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
                "%s call success: model=%s max_tokens=%d temp=%.2f duration=%.2fs",
                vendor, model, max_tokens, temperature, duration,
            )
            return text.strip() if text else ""

        except openai.RateLimitError as exc:
            wait = LLM_RETRY_DELAY * attempt
            log.warning("%s rate limited: attempt=%d waiting=%ds", vendor, attempt, wait)
            time.sleep(wait)
            continue

        except openai.AuthenticationError as exc:
            raise LLMCallError(
                f"{vendor} auth failed — check credentials in .env"
            ) from exc

        except openai.BadRequestError as exc:
            # Permanent — bad prompt, context too long, content filter
            raise LLMCallError(
                f"{vendor} bad request: model={model} body={exc.message}"
            ) from exc

        except openai.APITimeoutError as exc:
            last_exc = exc
            log.warning("%s timeout: attempt=%d/%d", vendor, attempt, LLM_MAX_RETRIES)
            time.sleep(LLM_RETRY_DELAY * attempt)

        except openai.APIStatusError as exc:
            if exc.status_code in retryable_status_codes:
                last_exc = exc
                log.warning(
                    "%s server error (%d): attempt=%d/%d",
                    vendor, exc.status_code, attempt, LLM_MAX_RETRIES,
                )
                time.sleep(LLM_RETRY_DELAY * attempt)
            else:
                raise LLMCallError(
                    f"{vendor} API error: model={model} status={exc.status_code} body={exc.message}"
                ) from exc

        except openai.APIError as exc:
            last_exc = exc
            log.warning("%s API error: attempt=%d/%d error=%s", vendor, attempt, LLM_MAX_RETRIES, exc)
            time.sleep(LLM_RETRY_DELAY * attempt)

        except Exception as exc:
            last_exc = exc
            log.warning("%s request error: attempt=%d/%d error=%s", vendor, attempt, LLM_MAX_RETRIES, exc)
            time.sleep(LLM_RETRY_DELAY)

    raise LLMCallError(
        f"{vendor} API failed after {LLM_MAX_RETRIES} attempts: model={model} last_error={last_exc}"
    )


# ---------------------------------------------------------------------------
# Ollama local inference — /api/generate (different protocol, not OpenAI SDK)
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
    url = f"{OLLAMA_BASE_URL}/api/generate"
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

        text = resp.json().get("response", "").strip()
        duration = time.monotonic() - t0
        log.info("Ollama call success: model=%s max_tokens=%d duration=%.2fs", model, max_tokens, duration)
        return text

    except LLMCallError:
        raise

    except requests.exceptions.ConnectionError as exc:
        raise LLMCallError(
            f"Ollama not reachable at {OLLAMA_BASE_URL} — is 'ollama serve' running?"
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise LLMCallError(f"Ollama timed out after 120s: model={model}") from exc


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def call(
    prompt: str,
    *,
    backend: str = "openai",
    model: str | None = None,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    stop_sequences: list[str] | None = None,
) -> str:
    """
    Make a single LLM completion call. Only function the codebase should call.

    Args:
        prompt:         Full prompt string. Caller formats it.
        backend:        "hf" | "openai" | "ollama". Default "openai".
        model:          Override model for this call only.
                        Defaults to HF_MODEL / OPENAI_MODEL / OLLAMA_MODEL.
        max_tokens:     Maximum tokens to generate.
        temperature:    Sampling temperature. HF rejects exactly 0.0 — use 0.01.
        stop_sequences: Optional list of strings that stop generation.

    Returns:
        Generated text string, stripped of whitespace.

    Raises:
        LLMCallError: API failure after retries, or permanent error.
        ValueError:   Unknown backend.
    """
    if backend == "hf":
        resolved = model or HF_MODEL
        log.info("LLM call: backend=hf model=%s max_tokens=%d temp=%.2f", resolved, max_tokens, temperature)
        return _openai_sdk_call(
            prompt, resolved, max_tokens, temperature, stop_sequences,
            api_key=HF_API_TOKEN,
            base_url=HF_API_BASE_URL,
            vendor="hf",
            min_temperature=0.01,       # HF rejects exactly 0.0
            retryable_status_codes=(503,),
        )

    elif backend == "openai":
        resolved = model or OPENAI_MODEL
        log.info("LLM call: backend=openai model=%s max_tokens=%d temp=%.2f", resolved, max_tokens, temperature)
        return _openai_sdk_call(
            prompt, resolved, max_tokens, temperature, stop_sequences,
            api_key=OPENAI_API_KEY,
            base_url=None,              # use OpenAI's default endpoint
            vendor="openai",
            min_temperature=0.0,        # OpenAI accepts exactly 0.0
            retryable_status_codes=(500, 503),
        )

    elif backend == "ollama":
        resolved = model or OLLAMA_MODEL
        log.info("LLM call: backend=ollama model=%s max_tokens=%d temp=%.2f", resolved, max_tokens, temperature)
        return _ollama_call(prompt, resolved, max_tokens, temperature, stop_sequences)

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'hf', 'openai', or 'ollama'.")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

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


def call_openai(
    prompt: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    model: str | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    """Call OpenAI API. Requires OPENAI_API_KEY in .env."""
    return call(prompt, backend="openai", model=model,
                max_tokens=max_tokens, temperature=temperature,
                stop_sequences=stop_sequences)


def call_ollama(
    prompt: str,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    model: str | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    """Call local Ollama instance. Requires 'ollama serve' running."""
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


def check_openai_connection() -> bool:
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set in .env")
        return False
    try:
        result = call_openai("Reply with the single word: ready", max_tokens=5)
        log.info("OpenAI connection OK: response=%r", result[:40])
        return True
    except Exception as exc:
        log.error("OpenAI connection failed: %s", exc)
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
        "--backend", choices=["hf", "openai", "ollama", "all"], default="openai",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="In one word, what was the INA? Answer:",
    )
    args = parser.parse_args()

    if args.backend in ("hf", "all"):
        print("Testing HF API...")
        if check_hf_connection():
            print(f"HF response: {call_hf(args.prompt, max_tokens=20)!r}")
        else:
            print("HF unavailable — check HF_API_TOKEN in .env")

    if args.backend in ("openai", "all"):
        print("Testing OpenAI API...")
        if check_openai_connection():
            print(f"OpenAI response: {call_openai(args.prompt, max_tokens=20)!r}")
        else:
            print("OpenAI unavailable — check OPENAI_API_KEY in .env")

    if args.backend in ("ollama", "all"):
        print("Testing Ollama...")
        if check_ollama_connection():
            print(f"Ollama response: {call_ollama(args.prompt, max_tokens=20)!r}")
        else:
            print("Ollama unavailable — run 'ollama serve' first")
