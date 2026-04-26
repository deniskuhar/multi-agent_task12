from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler


from functools import lru_cache
from langfuse import Langfuse
from config import get_settings

settings = get_settings()

@lru_cache(maxsize=1)
def get_langfuse_client():
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )


def auth_check() -> bool:
    try:
        return bool(get_langfuse_client().auth_check())
    except Exception:
        return False


def get_prompt_text(name: str, *, label: str | None = None, **variables: Any) -> str:
    client = get_langfuse_client()
    prompt = client.get_prompt(
        name,
        label=label or settings.langfuse_prompt_label,
        cache_ttl_seconds=settings.langfuse_prompt_cache_ttl_seconds,
    )

    compiled = prompt.compile(**variables) if variables else prompt.compile()

    if isinstance(compiled, str):
        return compiled

    if isinstance(compiled, list):
        parts: list[str] = []
        for item in compiled:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role == "system" and content:
                    parts.append(str(content))
                elif content:
                    parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if hasattr(compiled, "prompt"):
        return str(compiled.prompt)

    return str(compiled)


def make_trace_context(
    *,
    session_id: str | None,
    user_id: str | None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    if session_id:
        ctx["session_id"] = session_id
    if user_id:
        ctx["user_id"] = user_id
    ctx["tags"] = list(tags or settings.langfuse_tags)
    if metadata:
        ctx["metadata"] = metadata
    return ctx


def get_callback_handler(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    force_trace_context: bool = False,
) -> CallbackHandler:
    if force_trace_context:
        return CallbackHandler(trace_context=make_trace_context(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
        ))
    return CallbackHandler()


def trace_attributes(
    *,
    session_id: str | None,
    user_id: str | None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    attrs = make_trace_context(session_id=session_id, user_id=user_id, tags=tags, metadata=metadata)
    try:
        return propagate_attributes(**attrs)
    except Exception:
        return nullcontext()


def flush_langfuse() -> None:
    try:
        get_langfuse_client().flush()
    except Exception:
        pass
