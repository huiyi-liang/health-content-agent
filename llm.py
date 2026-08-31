"""Shared Nebius LLM configuration for all LLM-based workflow nodes."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# All LLM-based nodes import this shared configuration so they use the same
# provider and model settings.
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def create_nebius_llm() -> ChatOpenAI:
    """Create the project's configured Nebius chat model."""

    # llm.py is at the project root, so its neighboring .env is our local
    # credential file. override=False preserves values already set by the shell.
    project_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(project_env, override=False)

    # Only the secret key comes from .env. The chosen model stays visible in
    # code above so every future LLM node uses the same model by default.
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("NEBIUS_API_KEY is missing. Add it to the local .env file.")

    # Nebius implements the OpenAI-compatible chat interface used by ChatOpenAI.
    return ChatOpenAI(
        model=NEBIUS_MODEL,
        api_key=api_key,
        base_url=NEBIUS_BASE_URL,
        temperature=0,
        max_retries=0,
    )
