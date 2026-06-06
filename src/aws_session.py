"""
src/aws_session.py
------------------
Loads AWS credentials from the project .env file and returns a boto3 S3 client.
Import this instead of calling boto3.client("s3") directly.

Usage:
    from src.aws_session import get_s3_client
    s3 = get_s3_client()
"""

import os
from pathlib import Path
import boto3


def _load_env(env_path: Path | None = None) -> None:
    """
    Manually parse a .env file and inject into os.environ.
    Works without python-dotenv installed, but uses it if available.
    """
    # Try python-dotenv first (already in most ML venvs)
    try:
        from dotenv import load_dotenv
        path = env_path or _find_env()
        load_dotenv(path, override=False)   # override=False: real env vars win
        return
    except ImportError:
        pass

    # Fallback: manual parse
    path = env_path or _find_env()
    if path is None or not path.exists():
        return

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)   # don't override real env vars


def _find_env() -> Path | None:
    """Walk up from this file's location until we find a .env."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        p = candidate / ".env"
        if p.exists():
            return p
    return None


def get_s3_client(region: str | None = None):
    """
    Returns a boto3 S3 client using credentials from:
      1. Real environment variables (CI / EC2 instance role / etc.)
      2. .env file in the project root
      3. ~/.aws/credentials  (boto3 default chain)
    """
    _load_env()

    kwargs = {}
    if region or os.getenv("AWS_DEFAULT_REGION"):
        kwargs["region_name"] = region or os.getenv("AWS_DEFAULT_REGION")

    # Explicit credentials (only if present — don't pass None to boto3)
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key  = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_tok = os.getenv("AWS_SESSION_TOKEN")

    if access_key and secret_key:
        kwargs["aws_access_key_id"]     = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_tok:
            kwargs["aws_session_token"] = session_tok

    return boto3.client("s3", **kwargs)