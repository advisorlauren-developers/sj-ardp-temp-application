"""Secrets come from the environment, never from config.

A credential in configs/ would be hashed into every run id, copied verbatim
into runs/*/config.resolved.yaml, and rendered on a web page that prints the
config hash. So the split is absolute: config holds the coordinates of an
experiment, the environment holds the key that pays for it.

Nothing here logs a value. The only thing a caller may learn is whether a name
is set.
"""
from __future__ import annotations

import os
from pathlib import Path

# botocore reads this exact name for Bedrock bearer auth; BEDROCK_API_KEY is
# what the console calls the same string when it hands it to you.
_ALIASES = {"BEDROCK_API_KEY": "AWS_BEARER_TOKEN_BEDROCK"}


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Populate os.environ from a .env file. Returns the NAMES that were set.

    Existing environment variables win: a shell export is a deliberate act and
    must not be silently overridden by a file.
    """
    p = Path(path)
    if not p.exists():
        return []

    names: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        for name in (key, *([_ALIASES[key]] if key in _ALIASES else [])):
            if not os.environ.get(name):
                os.environ[name] = value
                names.append(name)
    return names
