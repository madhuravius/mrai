"""mrai — read-only dev agent that logs terse answers to .mrai/*.md."""

import os

# Silence pydantic-ai's first-run observability banner on stderr.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
