# mrai

Read-only dev agent. Answers a prompt against the current repo, prints with `rich`,
and logs to `.mrai/*.md` (topic-keyed). It has no repo write access; the only
write in the system is the CLI appending your answer to the log.

## Install

```bash
uv sync
cp .env.example .env     # then set OPENROUTER_API_KEY inside .env
```

## Run

```bash
uv run mrai "where is slugify defined?"       # auto-routed (Jev)
uv run mrai "..." --flash                     # force the flash-tier model (fast/cheap)
uv run mrai "..." --pro                       # force the pro-tier model (hard tasks)
uv run mrai "..." --model openai/gpt-4o       # any OpenRouter model id
uv run python -m mrai "..."                   # module form
```

`--flash` / `--pro` / `--model` bypass routing. Env equivalent: `MRAI_MODEL` (CLI wins).

## Routing (default)

The default asks Jev (`typesafe/jev-1.14`), a cheap decision model:

| Jev signal | Action |
| --- | --- |
| simple, conf ≥ 0.6 | flash-tier model |
| complex / shaky simple | pro-tier model |
| multi-loop prob ≥ 0.65 | enable `explorer` subagent |
| unreachable | fail-safe: Pro, no subagents |

Tier names are labels, not model IDs: the actual flash/pro models are the
`MRAI_FLASH_MODEL` / `MRAI_PRO_MODEL` settings defaults, overridable via
`--model`, `MRAI_MODEL`, or those env vars.

## Common flags

| Flag | Effect |
| --- | --- |
| `-f, --flash` | force Flash |
| `-p, --pro` | force Pro |
| `-m, --model <id>` | force a model |
| `-s, --subagents` | force explorer subagent on |
| `--no-subagents` | force explorer subagent off |
| `--no-memory` | don't inject long-term memory (logbook index + `.mrai/memories.md`) |
| `--no-write` | print answer only; skip the logbook and don't persist the session roll |
| `--no-web` | disable `web_fetch` |
| `-d, --deny <glob>` | add a read-deny pattern (repeatable) |
| `--reindex` | rebuild the HTML site for all workspaces, then exit |

## Notes

- `web_fetch` is human-in-the-loop: nothing is fetched until you approve the URL
  on the terminal (3 attempts; private/localhost URLs refused).
- Config lives in `mrai/settings.py` (pydantic-settings, env prefix `MRAI_`, `.env`).
- State lives under `.mrai/` (gitignored); use `--no-write` to skip it.
- Each logged answer also gets an HTML page in one shared static site,
  `~/.mrai/site/<workspace>--<topic>.html`, with keyword search and a file list
  across every workspace, plus a light/dark toggle. No server: pages share
  `app.js`/`style.css` (from `mrai/render/`) and `index.js` (the global search
  index, rewritten per answer). Run `--reindex` to rebuild the whole site.
- The read-only deny list is code-only and cannot be weakened by env.

## Tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
