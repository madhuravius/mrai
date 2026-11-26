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

## Config and providers

Settings come from `MRAI_*` env vars, `.env`, then `~/.mrai/mrai.yml` and
`.mrai/mrai.yml` (repo wins). Profiles switch provider, endpoints and models in
one go; keys stay in the environment:

```yaml
profile: openrouter # default; MRAI_PROFILE or --profile overrides
profiles:
  openrouter:
    provider: openrouter
    pro_model: deepseek/deepseek-v4-pro
  anthropic:
    provider: anthropic # native Claude API
    api_key_env: ANTHROPIC_API_KEY
    jev_api_key_env: OPENROUTER_API_KEY # Jev is OpenRouter-only
    pro_model: claude-opus-5-5
    flash_model: claude-haiku-5-5
```

`provider` is `openrouter`, `anthropic`, or `openai` (any OpenAI-compatible
gateway at `base_url`). Without `jev_api_key_env`, Jev uses the model key; off
OpenRouter, set it or `route_enabled: false` (web fetch needs Jev). See
`mrai.example.yml`.

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
| `--profile <name>` | use a profile from `mrai.yml` |
| `--reindex` | rebuild the HTML site for all workspaces, then exit |

## Notes

- `web_fetch` is human-in-the-loop: nothing is fetched until you approve the URL
  on the terminal (3 attempts; private/localhost URLs refused).
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
make test-container   # sandbox checks against ./ai's real docker flags
```

## Verifying a release image

`ai --update` pulls the newest release, verifies its signature with cosign, and
pins `ai` to that digest; a failed check leaves the current image in place. A
local `make install` / `make docker-build` replaces the pin. To verify by hand
(cosign v3+):

```bash
cosign verify ghcr.io/madhuravius/mrai:v0.3.0 \
  --certificate-identity-regexp '^https://github.com/madhuravius/mrai/\.github/workflows/release\.yml@refs/tags/v' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

## License

MIT — see [LICENSE](LICENSE).
