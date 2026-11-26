FROM python:3.14-alpine AS build
COPY --link --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.14-alpine
RUN adduser -D nonroot
COPY --link --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# The container is the sandbox (read-only repo, no caps, non-root); bwrap could
# not nest inside it anyway, so tell mrai it is already sandboxed.
ENV MRAI_SANDBOXED=1
WORKDIR /repo
USER nonroot
ENTRYPOINT ["mrai"]

