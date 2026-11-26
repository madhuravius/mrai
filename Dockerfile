FROM python:3.14-alpine AS build
COPY --link --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.14-alpine
RUN apk add --no-cache git && adduser -D nonroot
COPY --link --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# the container is the sandbox; bwrap can't nest inside it
ENV MRAI_SANDBOXED=1
WORKDIR /repo
USER nonroot
ENTRYPOINT ["mrai"]

