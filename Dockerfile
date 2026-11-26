FROM python:3.14-alpine@sha256:5862d971d1f0ae26b258cefa4e893556b1d3aba81bd2ff30888704ad378bab67 AS build
COPY --link --from=ghcr.io/astral-sh/uv:0.9.14@sha256:fef8e5fb8809f4b57069e919ffcd1529c92b432a2c8d8ad1768087b0b018d840 /uv /uvx /bin/
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.14-alpine@sha256:5862d971d1f0ae26b258cefa4e893556b1d3aba81bd2ff30888704ad378bab67
RUN apk add --no-cache git && adduser -D nonroot && python -m pip uninstall -y pip
COPY --link --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# the container is the sandbox; bwrap can't nest inside it
ENV MRAI_SANDBOXED=1
WORKDIR /repo
USER nonroot
ENTRYPOINT ["mrai"]

