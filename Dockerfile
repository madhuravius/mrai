FROM python:3.15-alpine@sha256:0f9552bb28711333da16915ec146f5e6bb44a7fdac9ced15dc17c2e0206346dd AS build
COPY --link --from=ghcr.io/astral-sh/uv:0.9.14@sha256:fef8e5fb8809f4b57069e919ffcd1529c92b432a2c8d8ad1768087b0b018d840 /uv /uvx /bin/
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.15-alpine@sha256:0f9552bb28711333da16915ec146f5e6bb44a7fdac9ced15dc17c2e0206346dd
RUN apk add --no-cache git && adduser -D nonroot && python -m pip uninstall -y pip
COPY --link --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# the container is the sandbox; bwrap can't nest inside it
ENV MRAI_SANDBOXED=1
WORKDIR /repo
USER nonroot
ENTRYPOINT ["mrai"]

