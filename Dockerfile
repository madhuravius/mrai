FROM cgr.dev/chainguard/python:latest-dev AS build
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY mrai/ ./mrai/
RUN python -m venv /app/venv && /app/venv/bin/pip install --no-cache-dir .

FROM cgr.dev/chainguard/python:latest
COPY --from=build /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
WORKDIR /repo
USER nonroot
ENTRYPOINT ["mrai"]

