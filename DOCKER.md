# Docker deployment

## Start TableAgent

Create the environment file and fill in the answer and layout model settings:

```bash
cp .env.example .env
docker compose up --build -d
```

The API is available from the host at `http://localhost:3636` by default.
Check it with:

```bash
curl http://localhost:3636/health/ready
```

## Connect AXIOM_DE-RD

Configure AXIOM with:

```dotenv
TABLE_AGENT_BASE_URL=http://table-agent:3636
TABLE_AGENT_SERVICE_API_KEY=
```

The API key must match `TABLE_AGENT_SERVICE_API_KEY` in this project's `.env`.
Use the Compose service name and container port, not the host port, for
container-to-container traffic.

## Configuration

The container starts with `config.example.yaml`. Its model credentials and URLs
are populated from `.env`. To use a private configuration instead, mount it and
override the command:

```yaml
services:
  table-agent:
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    command:
      - table-agent-api
      - --config
      - /app/config.yaml
      - --host
      - 0.0.0.0
      - --port
      - "3636"
```

The named volumes preserve the structure cache and CLI output between container
restarts. API uploads and results remain ephemeral as defined by the service.
