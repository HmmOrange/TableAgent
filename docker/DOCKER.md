# Docker deployment

## Configuration

From the repository root, create the Docker environment file:

```bash
cp docker/.env.example docker/.env
```

Set the model URLs, names, keys, and optional service API key in `docker/.env`.
Model thinking is disabled by default. Set `TABLE_AGENT_ENABLE_THINKING=true` to
enable it for answer and layout model requests.

## Start

```bash
docker network create axiom-k8s
docker compose --env-file docker/.env -f docker/docker-compose.yml up --build -d
curl http://localhost:3636/health/ready
```

Create `axiom-k8s` only once. The API is available at `http://localhost:3636`.

## Connect

Configure AXIOM with:

```dotenv
TABLE_AGENT_BASE_URL=http://table-agent:3636
TABLE_AGENT_SERVICE_API_KEY=
```

The API key must match `TABLE_AGENT_SERVICE_API_KEY` in `docker/.env`.

Pipeline artifacts are persisted in the repository's host `outputs/` directory,
which is mounted to `/app/outputs` in the container. This matches CLI output
storage and keeps artifacts available after the container is recreated.
