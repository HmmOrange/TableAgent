# TableAgent Web

Start the API from the repository root:

```powershell
uv run table-agent-api --config config.yaml --host 127.0.0.1 --port 3636
```

Then start the Vite app:

```powershell
cd web
npm install
npm run dev
```

The web app is available at `http://127.0.0.1:5172` by default.

Vite proxies `/v1` to `http://127.0.0.1:3636`. Set `VITE_API_PROXY_TARGET` to
use another API URL. If the service requires a key, set
`TABLE_AGENT_SERVICE_API_KEY` before starting Vite so the development proxy adds
the header without exposing the key to browser code.
