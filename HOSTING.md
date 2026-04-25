# Hosting

This guide walks through two scenarios:

1. **[Generic self-hosting](#1-generic-self-hosting-docker)** — any Linux
   box (VPS, home server, Raspberry Pi 4/5 with 4 GB+ RAM) running Docker.
2. **[Unraid tutorial](#2-unraid-tutorial)** — step-by-step for Unraid
   6.12+ using Compose Manager.

The app ships as two containers: `backend` (FastAPI on port 8000) and
`frontend` (Vite dev server on port 5173). A single `docker-compose.yml`
at the repo root brings them up together.

> **Note on the stock compose file.** `docker-compose.yml` bind-mounts
> `./backend` and `./frontend` into the containers so edits take effect
> live — that's a development setup. For production hosting you want
> baked-in images. This guide shows a minimal production-style override
> in the relevant steps.

---

## 1. Generic self-hosting (Docker)

### 1.1 Prerequisites

- A host with Docker Engine 24+ and the Compose plugin. On a fresh Debian
  or Ubuntu box:

  ```sh
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  newgrp docker
  docker compose version   # should print v2.x
  ```

- `git`, `curl`, and roughly 3 GB of free disk for the images.
- If you plan to expose the service on the public internet, a DNS name
  pointing at the host (e.g. `seestar.example.com`).

### 1.2 Clone the repo

```sh
sudo mkdir -p /opt/seestar-enhance
sudo chown "$USER": /opt/seestar-enhance
cd /opt/seestar-enhance
git clone https://github.com/kylecaulfield/Seestar-Enhance.git .
```

`/opt/seestar-enhance` is a convention — any directory is fine. Keep the
`samples/` directory on fast storage if you're processing large FITS
files.

### 1.3 Build and start

```sh
docker compose up -d --build
```

First build pulls base images and installs Python/Node dependencies. On
a 4-core VPS expect ~4–8 minutes. Subsequent `up -d` calls are fast.

Verify:

```sh
curl -fsS http://localhost:8000/health
# -> {"status":"ok"}
```

Open http://<host>:5173 in a browser for the frontend placeholder.

### 1.4 Production override

For a long-running host, disable the dev-mode bind mounts and pin image
tags. Create `docker-compose.prod.yml` next to the stock file:

```yaml
# docker-compose.prod.yml
services:
  backend:
    volumes:
      # Only the samples directory is a bind mount; source is baked in.
      - ./samples:/app/samples
    restart: unless-stopped
  frontend:
    # Serve the built bundle rather than the dev server.
    command: ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "5173"]
    restart: unless-stopped
```

Start with both files composed:

```sh
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

### 1.5 Reverse proxy + HTTPS

Don't expose the raw ports to the public internet. Put a reverse proxy
in front and let it terminate TLS.

**Example: Caddy** — `/etc/caddy/Caddyfile`:

```
seestar.example.com {
    reverse_proxy /api/* localhost:8000
    reverse_proxy localhost:5173
}
```

Caddy auto-provisions a Let's Encrypt certificate. Reload:

```sh
sudo systemctl reload caddy
```

**Example: Nginx** — `/etc/nginx/sites-available/seestar.conf` (assumes
certbot has issued a cert):

```nginx
server {
    listen 443 ssl http2;
    server_name seestar.example.com;

    ssl_certificate     /etc/letsencrypt/live/seestar.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/seestar.example.com/privkey.pem;

    client_max_body_size 500M;   # raise when you wire up FITS uploads.

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://127.0.0.1:5173/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable, test, reload:

```sh
sudo ln -s /etc/nginx/sites-available/seestar.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 1.6 Firewall

Close 8000 and 5173 at the firewall; only 80/443 should be reachable
from the public internet.

```sh
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 1.7 Persistence and backups

Everything important lives under the repo directory:

| Path | What it holds | Back up? |
| --- | --- | --- |
| `samples/*.fits` | User-uploaded FITS files | Yes |
| `samples/outputs/*.png` | Pipeline outputs | Optional (re-derivable) |
| `backend/app/models/` | (v2) ONNX weights | Yes |

A nightly `rsync` of the repo directory to another host is enough for v1.

### 1.8 Updating

```sh
cd /opt/seestar-enhance
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

### 1.9 Logs and health

```sh
docker compose logs -f backend
docker compose logs -f frontend
docker compose ps           # see health status
```

The backend container has a baked-in healthcheck against `/health` every
10 s.

### 1.10 Uninstall

```sh
cd /opt/seestar-enhance
docker compose down
docker image prune -f
```

---

## 2. Unraid tutorial

Tested on Unraid 6.12+ with the Community Applications and Compose
Manager plugins. If you're on an older Unraid, upgrade first — Compose
Manager is the cleanest way to run multi-container apps there.

### 2.1 One-time Unraid setup

1. **Install Community Applications (CA)** if you haven't already.
   Unraid GUI → **Plugins** → **Install Plugin** → paste
   `https://raw.githubusercontent.com/Squidly271/community.applications/master/plugins/community.applications.plg`
   → **Install**.
2. **Install Compose Manager.** Apps tab → search **Compose Manager**
   (by *dcflachs*) → **Install**.
3. **Install User Scripts** (optional but useful for git-pull cron
   later). Apps → search **User Scripts** → **Install**.

### 2.2 Create the appdata folder

Unraid convention is to keep per-app config under
`/mnt/user/appdata/<app>`. Open the terminal (top-right icon on the
Unraid GUI) or SSH in:

```sh
mkdir -p /mnt/user/appdata/seestar-enhance
cd /mnt/user/appdata/seestar-enhance
git clone https://github.com/kylecaulfield/Seestar-Enhance.git .
```

If you prefer to keep FITS files on a user share instead of appdata
(FITS files can be large, and user shares often span the array), make a
share called `seestar` in the Unraid GUI → **Shares** → **Add Share** →
name `seestar`, then symlink it:

```sh
rm -rf /mnt/user/appdata/seestar-enhance/samples
ln -s /mnt/user/seestar /mnt/user/appdata/seestar-enhance/samples
```

### 2.3 Register the stack in Compose Manager

1. Unraid GUI → **Docker** tab → scroll down → **Compose Manager** →
   **Add New Stack**.
2. Name it `seestar-enhance`.
3. Click the gear icon next to the stack → **Edit Stack** → **Compose
   File**.
4. Paste the contents of this production-ready compose (adapted from the
   repo's `docker-compose.yml` with bind-mounts pointed at the Unraid
   paths):

   ```yaml
   services:
     backend:
       build: /mnt/user/appdata/seestar-enhance/backend
       container_name: seestar-backend
       ports:
         - "8000:8000"
       volumes:
         - /mnt/user/appdata/seestar-enhance/samples:/app/samples
       environment:
         - PYTHONUNBUFFERED=1
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]
         interval: 30s
         timeout: 5s
         retries: 5

     frontend:
       build: /mnt/user/appdata/seestar-enhance/frontend
       container_name: seestar-frontend
       ports:
         - "5173:5173"
       environment:
         - VITE_API_BASE=http://<UNRAID-IP>:8000
       depends_on:
         - backend
       restart: unless-stopped
   ```

   Replace `<UNRAID-IP>` with your server's LAN IP (e.g. `192.168.1.10`)
   so the frontend knows where to find the backend from a LAN browser.

5. Save the compose file.

### 2.4 Build and start

1. Back on the Compose Manager screen, click **Compose Up** on the
   `seestar-enhance` stack.
2. Watch the log window — the first run builds the Python and Node
   images, which takes 5–10 minutes on typical Unraid hardware.
3. When both containers show **Started**, open a browser on the LAN:
   - Frontend: `http://<UNRAID-IP>:5173`
   - Backend health: `http://<UNRAID-IP>:8000/health`

### 2.5 Expose to the LAN with nicer URLs (optional)

If you want `http://seestar.local` instead of `:5173`, the typical Unraid
pattern is:

- Install **SWAG** (a preconfigured Nginx + Let's Encrypt container) from
  Community Apps if you don't already have a reverse proxy.
- Or install **NGINX Proxy Manager** — a web UI is easier for one-off
  setups.

**NGINX Proxy Manager quick path:**

1. Install NGINX Proxy Manager from CA. Accept defaults, note the admin
   port (default 81).
2. Open `http://<UNRAID-IP>:81`, log in with the default creds, change
   the password.
3. **Hosts → Proxy Hosts → Add Proxy Host:**
   - Domain Names: `seestar.local` (or your DNS name).
   - Scheme: `http`, Forward Hostname: `<UNRAID-IP>`, Forward Port:
     `5173`.
   - Enable **Block Common Exploits** and **Websockets Support**.
   - SSL tab: request a Let's Encrypt cert if the domain is publicly
     resolvable, otherwise leave as HTTP for LAN-only use.
4. Repeat for `api.seestar.local → <UNRAID-IP>:8000` if you want a
   separate API hostname, and update `VITE_API_BASE` in the compose file
   accordingly.

### 2.6 Put FITS files where the container can see them

Drop FITS files into `/mnt/user/seestar/` (or
`/mnt/user/appdata/seestar-enhance/samples/` if you didn't make a
separate share). The backend container sees them at `/app/samples/`
inside the container.

To run the CLI end-to-end on a sample from the Unraid terminal:

```sh
docker exec -it seestar-backend \
    python -m app.pipeline \
    /app/samples/nebula.fits \
    /app/samples/outputs/nebula.png \
    --profile nebula -v
```

### 2.7 Updating

```sh
cd /mnt/user/appdata/seestar-enhance
git pull
```

Then Compose Manager → **Compose Down** → **Compose Up** on the stack to
rebuild with the latest code.

For automation, drop this in a User Script (**Settings** → **User
Scripts** → **Add New Script** → set a weekly cron):

```sh
#!/bin/bash
cd /mnt/user/appdata/seestar-enhance || exit 1
git pull --ff-only || exit 2
cd /boot/config/plugins/compose.manager/projects/seestar-enhance || exit 3
docker compose up -d --build
```

### 2.8 Troubleshooting

- **Port collisions.** If you already run something on 8000 or 5173, edit
  the compose file's `ports:` mapping — e.g. `"8010:8000"`.
- **Permission errors on samples.** Unraid shares default to
  `99:100` (nobody:users). The backend image runs as root inside the
  container, so it can read/write fine — but if you've tightened share
  permissions, loosen them or chown.
- **Healthcheck failing.** `docker logs seestar-backend` usually reveals
  the cause; most often it's a Python dependency conflict during the
  first build. Re-run `Compose Up` with the log window open and read
  the pip output.
- **Build OOM on older hardware.** Unraid servers with 4 GB RAM can
  struggle during the `pip install scipy astropy` step. Temporarily stop
  other containers, or set a swapfile.
- **Frontend shows "Backend: unreachable".** `VITE_API_BASE` in the
  compose file must point at a host/port the browser can reach. From a
  LAN browser, `localhost` is the client, not the Unraid server — that's
  why we use `<UNRAID-IP>:8000` above. If you're using a reverse proxy,
  set it to the proxied URL instead (e.g. `https://api.seestar.local`).

### 2.9 Backups

Unraid's native CA Backup/Restore Appdata plugin is the easy path: add
`/mnt/user/appdata/seestar-enhance` to its backup set.

If you're using a separate `seestar` user share for FITS files, add that
too, or let the array parity handle drive-failure risk and rsync it
off-box for the real backup.

---

## Security knobs

Set these on the backend container if you expose the API beyond
localhost:

| Env var | Default | Purpose |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | `http://localhost:8000,http://localhost:5173,http://127.0.0.1:*` | Comma-separated CORS allowlist. Set to the full origin (`https://astro.example.com`) of any SPA you front the API with. The literal `*` is supported but only sane behind a reverse proxy that does its own auth. |

Every response also carries a tight `Content-Security-Policy`, plus
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and
`Referrer-Policy: no-referrer`. These are added by middleware in
`backend/app/main.py`; you can override with your own headers at the
reverse-proxy layer if needed.

The backend has **no built-in authentication**. If you expose it on a
public port, put a reverse proxy (Caddy, NGINX Proxy Manager, etc.) in
front and require basic-auth or OIDC at that layer.

## Resource sizing

Quick guidance for picking a host:

| Sensor | Image size | Per-stage RAM (peak) | Pipeline runtime (CPU) |
| --- | --- | --- | --- |
| Seestar S30 (1920×1080) | ~8 MP | ~1.5 GB | 20–45 s |
| Seestar S50 (1920×1080) | ~8 MP | ~1.5 GB | 20–45 s |
| Stacked 4× mosaic | ~32 MP | ~5 GB | 2–4 min |

BM3D is the bottleneck (CPU-bound, scales with pixel count). Pick a host
with at least 4 cores and 8 GB RAM for comfortable interactive use.
Anything smaller will still work for single images but expect to wait.
