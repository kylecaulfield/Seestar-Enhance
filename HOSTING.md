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

> **The fast path** is `docker run -d --name seestar-enhance -p 8000:8000
> --restart unless-stopped ghcr.io/kylecaulfield/seestar-enhance:latest`
> as documented in the README. Every code-only commit to `main`
> publishes a fresh multi-arch (amd64+arm64) image to ghcr.io — no
> git clone, no local build, no Node toolchain on the host.
>
> The walkthrough below is the **build-from-source** path, useful when
> you want to modify the code, run a non-`latest` branch, or air-gap a
> deployment. Skip to §2 for the Unraid-specific tutorial.

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

If you're on the build-from-source path (this section), `git pull` +
rebuild:

```sh
cd /opt/seestar-enhance
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

If you switched to the prebuilt image (the README's quick-run
recipe), `docker pull` + restart — no rebuild needed:

```sh
docker pull ghcr.io/kylecaulfield/seestar-enhance:latest
docker rm -f seestar-enhance
docker run -d --name seestar-enhance -p 8000:8000 \
  --restart unless-stopped \
  ghcr.io/kylecaulfield/seestar-enhance:latest
```

A `:latest` build is published per code-only commit to `main` (see
`How CI publishes the image` further down for the exact path filter).
Documentation-only commits don't trigger a rebuild, so a `docker pull`
is a no-op when the bytes haven't changed.

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

> **TL;DR.** Pull the prebuilt image from `ghcr.io/kylecaulfield/seestar-enhance:latest`
> and you're one Compose Up away from a working install. No `git clone`,
> no local build, no Node toolchain on the Unraid box.
>
> Every code-only commit to `main` triggers a fresh multi-arch
> (amd64 + arm64) build via the GitHub Actions workflow at
> `.github/workflows/docker.yml`. Documentation-only commits skip the
> rebuild — see the path filter in that file. So a `docker pull` always
> gives you the latest behaviour without busy-rebuilding for README
> tweaks.

### 2.1 One-time Unraid setup

1. **Install Community Applications (CA)** if you haven't already.
   Unraid GUI → **Plugins** → **Install Plugin** → paste
   `https://raw.githubusercontent.com/Squidly271/community.applications/master/plugins/community.applications.plg`
   → **Install**.
2. **Install Compose Manager.** Apps tab → search **Compose Manager**
   (by *dcflachs*) → **Install**.
3. **Install User Scripts** (optional but useful for the auto-update
   cron in §2.7). Apps → search **User Scripts** → **Install**.

### 2.2 Create the appdata folder

Unraid convention is to keep per-app config under
`/mnt/user/appdata/<app>`. Open the terminal (top-right icon on the
Unraid GUI) or SSH in:

```sh
mkdir -p /mnt/user/appdata/seestar-enhance/jobs
```

That's all that's needed in appdata — the image bundles its own code,
and the only mount-worthy state is the per-job temp directory used by
the pipeline (so jobs survive a container restart).

If you want to drop FITS files onto the server for the included CLI to
process, also make a `seestar` user share in the Unraid GUI →
**Shares** → **Add Share** → name `seestar`. The compose mounts it as
`/app/samples/` inside the container. (Optional — you can also just
upload through the web UI.)

### 2.3 Register the stack in Compose Manager

1. Unraid GUI → **Docker** tab → scroll down → **Compose Manager** →
   **Add New Stack**.
2. Name it `seestar-enhance`.
3. Click the gear icon next to the stack → **Edit Stack** → **Compose
   File**.
4. Paste:

   ```yaml
   services:
     seestar-enhance:
       image: ghcr.io/kylecaulfield/seestar-enhance:latest
       container_name: seestar-enhance
       ports:
         - "8000:8000"
       volumes:
         # Per-job temp dir. Keeping it on appdata means jobs survive
         # a container restart (the in-memory job dict is rebuilt by
         # the orphan-sweep on startup).
         - /mnt/user/appdata/seestar-enhance/jobs:/tmp/seestar-enhance-jobs
         # Optional: bind a FITS share so the CLI can process files
         # without uploading. Drop or rename if not needed.
         - /mnt/user/seestar:/app/samples
       environment:
         - PYTHONUNBUFFERED=1
         # CORS allowlist. Leave the default for LAN-only use behind
         # the nginx-proxy-manager pattern in §2.5; set to your real
         # SPA origin if fronting with a custom hostname. The literal
         # "*" works but is only safe behind a reverse proxy that does
         # its own auth.
         - ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]
         interval: 30s
         timeout: 5s
         retries: 5
   ```

   The image serves both API and SPA on port 8000 — there's no
   separate frontend container any more (FastAPI mounts the built
   React bundle at `/`).

5. Save the compose file.

The image is non-root (UID 10001). If you're using a tight Unraid
share permission, make sure `/mnt/user/appdata/seestar-enhance/jobs`
is writable by that UID:

```sh
chown -R 10001:10001 /mnt/user/appdata/seestar-enhance
```

### 2.4 Start the stack

1. Back on the Compose Manager screen, click **Compose Up** on the
   `seestar-enhance` stack.
2. First run pulls the multi-arch image (~600 MB compressed) from
   ghcr.io. Subsequent updates are layer-cached and only download the
   delta — typically ~50 MB per release.
3. When the container shows **Started** (and the healthcheck flips to
   `healthy` after ~30s), open a browser on the LAN:
   - Web UI: `http://<UNRAID-IP>:8000`
   - Health probe: `http://<UNRAID-IP>:8000/health`

### 2.5 Expose to the LAN with nicer URLs (optional)

If you want `http://seestar.local` instead of `:8000`, the typical
Unraid pattern is **NGINX Proxy Manager** (also from CA — easier than
SWAG for one-off setups):

1. Install NGINX Proxy Manager from CA. Accept defaults, note the
   admin port (default 81).
2. Open `http://<UNRAID-IP>:81`, log in with the default creds, change
   the password.
3. **Hosts → Proxy Hosts → Add Proxy Host:**
   - Domain Names: `seestar.local` (or your DNS name).
   - Scheme: `http`, Forward Hostname: `<UNRAID-IP>`, Forward Port:
     `8000`.
   - Enable **Block Common Exploits** and **Websockets Support**.
   - SSL tab: request a Let's Encrypt cert if the domain is publicly
     resolvable, otherwise leave as HTTP for LAN-only use.
4. If you set a custom hostname, update `ALLOWED_ORIGINS` in the
   compose file to match — e.g.
   `ALLOWED_ORIGINS=https://seestar.local`.

### 2.6 Put FITS files where the container can see them

Drop FITS files into `/mnt/user/seestar/` (the share you mounted at
`/app/samples/` in the compose). To run the CLI end-to-end on a sample
from the Unraid terminal:

```sh
docker exec -it seestar-enhance \
    python -m app.pipeline \
    /app/samples/nebula.fits \
    /app/samples/outputs/nebula.png \
    --profile nebula -v
```

For a whole night's batch:

```sh
docker exec -it seestar-enhance \
    python -m app.pipeline \
    --batch /app/samples/*.fit \
    --output-dir /app/samples/outputs \
    --jobs 0      # one worker per CPU core
```

### 2.7 Updating

Every code-only commit to `main` rebuilds the image and republishes
`:latest` to ghcr.io. To pull the new image and restart the container:

```sh
docker pull ghcr.io/kylecaulfield/seestar-enhance:latest
docker compose -p seestar-enhance up -d
```

Or, in the Compose Manager UI: click the stack → **Compose Down** →
**Compose Up**.

For automation, drop this in a User Script (**Settings** → **User
Scripts** → **Add New Script** → set a daily or weekly cron):

```sh
#!/bin/bash
set -eu
docker pull ghcr.io/kylecaulfield/seestar-enhance:latest
# Compose Manager projects live under /boot/config/plugins.
cd /boot/config/plugins/compose.manager/projects/seestar-enhance
docker compose up -d
```

The User Script is idempotent — if the digest hasn't changed, `docker
pull` is a no-op and `compose up -d` doesn't restart the container.

### Pinning to a specific release

If you don't want auto-updates, pin to a specific commit-SHA tag:

```yaml
image: ghcr.io/kylecaulfield/seestar-enhance:sha-abc1234
```

Every build publishes a `sha-<7>` tag in addition to `latest`, so you
can roll back to any prior commit by changing this line and running
**Compose Down** + **Compose Up**. Once we cut semver releases,
`v1.2.3` and floating `v1.2` / `v1` tags will be available too.

### 2.8 Troubleshooting

- **Port collision on 8000.** Edit the compose file's `ports:` mapping
  — e.g. `"8010:8000"`.
- **Image pull fails.** ghcr.io is rate-limited for unauthenticated
  pulls (~100/hour per IP). On a busy NAT'd LAN the limit can hit
  unexpectedly. Authenticate with a GitHub PAT:
  `docker login ghcr.io -u <username> -p <token>`.
- **Permission errors on the jobs volume.** The container runs as UID
  10001. `chown -R 10001:10001 /mnt/user/appdata/seestar-enhance`.
- **Healthcheck failing.** `docker logs seestar-enhance` usually
  reveals it. Most often a bad `ALLOWED_ORIGINS` value (must be a
  comma-separated list, no spaces around commas).
- **Web UI shows "Backend: unreachable".** The SPA is served by the
  same container as the API — there's no cross-origin issue out of
  the box. If you're behind NGINX Proxy Manager, make sure
  Websockets Support is on and `ALLOWED_ORIGINS` matches the proxied
  hostname.
- **FITS upload returns HTTP 429.** Six concurrent jobs are already
  in flight. Wait for them to finish, or bump `_MAX_INFLIGHT_JOBS` in
  `backend/app/main.py` and rebuild from source (the prebuilt image
  uses the default of 6).

### 2.9 Backups

Unraid's native CA Backup/Restore Appdata plugin is the easy path:
add `/mnt/user/appdata/seestar-enhance` to its backup set. Per-job
temp dirs live there and the orphan-sweep on container start only
prunes terminal jobs older than the TTL (1 hour by default).

If you're using a separate `seestar` user share for FITS files, add
that to the backup set too — or let the array parity handle drive-
failure risk and rsync it off-box for the real backup.

---

## How CI publishes the image

(For the curious; you don't need to read this to host the app.)

The `.github/workflows/docker.yml` workflow runs on every commit to
`main`, but only when files that actually affect the image change:

- `Dockerfile`
- `backend/app/**` (runtime code)
- `backend/requirements.txt` / `pyproject.toml` (Python deps)
- `frontend/**` (the SPA built into the runtime image)
- `.dockerignore`

Documentation commits (README, BACKLOG, HOSTING, the CI workflow
itself) skip the rebuild — the image bytes wouldn't change. Tagged
releases (`v*.*.*`) and manual `workflow_dispatch` (e.g. for a
base-image security refresh) always rebuild regardless.

Tags published per build:
- `:latest` — the most recent code-change build on `main`
- `:sha-<7>` — immutable, useful for pinning rollbacks
- `:v1.2.3`, `:1.2`, `:1` — on semver tag pushes

Multi-arch (amd64 + arm64) — same image on Intel servers, ARM SBCs,
and Apple Silicon dev boxes. QEMU does the cross-compile in CI.

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
