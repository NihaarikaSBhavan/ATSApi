# ATS Engine — Production Deployment Guide

## What you have
- FastAPI backend (uvicorn, port 8000)
- nginx reverse proxy (port 80/443)
- Dynamic skill graph (DynamicSkillGraph)
- 3-tier extraction: SLM → spaCy → regex
- SSE streaming pipeline with live progress

---

## Phase 1 — GitHub setup

```bash
# 1. Create a new repo on github.com, then:
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ats-engine.git
git push -u origin main
```

`.env` is in `.gitignore` — it will NOT be pushed. Good.

---

## Phase 2 — Provision a server

### Option A: Render (easiest, no SSH needed)
1. Go to render.com → New → Web Service
2. Connect your GitHub repo
3. Set:
   - Environment: Docker
   - Port: 8000
   - Instance type: Starter ($7/mo) or Standard ($25/mo for more RAM)
4. Add environment variables (from `.env.example`) in the Render dashboard
5. Deploy → done. Render gives you HTTPS automatically.

**Note for Render:** remove the `nginx` service from `docker-compose.yml`
(Render handles TLS termination itself). The `api` service runs alone.

### Option B: GCP VM (full nginx stack, closest to Colab setup)

```bash
# Create VM (e2-small = 2 vCPU, 2 GB RAM, ~$15/mo)
gcloud compute instances create ats-engine \
  --machine-type=e2-small \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --tags=http-server,https-server \
  --zone=us-central1-a

# Open firewall
gcloud compute firewall-rules create allow-http  --allow=tcp:80  --target-tags=http-server
gcloud compute firewall-rules create allow-https --allow=tcp:443 --target-tags=https-server

# SSH in
gcloud compute ssh ats-engine
```

---

## Phase 3 — Server setup (GCP/AWS VM only)

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install git
sudo apt-get install -y git

# Clone your repo
git clone https://github.com/YOUR_USERNAME/ats-engine.git
cd ats-engine

# Copy .env and fill in values
cp .env.example .env
nano .env   # set FORCE_REGEX, MIN_FREE_MB, etc.

# Start everything
docker compose up -d --build

# Watch logs
docker compose logs -f api
```

Check it's working:
```bash
curl http://localhost/health
# → {"status":"ok"}

curl http://localhost/info
# → {"backend":"spacy","model":"en_core_web_sm"}  (or "slm" if enough disk)
```

---

## Phase 4 — HTTPS with your domain (GCP/AWS only)

```bash
# Point your domain's A record to the server's external IP first, then:

# Get the external IP
gcloud compute instances describe ats-engine --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Install certbot
sudo apt-get install -y certbot

# Get certificate (stops nginx briefly)
docker compose stop nginx
sudo certbot certonly --standalone -d yourdomain.com
docker compose start nginx

# Update nginx/nginx.conf:
# 1. Uncomment the HTTPS server block
# 2. Replace "yourdomain.com" with your actual domain
# 3. Uncomment the redirect in the HTTP block

# Restart nginx
docker compose restart nginx
```

Auto-renew (certbot does this automatically, but also add to cron):
```bash
sudo crontab -e
# Add: 0 3 * * * certbot renew --quiet && docker compose -f /home/USER/ats-engine/docker-compose.yml restart nginx
```

---

## Phase 5 — CI/CD (GitHub Actions auto-deploy)

In your GitHub repo → Settings → Secrets → Actions, add:

| Secret name      | Value |
|-----------------|-------|
| `SERVER_HOST`   | Your server's external IP |
| `SERVER_USER`   | Your SSH username (e.g. `user`) |
| `SERVER_SSH_KEY`| Contents of your private SSH key (`cat ~/.ssh/id_rsa`) |

Now every `git push origin main` automatically:
1. SSHs into your server
2. Pulls latest code
3. Rebuilds Docker image
4. Restarts containers

---

## Disk space notes

The SLM (Qwen2-0.5B) needs ~400 MB free to download.
A default e2-small has 10 GB boot disk — increase to 20 GB (done above).

If disk is still tight, set in `.env`:
```
FORCE_REGEX=0
MIN_FREE_MB=0    # attempt SLM regardless of disk check
```

Or skip SLM entirely and rely on spaCy (works well):
```
FORCE_REGEX=0
MIN_FREE_MB=999999   # always skip SLM
```

---

## Useful commands

```bash
# View live logs
docker compose logs -f api

# Restart just the API (after code change)
docker compose restart api

# Full rebuild (after dependency change)
docker compose up -d --build

# Check container status
docker compose ps

# Shell into running container
docker compose exec api bash

# Check disk usage
df -h
docker system df
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| GET | `/health` | Liveness check |
| GET | `/info` | Active extraction backend |
| GET | `/info/wait` | Waits until extractor loaded |
| POST | `/analyze` | Full JSON result |
| POST | `/analyze/stream` | SSE with live progress |
| GET | `/graph/related?skill=python` | Related skills |
| GET | `/docs` | Swagger API docs |
