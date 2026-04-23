# PHN Technology — WhatsApp AI Agent 🤖

An intelligent WhatsApp agent for **PHN Technology Pvt Limited**, powered by LangGraph + RAG + Ollama.  
Handles student queries about courses, workshops, bootcamps, pricing, and placements — all running **locally on your GPU**.

---

## 📋 Table of Contents

1. [Architecture](#-architecture)
2. [Prerequisites](#-prerequisites)
3. [Environment Configuration](#-environment-configuration)
4. [Running the Agent](#-running-the-agent)
5. [All Docker Commands](#-all-docker-commands)
6. [Checking Leads & Monitoring](#-checking-leads--monitoring)
7. [Testing the Agent](#-testing-the-agent)
8. [WhatsApp & ngrok Setup](#-whatsapp--ngrok-setup)
9. [Updating the Knowledge Base](#-updating-the-knowledge-base)
10. [Troubleshooting](#-troubleshooting)
11. [Project Structure](#-project-structure)
12. [Tech Stack](#-tech-stack)

---

## 🏗️ Architecture

```
User (WhatsApp) → Meta Cloud API → ngrok tunnel → FastAPI Webhook (:8000)
                                                         ↓
                                                   LangGraph Agent
                                                   ├── Intent Classifier
                                                   ├── RAG Retriever (ChromaDB)
                                                   ├── Response Generator (Aryan persona)
                                                   ├── Lead Scorer (hot/warm/cold)
                                                   ├── Re-engagement Worker
                                                   └── Human Handoff (Counselor Alert)
                                                         ↓
                                                   Ollama (Local GPU)
                                                   ├── Qwen 2.5 7B Instruct (Chat)
                                                   └── nomic-embed-text (Embeddings)
                                                         ↓
                                                   SQLite (Lead DB) + ChromaDB (Vector DB)
```

---

## ✅ Prerequisites

| Requirement | Notes |
|---|---|
| **Docker Desktop** | With WSL2 backend on Windows |
| **NVIDIA GPU** | Drivers must be installed |
| **NVIDIA Container Toolkit** | For GPU passthrough in Docker |
| **ngrok** | For exposing local server to Meta webhook |

---

## ⚙️ Environment Configuration

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with these values:

```env
# --- Ollama Settings (do not change if using Docker Compose) ---
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct
OLLAMA_EMBED_MODEL=nomic-embed-text

# --- WhatsApp Business API (get from developers.facebook.com) ---
WHATSAPP_VERIFY_TOKEN=phn-tech-verify-2024
WHATSAPP_ACCESS_TOKEN=your_meta_access_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id_here

# --- Application Settings ---
APP_ENV=development
LOG_LEVEL=info
CHROMA_PERSIST_DIR=/app/chroma_data
SQLITE_DB_PATH=/app/data/phn_agent.db
```

> ⚠️ **Never commit your `.env` file to Git.** It is already in `.gitignore`.

---

## 🚀 Running the Agent

### First Time Setup (full start)

```bash
# Build image and start all services (Ollama + model-setup + agent)
docker compose up --build -d

# On first run, models are auto-pulled (takes 5–10 min for Qwen 7B)
# Watch model download progress:
docker compose logs -f model-setup
```

### Subsequent Starts (no rebuild needed)

```bash
# Start all services (containers already built)
docker compose up -d

# OR start only the agent (Ollama already running)
docker compose up -d agent-app
```

### Stop Everything

```bash
docker compose down
```

### Stop and Remove All Data (full reset)

```bash
# WARNING: Deletes all ChromaDB indexes, SQLite leads, and Ollama model cache
docker compose down -v
```

---

## 🐳 All Docker Commands

### Container Management

```bash
# Start all services
docker compose up -d

# Start with a fresh build (after code changes)
docker compose up --build -d

# Start only the agent (not Ollama — useful if Ollama is already running)
docker compose up -d agent-app

# Stop all services (keeps data volumes)
docker compose down

# Stop all + delete volumes (full reset — loses leads & ChromaDB)
docker compose down -v

# Restart a specific service
docker compose restart agent-app
docker compose restart ollama
```

### Viewing Status

```bash
# See all running containers with status and ports
docker ps

# See containers in a neat table format
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# See all containers (including stopped ones)
docker ps -a

# See services defined in this project
docker compose ps
```

### Viewing Logs

```bash
# Follow live logs for the agent
docker compose logs -f agent-app

# Follow live logs for Ollama
docker compose logs -f ollama

# Follow model download progress (first run only)
docker compose logs -f model-setup

# Last 50 lines from agent
docker logs phn-whatsapp-agent --tail 50

# Last 100 lines with timestamps
docker logs phn-whatsapp-agent --tail 100 --timestamps

# Follow live agent logs with container name
docker logs -f phn-whatsapp-agent
```

### Rebuilding After Code Changes

```bash
# Rebuild and restart only the agent (fastest for code changes)
docker compose up --build -d agent-app

# Rebuild everything from scratch (no cache)
docker compose build --no-cache
docker compose up -d
```

### Running Commands Inside Containers

```bash
# Open a bash shell inside the agent container
docker exec -it phn-whatsapp-agent bash

# Run the CLI test mode interactively
docker exec -it phn-whatsapp-agent python -m scripts.test_agent

# Run quick test script
docker exec -it phn-whatsapp-agent python -m scripts.quick_test

# Check Python imports manually
docker exec -it phn-whatsapp-agent python -c "from app.agent.prompts import RAG_RESPONSE_PROMPT; print('OK')"
```

### Ollama Model Management

```bash
# List all downloaded models inside Ollama container
docker exec phn-ollama ollama list

# Pull a model manually
docker exec phn-ollama ollama pull qwen2.5:7b-instruct
docker exec phn-ollama ollama pull nomic-embed-text

# Check Ollama API is responding
curl http://localhost:11435/api/tags

# Check which models are available via API
curl http://localhost:11435/api/tags | python -m json.tool

# Run a quick Ollama inference test
curl http://localhost:11435/api/generate -d '{"model":"qwen2.5:7b-instruct","prompt":"Hello","stream":false}'
```

### Volume & Storage

```bash
# List all Docker volumes
docker volume ls

# Inspect a volume (see where data is stored)
docker volume inspect phn_chroma_data
docker volume inspect phn_sqlite_data
docker volume inspect phn_ollama_data

# Delete only ChromaDB volume (forces re-indexing on next start)
docker volume rm phn_chroma_data

# Delete only leads database (WARNING: loses all lead data)
docker volume rm phn_sqlite_data
```

### Health Checks

```bash
# Agent health endpoint
curl http://localhost:8000/health

# Ollama health check
curl http://localhost:11435/api/tags

# Expected agent response:
# {"status":"healthy","service":"PHN Technology WhatsApp Agent","version":"1.0.0"}
```

---

## 📊 Checking Leads & Monitoring

### Via API (HTTP)

```bash
# Get all recent leads (last 50)
curl http://localhost:8000/api/leads

# Get only HOT leads (ready for sales follow-up)
curl http://localhost:8000/api/leads/hot

# Pretty-print JSON output (requires Python)
curl http://localhost:8000/api/leads | python -m json.tool
curl http://localhost:8000/api/leads/hot | python -m json.tool
```

### Via Swagger UI (Browser)

Open your browser and go to:

```
http://localhost:8000/docs
```

You'll see all available API endpoints with interactive forms — no curl needed.

### Via SQLite Directly (Inside Container)

```bash
# Open SQLite shell inside the container
docker exec -it phn-whatsapp-agent sqlite3 /app/data/phn_agent.db

# Once inside sqlite3 shell:

# List all tables
.tables

# See all leads (formatted)
.mode column
.headers on
SELECT phone, name, city, occupation, lead_score, interested_courses, total_messages, last_contact FROM leads ORDER BY last_contact DESC;

# See only HOT leads
SELECT phone, name, city, lead_score, interested_courses FROM leads WHERE lead_score = 'hot';

# See only WARM leads
SELECT phone, name, city, lead_score, interested_courses FROM leads WHERE lead_score = 'warm';

# Count leads by score
SELECT lead_score, COUNT(*) as count FROM leads GROUP BY lead_score;

# See full conversation history for a phone number
SELECT direction, message, intent, timestamp FROM conversations WHERE phone = '+91XXXXXXXXXX' ORDER BY timestamp;

# See last 20 conversations across all leads
SELECT phone, direction, intent, message, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 20;

# Exit sqlite3
.exit
```

### Lead Score Meanings

| Score | Meaning | Action |
|---|---|---|
| 🔥 `hot` | Asked about fees/registration, said "join karna hai", ready to enroll | Call immediately |
| 🌡️ `warm` | Asking about syllabus, placements, multi-turn interest | Send brochure, follow up |
| ❄️ `cold` | One-word replies, only greeted, no interest shown | Re-engage later |

### Sales Stage Meanings

| Stage | Meaning |
|---|---|
| `awareness` | Just found out about PHN, very early |
| `consideration` | Comparing options, asking detailed questions |
| `decision` | Asking about fees, registration, seat availability |
| `objection` | Expressed hesitation, price concern, "I'll think about it" |

---

## 🧪 Testing the Agent

### CLI Test Mode (No WhatsApp Needed)

```bash
# Interactive CLI test — type messages, get agent responses
docker exec -it phn-whatsapp-agent python -m scripts.test_agent

# Quick automated test
docker exec -it phn-whatsapp-agent python -m scripts.quick_test
```

### Test the Webhook Directly

```bash
# Simulate a WhatsApp message via curl
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "object": "whatsapp_business_account",
    "entry": [{
      "changes": [{
        "value": {
          "messages": [{
            "from": "919999999999",
            "type": "text",
            "text": {"body": "Hi, what courses do you offer?"}
          }]
        }
      }]
    }]
  }'
```

---

## 📱 WhatsApp & ngrok Setup

### Step 1: Start ngrok

```bash
# From the project root (or any terminal)
ngrok http 8000

# ngrok will show a public URL like:
# Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

### Step 2: Set Webhook in Meta Dashboard

1. Go to [Meta for Developers](https://developers.facebook.com/) → Your App → WhatsApp → Configuration
2. **Callback URL**: `https://abc123.ngrok-free.app/webhook`
3. **Verify Token**: Value of `WHATSAPP_VERIFY_TOKEN` in your `.env` (default: `phn-tech-verify-2024`)
4. Click **Verify and Save**
5. Subscribe to the `messages` field under Webhook Fields

### Step 3: Verify Connection

```bash
# Confirm ngrok tunnel is active
curl https://abc123.ngrok-free.app/health
# Should return: {"status":"healthy",...}
```

> ℹ️ ngrok URL **changes every time** you restart ngrok (on free plan). Update Meta webhook each time.

---

## 📚 Updating the Knowledge Base

1. Edit JSON files in the `knowledge_base/` directory:
   - `courses.json` — course details, fees, duration
   - `bootcamps.json` — bootcamp info
   - `faqs.json` — common questions and answers
   - `company_info.json` — PHN company details

2. Delete the old ChromaDB index and restart:

```bash
# Option A: Delete volume and restart (full re-index)
docker compose down
docker volume rm phn_chroma_data
docker compose up -d

# Option B: Restart agent only (re-indexes on startup)
docker compose restart agent-app
```

---

## 🛠️ Troubleshooting

### ❌ Agent container keeps restarting

```bash
# Check what error is causing the crash
docker logs phn-whatsapp-agent --tail 50

# Common causes:
# 1. SyntaxError in prompts.py or any Python file → fix the file, rebuild
# 2. Ollama not ready yet → wait 30s, then: docker compose restart agent-app
# 3. Missing .env values → check your .env file
```

### ❌ SyntaxError in prompts.py (em-dash or special character)

```bash
# Fix the file, then rebuild only the agent
docker compose up --build -d agent-app
```

### ❌ Ollama not responding / models not loaded

```bash
# Check if Ollama container is healthy
docker ps --format "table {{.Names}}\t{{.Status}}"

# Check Ollama logs
docker logs phn-ollama --tail 30

# Test Ollama API directly
curl http://localhost:11435/api/tags

# If models are missing, pull them manually
docker exec phn-ollama ollama pull qwen2.5:7b-instruct
docker exec phn-ollama ollama pull nomic-embed-text

# Restart Ollama
docker compose restart ollama
```

### ❌ Knowledge base / RAG not working

```bash
# Check indexing logs on startup
docker logs phn-whatsapp-agent | grep -i "index\|chroma\|embed"

# Force re-index by deleting ChromaDB volume
docker compose down
docker volume rm phn_chroma_data
docker compose up -d

# Verify ChromaDB loaded correctly (look for "X documents" in startup log)
docker logs phn-whatsapp-agent | grep "ChromaDB"
```

### ❌ WhatsApp messages not arriving (webhook not triggered)

```bash
# 1. Verify ngrok is running and tunnel is active
curl https://YOUR-NGROK-URL/health

# 2. Check agent is receiving requests
docker logs -f phn-whatsapp-agent

# 3. Check Meta webhook is verified (green tick in Meta dashboard)

# 4. Make sure you're subscribed to the "messages" webhook field in Meta

# 5. Verify token matches between .env and Meta dashboard
grep WHATSAPP_VERIFY_TOKEN .env
```

### ❌ GPU not being used by Ollama

```bash
# Check if NVIDIA runtime is available in Docker
docker info | grep -i nvidia

# Check GPU is visible inside Ollama container
docker exec phn-ollama nvidia-smi

# If nvidia-smi fails, reinstall NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Verify Ollama is using GPU (look for GPU in logs)
docker logs phn-ollama | grep -i gpu
```

### ❌ Port already in use

```bash
# Find what's using port 8000
netstat -ano | findstr :8000

# Find what's using port 11435
netstat -ano | findstr :11435

# Kill the process using the port (replace PID with actual PID)
taskkill /PID <PID> /F

# Or change the port mapping in docker-compose.yml
```

### ❌ Container won't start — "network not found" or volume errors

```bash
# Full clean reset
docker compose down -v
docker system prune -f
docker compose up --build -d
```

### ❌ Out of disk space (Docker images/volumes)

```bash
# Check Docker disk usage
docker system df

# Remove unused images, containers, networks (safe)
docker system prune -f

# Remove ALL unused volumes (WARNING: loses data)
docker volume prune -f

# Nuclear option — remove everything Docker related
docker system prune -a --volumes -f
```

---

## 📁 Project Structure

```
whatsapp-agent/
├── app/
│   ├── main.py              # FastAPI server, startup, API routes
│   ├── config.py            # All config via environment variables
│   ├── agent/
│   │   ├── graph.py         # LangGraph agent definition & flow
│   │   ├── nodes.py         # Agent processing nodes (classify, rag, score...)
│   │   ├── state.py         # Conversation state schema
│   │   └── prompts.py       # All system prompts (Aryan persona, bilingual)
│   ├── rag/
│   │   ├── indexer.py       # Loads knowledge_base/ → ChromaDB
│   │   └── retriever.py     # Semantic search over ChromaDB
│   ├── whatsapp/
│   │   ├── client.py        # Meta WhatsApp API client (send messages, media)
│   │   └── webhook.py       # Incoming message handler
│   ├── db/
│   │   └── models.py        # SQLite: lead tracking, conversation logging
│   └── static/              # Promotional posters and PDF brochures
├── knowledge_base/           # Course data fed into ChromaDB
│   ├── courses.json
│   ├── bootcamps.json
│   ├── faqs.json
│   └── company_info.json
├── scripts/
│   ├── test_agent.py        # Interactive CLI test mode
│   └── quick_test.py        # Automated quick test
├── docker-compose.yml        # All services: ollama + model-setup + agent-app
├── Dockerfile                # Agent container build definition
├── requirements.txt          # Python dependencies
├── .env                      # Your secrets (never commit this)
└── .env.example              # Template for .env
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{"status":"healthy"}` |
| `GET` | `/webhook` | Meta webhook verification (GET with verify token) |
| `POST` | `/webhook` | Receives incoming WhatsApp messages |
| `GET` | `/api/leads` | All recent leads (last 50) |
| `GET` | `/api/leads/hot` | Only hot leads for sales team |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Qwen 2.5 7B Instruct (via Ollama) |
| Embeddings | nomic-embed-text (via Ollama) |
| Vector DB | ChromaDB (persistent volume) |
| Web Framework | FastAPI + Uvicorn |
| Lead Database | SQLite (persistent volume) |
| WhatsApp Integration | Meta Cloud API |
| Tunnel (dev) | ngrok |
| Containerization | Docker Compose |
| GPU | NVIDIA (Docker GPU passthrough) |

---


