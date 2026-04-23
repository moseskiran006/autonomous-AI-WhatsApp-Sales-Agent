"""
PHN Technology WhatsApp Agent — FastAPI Application

Main entry point for the WhatsApp agent server.
Handles startup initialization and routes.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.whatsapp.webhook import router as webhook_router
from app.db.models import init_database
from app.rag.indexer import index_knowledge_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs initialization on startup and cleanup on shutdown.
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("🚀 PHN Technology WhatsApp Agent — Starting Up")
    logger.info(f"   Environment: {settings.app_env}")
    logger.info(f"   Ollama URL:  {settings.ollama_base_url}")
    logger.info(f"   Chat Model:  {settings.ollama_chat_model}")
    logger.info(f"   Embed Model: {settings.ollama_embed_model}")
    logger.info("=" * 60)

    # Initialize database
    logger.info("📦 Initializing database...")
    init_database()

    # Index knowledge base into ChromaDB
    logger.info("📚 Indexing knowledge base...")
    try:
        index_knowledge_base()
    except Exception as e:
        logger.error(f"⚠️ Knowledge base indexing failed: {e}")
        logger.error("   The agent will start but RAG retrieval may not work.")
        logger.error("   Make sure Ollama is running and models are pulled.")

    logger.info("✅ PHN Technology WhatsApp Agent is ready!")
    logger.info(f"   Webhook URL: http://localhost:8000/webhook")
    logger.info(f"   Health URL:  http://localhost:8000/health")
    logger.info("=" * 60)

    # Start the background re-engagement worker
    from app.db.models import get_inactive_leads, mark_follow_up_sent
    from app.whatsapp.client import send_text_message
    import asyncio

    async def reengagement_worker():
        while True:
            try:
                # Check for leads inactive for 2+ hours
                inactive_leads = get_inactive_leads(hours_inactive=2)
                for lead in inactive_leads:
                    phone = lead["phone"]
                    name = lead["name"] or "there"
                    
                    # Send follow-up message
                    message = f"Hey {name}! Are you still there? Let me know if you have any questions about our tech programs, I'm here to help!"
                    await send_text_message(phone, message)
                    
                    # Mark as followed up
                    mark_follow_up_sent(phone)
                    logger.info(f"🔄 Sent re-engagement message to {phone}")
                    
                    # Prevent rate limiting
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Error in re-engagement worker: {e}")
            
            # Check every 30 minutes
            await asyncio.sleep(1800)

    # Launch worker as background task
    worker_task = asyncio.create_task(reengagement_worker())

    yield  # App is running

    # Shutdown
    worker_task.cancel()
    logger.info("👋 PHN Technology WhatsApp Agent — Shutting Down")


# Create FastAPI app
app = FastAPI(
    title="PHN Technology WhatsApp Agent",
    description="AI-powered WhatsApp agent for PHN Technology Pvt Limited EdTech",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (needed for potential web dashboard later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Include routers
app.include_router(webhook_router)

# Serve static files (like promotional posters/PDFs)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ============================================
# Health Check Endpoint
# ============================================
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    return {
        "status": "healthy",
        "service": "PHN Technology WhatsApp Agent",
        "version": "1.0.0",
    }


# ============================================
# Dashboard Endpoints (for monitoring)
# ============================================
@app.get("/api/leads", tags=["Dashboard"])
async def get_leads():
    """Get recent leads (for monitoring dashboard)."""
    from app.db.models import get_recent_leads
    return {"leads": get_recent_leads(limit=50)}


@app.get("/api/leads/hot", tags=["Dashboard"])
async def get_hot_leads():
    """Get hot leads for sales follow-up."""
    from app.db.models import get_hot_leads
    return {"hot_leads": get_hot_leads()}
