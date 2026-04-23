"""
PHN Technology WhatsApp Agent — Database Models

SQLite database for lead tracking and conversation logging.
Uses raw SQL for simplicity — no heavy ORM needed.
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_db_connection() -> sqlite3.Connection:
    """Get a SQLite database connection."""
    settings = get_settings()
    os.makedirs(os.path.dirname(settings.sqlite_db_path), exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_database():
    """Initialize database tables if they don't exist."""
    conn = get_db_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT UNIQUE NOT NULL,
                name TEXT DEFAULT '',
                lead_score TEXT DEFAULT 'cold',
                interested_courses TEXT DEFAULT '',
                language TEXT DEFAULT 'en',
                first_contact TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_contact TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                direction TEXT NOT NULL,
                message TEXT NOT NULL,
                intent TEXT DEFAULT '',
                response TEXT DEFAULT '',
                lead_score TEXT DEFAULT '',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (phone) REFERENCES leads(phone)
            );

            CREATE INDEX IF NOT EXISTS idx_leads_phone ON leads(phone);
            CREATE INDEX IF NOT EXISTS idx_conversations_phone ON conversations(phone);
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
        """)
        
        # Safely add new columns if they don't exist
        new_columns = [
            ("city", "TEXT DEFAULT ''"),
            ("occupation", "TEXT DEFAULT ''"), 
            ("interest_field", "TEXT DEFAULT ''"),
            ("is_interested", "TEXT DEFAULT ''"),
            ("extracted_name", "TEXT DEFAULT ''"),
            ("follow_up_sent", "INTEGER DEFAULT 0")
        ]
        
        for col_name, col_type in new_columns:
            try:
                conn.execute(f"ALTER TABLE leads ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        logger.info("✅ Database initialized successfully")
    finally:
        conn.close()


def save_lead(phone: str, name: str = "", language: str = "en") -> None:
    """Create or update a lead record."""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO leads (phone, name, language, last_contact, total_messages)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 1)
            ON CONFLICT(phone) DO UPDATE SET
                name = CASE WHEN excluded.name != '' THEN excluded.name ELSE leads.name END,
                language = excluded.language,
                last_contact = CURRENT_TIMESTAMP,
                total_messages = leads.total_messages + 1,
                follow_up_sent = 0
        """, (phone, name, language))
        conn.commit()
    finally:
        conn.close()

def get_inactive_leads(hours_inactive: int = 2) -> list[dict]:
    """Retrieve leads that have been inactive for a certain number of hours and haven't been followed up on."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(f"""
            SELECT phone, name, language 
            FROM leads 
            WHERE follow_up_sent = 0 
            AND is_interested != 'No'
            AND last_contact <= datetime('now', 'localtime', '-{hours_inactive} hours')
        """)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def mark_follow_up_sent(phone: str) -> None:
    """Mark a lead as having received a follow-up."""
    conn = get_db_connection()
    try:
        conn.execute("UPDATE leads SET follow_up_sent = 1 WHERE phone = ?", (phone,))
        conn.commit()
    finally:
        conn.close()


def update_lead_score(phone: str, score: str, interested_courses: str = "") -> None:
    """Update lead score and interested courses."""
    conn = get_db_connection()
    try:
        conn.execute("""
            UPDATE leads SET
                lead_score = ?,
                interested_courses = CASE
                    WHEN ? != '' THEN ?
                    ELSE interested_courses
                END,
                last_contact = CURRENT_TIMESTAMP
            WHERE phone = ?
        """, (score, interested_courses, interested_courses, phone))
        conn.commit()
    finally:
        conn.close()


def update_extracted_info(
    phone: str, 
    extracted_name: str = "", 
    city: str = "", 
    occupation: str = "", 
    interest_field: str = "", 
    is_interested: str = ""
) -> None:
    """Update demographic and interest details extracted by the LLM."""
    conn = get_db_connection()
    try:
        conn.execute("""
            UPDATE leads SET
                extracted_name = CASE WHEN ? != '' AND LOWER(?) != 'unknown' THEN ? ELSE extracted_name END,
                name = CASE WHEN ? != '' AND LOWER(?) != 'unknown' AND name = '' THEN ? ELSE name END,
                city = CASE WHEN ? != '' AND LOWER(?) != 'unknown' THEN ? ELSE city END,
                occupation = CASE WHEN ? != '' AND LOWER(?) != 'unknown' THEN ? ELSE occupation END,
                interest_field = CASE WHEN ? != '' AND LOWER(?) != 'unknown' THEN ? ELSE interest_field END,
                is_interested = CASE WHEN ? != '' AND LOWER(?) != 'unknown' THEN ? ELSE is_interested END,
                last_contact = CURRENT_TIMESTAMP
            WHERE phone = ?
        """, (
            extracted_name, extracted_name, extracted_name,
            extracted_name, extracted_name, extracted_name,
            city, city, city,
            occupation, occupation, occupation,
            interest_field, interest_field, interest_field,
            is_interested, is_interested, is_interested,
            phone
        ))
        conn.commit()
    finally:
        conn.close()


def log_conversation(
    phone: str,
    direction: str,
    message: str,
    intent: str = "",
    response: str = "",
    lead_score: str = ""
) -> None:
    """Log a conversation message."""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO conversations (phone, direction, message, intent, response, lead_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (phone, direction, message, intent, response, lead_score))
        conn.commit()
    finally:
        conn.close()


def get_lead_info(phone: str) -> Optional[dict]:
    """Retrieve lead information by phone number."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM leads WHERE phone = ?", (phone,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_recent_leads(limit: int = 20) -> list[dict]:
    """Get recent leads sorted by last contact."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM leads ORDER BY last_contact DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_hot_leads() -> list[dict]:
    """Get all hot leads for sales team follow-up."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM leads WHERE lead_score = 'hot' ORDER BY last_contact DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
