from __future__ import annotations
import sys, os, json, time, io, re, hashlib
from collections import deque
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import sqlite3

# Intra-package imports
from .constants import MEMORY_DB, VERSION
from .events import EventBus, Event, EventType, bus

class MemorySystem:
    """
    Persistent memory with SQLite backend.
    Stores: conversation summaries, user preferences, task history.
    Implements semantic preference extraction from conversations.
    """
    
    def __init__(self, db_path: Path = MEMORY_DB):
        self.db_path = db_path
        self._conn = None
        self._init_db()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def close(self):
        """Explicitly close SQLite connection."""
        if self._conn:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None
    
    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                expert_id TEXT,
                tokens INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
            
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                key_points TEXT,       -- JSON array
                turn_range TEXT,       -- e.g., "1-5"
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL UNIQUE,
                user_value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                source_session TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT,
                prompt TEXT,
                result_path TEXT,
                status TEXT,
                tokens_used INTEGER DEFAULT 0,
                duration_sec REAL,
                models_used TEXT,      -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message_index INTEGER,
                rating INTEGER,         -- 1=bad, 2=meh, 3=good
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    
    def save_message(self, session_id: str, role: str, content: str,
                     model: str = "", expert_id: str = "", tokens: int = 0):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, model, expert_id, tokens) VALUES (?,?,?,?,?,?)",
            (session_id, role, content[:50000], model, expert_id, tokens)
        )
        conn.commit()
    
    def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    
    def save_summary(self, session_id: str, summary: str, key_points: list, turn_range: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO summaries (session_id, summary_text, key_points, turn_range) VALUES (?,?,?,?)",
            (session_id, summary, json.dumps(key_points, ensure_ascii=False), turn_range)
        )
        conn.commit()
        bus.publish(Event(EventType.AGENT_MEMORY, {"session": session_id, "action": "summary_saved"}, source="Memory"))
    
    def get_summaries(self, session_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM summaries WHERE session_id=? ORDER BY id", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    
    def learn_preference(self, key: str, value: str, confidence: float = 0.5,
                         source: str = ""):
        """Extract and store user preference from conversation context."""
        conn = self._get_conn()
        existing = conn.execute("SELECT id, confidence FROM preferences WHERE user_key=?", (key,)).fetchone()
        if existing:
            # Increase confidence over time
            new_conf = min(1.0, existing["confidence"] + confidence * 0.1)
            conn.execute(
                "UPDATE preferences SET user_value=?, confidence=?, source_session=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (value, new_conf, source, existing["id"])
            )
        else:
            conn.execute(
                "INSERT INTO preferences (user_key, user_value, confidence, source_session) VALUES (?,?,?,?)",
                (key, value, confidence, source)
            )
        conn.commit()
    
    def get_preferences(self, query: str = "", top_k: int = 5) -> list[dict]:
        """Get relevant user preferences, optionally filtered by keyword."""
        conn = self._get_conn()
        if query:
            rows = conn.execute(
                "SELECT * FROM preferences WHERE user_key LIKE ? OR user_value LIKE ? ORDER BY confidence DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", top_k)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM preferences ORDER BY confidence DESC LIMIT ?", (top_k,)
            ).fetchall()
        return [dict(r) for r in rows]
    
    def build_context_header(self, session_id: str) -> str:
        """Build contextualized system prompt with learned preferences."""
        prefs = self.get_preferences(top_k=3)
        summaries = self.get_summaries(session_id)
        
        parts = []
        if prefs:
            pref_lines = [f"- {p['user_key']}: {p['user_value']} (置信度:{p['confidence']:.0%})" for p in prefs]
            parts.append(f"【用户偏好（已学习）】\n" + "\n".join(pref_lines))
        if summaries:
            latest = summaries[-1]
            parts.append(f"【对话摘要】\n{latest['summary_text']}")
        
        return "\n\n".join(parts) if parts else ""
    
    def log_task(self, task_type: str, prompt: str, result_path: str,
                 status: str, tokens: int, duration: float, models: list):
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO task_history 
               (task_type, prompt, result_path, status, tokens_used, duration_sec, models_used)
               VALUES (?,?,?,?,?,?,?)""",
            (task_type, prompt[:500], result_path, status, tokens, duration,
             json.dumps(models, ensure_ascii=False))
        )
        conn.commit()
    
    def save_feedback(self, session_id: str, msg_idx: int, rating: int, comment: str = ""):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO feedback (session_id, message_index, rating, comment) VALUES (?,?,?,?)",
            (session_id, msg_idx, rating, comment)
        )
        conn.commit()
        bus.publish(Event(EventType.USER_FEEDBACK, {"rating": rating, "comment": comment}, source="Memory"))
    
    def extract_preferences_from_chat(self, messages: list[dict], session_id: str):
        """Analyze conversation to extract implicit preferences."""
        # Simple heuristic extraction (in production, use LLM for this)
        patterns = [
            (r"(?:总是|通常|喜欢|偏好|希望|要用|应该)(.+?)[，。！？]", "preference"),
            (r"不要|别|禁止|避免|不用(.+?)[，。！？]", "negative_preference"),
            (r"(?:我是|我的|我叫)(.{2,10})", "self_info"),
        ]
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        
        for pattern, ptype in patterns:
            matches = re.findall(pattern, last_user_msg)
            for match in matches:
                clean = match.strip()
                if len(clean) > 2:
                    self.learn_preference(
                        f"{ptype}_{hashlib.md5(clean.encode()).hexdigest()[:8]}",
                        clean, confidence=0.6, source=session_id
                    )




