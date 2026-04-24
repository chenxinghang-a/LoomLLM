"""
REST-like API for AI-Staff V4.

Provides HTTP endpoints for:
  - POST /chat — Send message, get response
  - POST /run   — Execute task with auto-routing
  - GET  /skills — List available skills
  - GET  /status — System health & stats
  - GET  /experts — List expert roles
  - POST /improve — Trigger self-improvement cycle
  
Lightweight: uses Python's built-in http.server (no flask/fastapi dependency).
For production, replace with FastAPI.
"""

from __future__ import annotations

import json
import time
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional, Callable
from urllib.parse import urlparse, parse_qs

# We'll lazily reference the main AIStaff instance


class RestAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for AI-Staff REST API."""
    
    # Set by RestAPIServer before serving
    staff_instance = None
    skill_registry = None
    
    def log_message(self, format, *args):
        """Quiet logging (optional: redirect to our event bus)."""
        pass  # Suppress default stderr logging
    
    def _send_json(self, data: Any, status: int = 200):
        """Send JSON response."""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
    
    def _send_error(self, message: str, status: int = 400, error_type: str = "error"):
        self._send_json({"ok": False, "error": message, "type": error_type}, status)
    
    def _read_body(self) -> dict:
        """Read and parse JSON request body."""
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        params = parse_qs(parsed.query)
        
        routes = {
            '/': self._handle_root,
            '/status': self._handle_status,
            '/skills': self._handle_list_skills,
            '/experts': self._handle_list_experts,
            '/health': self._handle_health,
        }
        
        handler = routes.get(path)
        if handler:
            handler(params)
        else:
            self._send_error(f"Not found: {path}", 404, "not_found")
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        body = self._read_body()
        
        routes = {
            '/chat': self._handle_chat,
            '/run': self._handle_run,
            '/improve': self._handle_improve,
            '/skills/discover': self._handle_discover_skills,
        }
        
        handler = routes.get(path)
        if handler:
            handler(body)
        else:
            self._send_error(f"Not found: {path}", 404, "not_found")
    
    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    # ---- Route Handlers ------------------------------------------------
    
    def _handle_root(self, params):
        """API info endpoint."""
        self._send_json({
            "name": "AI-Staff V4 REST API",
            "version": "4.0.0",
            "endpoints": {
                "GET /": "This info",
                "GET /status": "System status & stats",
                "GET /health": "Health check",
                "GET /skills": "List all skills",
                "GET /experts": "List expert roles",
                "POST /chat": "Send chat message",
                "POST /run": "Run task (auto-routed)",
                "POST /improve": "Trigger self-improvement",
                "POST /skills/discover": "Discover & load new skills",
            }
        })
    
    def _handle_health(self, params):
        """Health check."""
        self._send_json({
            "ok": True,
            "timestamp": time.time(),
            "staff_loaded": self.staff_instance is not None,
        })
    
    def _handle_status(self, params):
        """Detailed status."""
        staff = self.staff_instance
        registry = self.skill_registry
        
        info = {
            "version": "4.0.0",
            "uptime_sec": getattr(self.server, '_start_time', 0) and time.time() - self.server._start_time,
            "staff": {
                "loaded": staff is not None,
                "mode": getattr(staff, 'mode', None) if staff else None,
                "current_backend": getattr(staff, '_current_backend', None) if staff else None,
            },
            "skills": {
                "total": len(registry) if registry else 0,
                "categories": dict(registry.stats().get("categories", {})) if registry else {},
            } if registry else {"total": 0, "error": "No registry"},
        }
        
        if staff and hasattr(staff, 'budget'):
            info["budget"] = staff.budget.summary()
        
        self._send_json(info)
    
    def _handle_list_skills(self, params):
        """List skills with optional filtering."""
        registry = self.skill_registry
        if not registry:
            self._send_error("Skill registry not initialized", 503)
            return
        
        query = params.get('q', [''])[0]
        category = params.get('category', [''])[0]
        
        if query:
            skills = registry.search(query)
        elif category:
            skills = registry.list_by_category(category)
        else:
            skills = registry.all_skills()
        
        result = [{
            "name": s.metadata.name,
            "description": s.metadata.description,
            "category": s.metadata.category,
            "tags": s.metadata.tags,
            "version": s.metadata.version,
            "source": s.source,
        } for s in skills]
        
        self._send_json({"ok": True, "count": len(result), "skills": result})
    
    def _handle_list_experts(self, params):
        """List available expert roles."""
        staff = self.staff_instance
        if not staff or not hasattr(staff, 'expert_registry'):
            # Return empty rather than error
            self._send_json({"ok": True, "experts": []})
            return
        
        experts = []
        for exp_id, exp in staff.expert_registry._experts.items():
            experts.append({
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "domain_tags": exp.domain_tags,
                "tools": exp.tools,
            })
        
        self._send_json({"ok": True, "count": len(experts), "experts": experts})
    
    def _handle_chat(self, body: dict):
        """Chat endpoint — send a message to AI (unified entry)."""
        staff = self.staff_instance
        if not staff:
            self._send_error("AI-Staff not initialized. POST /run or configure first.", 503)
            return
        
        prompt = body.get('prompt') or body.get('message') or body.get('q', '')
        if not prompt:
            self._send_error("Missing 'prompt' field")
            return
        
        mode = body.get('mode', 'auto')  # auto | direct | code | research | decision | creative | collab | arena
        
        try:
            t0 = time.time()
            response = staff.chat(prompt, mode=mode)
            self._send_json({
                "ok": True,
                "response": response,
                "mode": mode,
                "duration_ms": round((time.time() - t0) * 1000),
            })
        except Exception as e:
            self._send_error(str(e), 500, "execution_error")
    
    def _handle_run(self, body: dict):
        """Run task with auto-routing (V5 closed-loop collaboration)."""
        staff = self.staff_instance
        if not staff:
            self._send_error("AI-Staff not initialized.", 503)
            return
        
        prompt = body.get('prompt') or body.get('task', '')
        if not prompt:
            self._send_error("Missing 'prompt' or 'task' field")
            return
        
        try:
            t0 = time.time()
            result = staff.auto_run_v5(
                prompt,
                max_iterations=body.get('max_iterations', 0),
                quality_threshold=body.get('quality_threshold', 80),
            )
            self._send_json({
                "ok": True,
                "status": result.status,
                "quality_score": result.quality_score,
                "deliverables": list(result.deliverables.keys()),
                "mode": result.strategy_mode,
                "iterations": result.rounds_used,
                "duration_ms": round((time.time() - t0) * 1000),
            })
        except Exception as e:
            self._send_error(str(e), 500, "execution_error")
    
    def _handle_improve(self, body: dict):
        """Trigger self-improvement cycle."""
        staff = self.staff_instance
        if not staff or not hasattr(staff, 'self_improve'):
            self._send_error("Self-improvement engine not available", 503)
            return
        
        try:
            target = body.get('target', 'all')  # all | prompts | strategy
            report = staff.self_improve.run_cycle(target)
            self._send_json({"ok": True, "report": report})
        except Exception as e:
            self._send_error(str(e), 500, "improvement_error")
    
    def _handle_discover_skills(self, body: dict):
        """Discover and load new skills from a directory."""
        registry = self.skill_registry
        if not registry:
            self._send_error("Skill registry not initialized", 503)
            return
        
        directory = body.get('directory', '')
        try:
            count = registry.discover_from_directory(directory) if directory else registry.reload_all()
            self._send_json({"ok": True, "new_skills": count, "total": len(registry)})
        except Exception as e:
            self._send_error(str(e), 500)


class RestAPIServer:
    """
    Lightweight HTTP server for AI-Staff V4.
    
    Usage:
        api = RestAPIServer(staff, port=8899)
        api.start()  # Non-blocking
        # ... later ...
        api.stop()
    """
    
    def __init__(self, staff=None, skill_registry=None, port: int = 8899, host: str = "localhost"):
        self.staff = staff
        self.skill_registry = skill_registry
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self, blocking: bool = False) -> None:
        """Start the REST API server."""
        RestAPIHandler.staff_instance = self.staff
        RestAPIHandler.skill_registry = self.skill_registry
        
        self._server = HTTPServer((self.host, self.port), RestAPIHandler)
        self._server._start_time = time.time()
        
        if blocking:
            print(f"[REST] Serving on http://{self.host}:{self.port}/ (Ctrl+C to stop)")
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"[REST] API running on http://{self.host}:{self.port}/")
    
    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            print("[REST] Server stopped")
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def status(self) -> dict:
        """Return server status."""
        return {
            "running": self._server is not None,
            "url": self.url,
            "port": self.port,
        }


# Demo / Test
if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    
    print("=" * 60)
    print("AI-Staff V4 REST API — Self-Test")
    print("=" * 60)
    
    # Start without a real staff instance (test infrastructure only)
    from ..skills.registry import create_builtin_registry
    
    api = RestAPIServer(
        staff=None,
        skill_registry=create_builtin_registry(),
        port=18899,
    )
    
    # Quick test using raw HTTP
    import urllib.request, json
    
    def api_call(method: str, path: str, data=None):
        url = f"http://localhost:{api.port}{path}"
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}
    
    # Start in background
    api.start()
    import time; time.sleep(0.5)
    
    # Test endpoints
    tests = [
        ("GET", "/", None),
        ("GET", "/health", None),
        ("GET", "/skills", None),
        ("GET", "/skills?q=code", None),
        ("POST", "/chat", {"prompt": "hello"}),
    ]
    
    for method, path, data in tests:
        result = api_call(method, path, data)
        ok = result.get("ok", result.get("ok", "N/A" if "error" not in result else "FAIL"))
        print(f"  {method:4s} {path:<20s} -> {ok}")
    
    api.stop()
    print("\nREST API self-test complete!")
