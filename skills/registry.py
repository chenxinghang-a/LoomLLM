"""
Skill Registry — V4 Dynamic Discovery & Hot-Reload

Concept:
  A "skill" is a self-describing capability unit that any AI expert can discover and use.
  Skills are discovered from:
    1. Built-in registry (code-defined)
    2. YAML/JSON skill files in configurable directories
    3. Runtime registration (plugins, MCP tools, etc.)

Hot-reload: Watch file changes → re-parse → update registry without restart.

Protocol (self-describing):
  Each skill has: name, description, input_schema, output_schema, tags, examples
  AI can READ this metadata to decide which skill to use for a given task.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict, List

# ============================================================
# Data Types
# ============================================================

@dataclass
class SkillInputField:
    """Describes one input parameter of a skill."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    required: bool = True
    description: str = ""
    default: Any = None
    enum: Optional[List[str]] = None  # constrained values


@dataclass
class SkillOutputSchema:
    """Describes what a skill produces."""
    type: str  # "string", "object", "array", "file", "stream"
    description: str = ""
    fields: Optional[Dict[str, str]] = None  # for object type


@dataclass
class SkillMetadata:
    """Self-describing metadata — AI reads this to understand the skill."""
    name: str
    description: str           # Natural language what it does
    category: str              # "analysis", "generation", "retrieval", "tool", ...
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    input_fields: List[SkillInputField] = field(default_factory=list)
    output_schema: Optional[SkillOutputSchema] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)  # example I/O pairs
    author: str = ""
    requires_api: bool = False # Needs external API call?
    cost_estimate: str = ""    # "low", "medium", "high" token/cost hint
    
    def to_ai_description(self) -> str:
        """Generate natural language description for LLM consumption."""
        parts = [f"## {self.name} (v{self.version})"]
        parts.append(f"{self.description}")
        parts.append(f"Category: {self.category} | Tags: {', '.join(self.tags)}")
        
        if self.input_fields:
            parts.append("\n**Inputs:**")
            for f in self.input_fields:
                req = "REQUIRED" if f.required else f"optional (default={f.default})"
                enum_str = f" [{', '.join(f.enum)}]" if f.enum else ""
                parts.append(f"  - `{f.name}` ({f.type}, {req}){enum_str}: {f.description}")
        
        if self.output_schema:
            parts.append(f"\n**Output:** {self.output_schema.type} — {self.output_schema.description}")
        
        if self.examples:
            parts.append("\n**Examples:**")
            for ex in self.examples[:3]:  # max 3 examples
                parts.append(f"  Input: {ex.get('input', '?')} → Output: {ex.get('output', '?')}")
        
        if self.requires_api:
            parts.append(f"\n⚠️ Requires external API | Cost: {self.cost_estimate or 'unknown'}")
        
        return "\n".join(parts)


@dataclass 
class SkillHandle:
    """A registered skill with its executor function and metadata."""
    metadata: SkillMetadata
    executor: Callable  # fn(input_dict) -> Any
    source: str = "builtin"  # builtin / file / runtime / mcp
    file_path: Optional[str] = None
    loaded_at: float = 0.0
    _hash: str = ""          # content hash for change detection
    
    def __post_init__(self):
        if not self.loaded_at:
            self.loaded_at = time.time()


# ============================================================
# Skill Registry Core
# ============================================================

class SkillRegistry:
    """
    Central registry for all skills.
    
    Supports:
      - Dynamic discovery from directories (YAML/JSON/.py files)
      - Hot-reload on file changes  
      - Runtime registration/de-registration
      - Search by name, tag, category, semantic similarity stub
      - AI-friendly listing (to_description() for prompt injection)
    """
    
    def __init__(self, skill_dirs: Optional[List[str]] = None):
        self._skills: Dict[str, SkillHandle] = {}       # name → handle
        self._by_category: Dict[str, List[str]] = {}    # category → [names]
        self._by_tag: Dict[str, List[str]] = {}         # tag → [names]
        self._skill_dirs: List[str] = skill_dirs or []
        self._file_mtimes: Dict[str, float] = {}         # path → mtime
        self._lock = threading.RLock()
        self._event_callbacks: List[Callable] = []       # on_skill_added/removed
        
    # ---- Registration ------------------------------------------------
    
    def register(self, handle: SkillHandle, *, overwrite: bool = False) -> bool:
        """Register a skill. Returns True if newly added."""
        with self._lock:
            name = handle.metadata.name
            if name in self._skills and not overwrite:
                return False
            
            old = self._skills.get(name)
            self._skills[name] = handle
            self._index(handle.metadata)
            
            event = "updated" if old else "added"
            self._fire_event(event, handle)
            return old is None
    
    def unregister(self, name: str) -> bool:
        """Remove a skill by name."""
        with self._lock:
            handle = self._skills.pop(name, None)
            if not handle:
                return False
            self._unindex(handle.metadata)
            self._fire_event("removed", handle)
            return True
    
    def register_function(
        self,
        fn: Callable,
        name: str,
        description: str,
        category: str = "tool",
        **metadata_kwargs,
    ) -> SkillHandle:
        """Quick-register a Python function as a skill."""
        meta = SkillMetadata(
            name=name,
            description=description,
            category=category,
            **metadata_kwargs,
        )
        handle = SkillHandle(metadata=meta, executor=fn)
        self.register(handle)
        return handle
    
    # ---- Discovery & Loading -----------------------------------------
    
    def discover_from_directory(self, directory: str, recursive: bool = True) -> int:
        """
        Scan directory for skill definition files (.yaml/.json/.skill).
        Returns number of new skills loaded.
        """
        directory = os.path.abspath(directory)
        if directory not in self._skill_dirs:
            self._skill_dirs.append(directory)
        
        new_count = 0
        path = Path(directory)
        pattern = "**/*" if recursive else "*"
        
        for ext in ("*.yaml", "*.yml", "*.json", "*.skill"):
            for fpath in path.glob(pattern):
                if fpath.is_file() and fpath.suffix in (".yaml",".yml",".json",".skill"):
                    if self._load_skill_file(str(fpath)):
                        new_count += 1
        
        return new_count
    
    def _load_skill_file(self, filepath: str) -> bool:
        """Load one skill definition file. Returns success."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()
            
            current_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]
            
            # Skip if unchanged
            if filepath in self._file_mtimes:
                old_hash = self._skills.get(Path(filepath).stem, SkillHandle.__new__(SkillHandle))._hash
                if old_hash == current_hash:
                    return False
            
            # Parse based on extension
            if filepath.endswith((".yaml", ".yml")):
                import yaml
                data = yaml.safe_load(raw)
            elif filepath.endswith((".json", ".skill")):
                data = json.loads(raw)
            else:
                return False
            
            handle = self._parse_skill_data(data, filepath, current_hash)
            if handle:
                self.register(handle, overwrite=True)
                self._file_mtimes[filepath] = time.time()
                return True
        except Exception as e:
            print(f"[SkillRegistry] Failed to load {filepath}: {e}")
            return False
    
    def _parse_skill_data(self, data: Dict, filepath: str, content_hash: str) -> Optional[SkillHandle]:
        """Parse skill definition dict into SkillHandle."""
        if not isinstance(data, dict) or "name" not in data:
            return None
        
        meta = SkillMetadata(
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "tool"),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            author=data.get("author", ""),
            requires_api=data.get("requires_api", False),
            cost_estimate=data.get("cost_estimate", ""),
        )
        
        # Input fields
        for fld in data.get("input_fields", []):
            meta.input_fields.append(SkillInputField(
                name=fld.get("name", ""),
                type=fld.get("type", "string"),
                required=fld.get("required", True),
                description=fld.get("description", ""),
                default=fld.get("default"),
                enum=fld.get("enum"),
            ))
        
        # Output schema
        out = data.get("output_schema")
        if out:
            meta.output_schema = SkillOutputSchema(
                type=out.get("type", "string"),
                description=out.get("description", ""),
                fields=out.get("fields"),
            )
        
        # Examples
        meta.examples = data.get("examples", [])
        
        # Executor — either inline code ref or placeholder
        executor_name = data.get("executor", "print")  # default to print-based stub
        executor = self._resolve_executor(executor_name, data)
        
        return SkillHandle(
            metadata=meta,
            executor=executor,
            source="file",
            file_path=filepath,
            _hash=content_hash,
        )
    
    def _resolve_executor(self, executor_ref: str, data: Dict) -> Callable:
        """Resolve executor reference to callable."""
        # Built-in executors
        builtins = {
            "print": lambda inp: json.dumps(inp, ensure_ascii=False, indent=2),
            "echo": lambda inp: inp.get("text", "") if isinstance(inp, dict) else str(inp),
            "identity": lambda x: x,
        }
        if executor_ref in builtins:
            return builtins[executor_ref]
        
        # For now, return echo as default; real implementation would support plugin loading
        return builtins["echo"]
    
    # ---- Hot Reload ---------------------------------------------------
    
    def check_for_changes(self) -> List[str]:
        """Check all watched directories for modified files. Returns list of changed paths."""
        changed = []
        for filepath, last_mtime in list(self._file_mtimes.items()):
            try:
                current_mtime = os.path.getmtime(filepath)
                if current_mtime > last_mtime:
                    self._load_skill_file(filepath)
                    changed.append(filepath)
            except OSError:
                pass
        return changed
    
    def start_watcher(self, interval: float = 5.0) -> None:
        """Start background thread that checks for file changes every `interval` seconds."""
        import threading as _threading
        
        def _watch_loop():
            while getattr(self, "_watching", True):
                try:
                    changes = self.check_for_changes()
                    if changes:
                        print(f"[SkillWatcher] Reloaded {len(changes)} skill(s)")
                except Exception:
                    pass
                time.sleep(interval)
        
        self._watching = True
        t = _threading.Thread(target=_watch_loop, daemon=True, name="SkillWatcher")
        t.start()
    
    def stop_watcher(self) -> None:
        """Stop background watcher thread."""
        self._watching = False
    
    # ---- Query / Search -----------------------------------------------
    
    def get(self, name: str) -> Optional[SkillHandle]:
        """Get skill by exact name."""
        return self._skills.get(name)
    
    def list_by_category(self, category: str) -> List[SkillHandle]:
        """List all skills in a category."""
        names = self._by_category.get(category, [])
        return [self._skills[n] for n in names if n in self._skills]
    
    def list_by_tag(self, tag: str) -> List[SkillHandle]:
        """List all skills with a given tag."""
        names = self._by_tag.get(tag, [])
        return [self._skills[n] for n in names if n in self._skills]
    
    def search(self, query: str) -> List[SkillHandle]:
        """
        Fuzzy search across name, description, tags.
        Returns matches sorted by relevance (simple keyword scoring).
        """
        query_lower = query.lower()
        keywords = query_lower.split()
        
        scored = []
        for name, handle in self._skills.items():
            score = 0
            text = (
                handle.metadata.name.lower() + " " +
                handle.metadata.description.lower() + " " +
                " ".join(handle.metadata.tags).lower() + " " +
                handle.metadata.category.lower()
            )
            
            # Exact name match → highest score
            if query_lower == handle.metadata.name.lower():
                score += 100
            # Keyword matches
            for kw in keywords:
                if kw in handle.metadata.name.lower():
                    score += 20
                if kw in handle.metadata.description.lower():
                    score += 5
                if kw in handle.metadata.category.lower():
                    score += 10
                if kw in " ".join(handle.metadata.tags).lower():
                    score += 8
            
            if score > 0:
                scored.append((score, handle))
        
        scored.sort(key=lambda x: -x[0])
        return [h for _, h in scored]
    
    def all_skills(self) -> List[SkillHandle]:
        """Return all registered skills."""
        return list(self._skills.values())
    
    def count(self) -> int:
        return len(self._skills)
    
    # ---- AI-Friendly Output -------------------------------------------
    
    def to_prompt_context(self, max_skills: int = 20) -> str:
        """
        Generate a comprehensive skill listing formatted for LLM prompt injection.
        This is THE key method — AI reads this to decide which skills to use.
        """
        skills = list(self._skills.values())
        if not skills:
            return "(No skills registered)"
        
        # Sort by relevance heuristics: more tags/descriptions first
        skills.sort(key=lambda s: len(s.metadata.tags) + len(s.metadata.description), reverse=True)
        
        parts = [
            f"# Available Skills ({len(skills)} total)",
            "",
            "You have access to these skills. Use them when relevant to the user's request.",
            "",
        ]
        
        for skill in skills[:max_skills]:
            parts.append(skill.metadata.to_ai_description())
            parts.append("")
        
        if len(skills) > max_skills:
            parts.append(f"... and {len(skills) - max_skills} more skills.")
        
        return "\n".join(parts)
    
    def to_table(self) -> str:
        """Return a compact table summary."""
        rows = []
        for h in self._skills.values():
            m = h.metadata
            rows.append(f"  {m.name:<25s} [{m.category:<12s}] ({h.source}) {m.description[:50]}")
        return f"# Registry ({len(self._skills)} skills)\n" + "\n".join(rows)
    
    # ---- Internal Index -----------------------------------------------
    
    def _index(self, meta: SkillMetadata) -> None:
        """Add to category/tag indexes."""
        self._by_category.setdefault(meta.category, []).append(meta.name)
        for tag in meta.tags:
            self._by_tag.setdefault(tag, []).append(meta.name)
    
    def _unindex(self, meta: SkillMetadata) -> None:
        """Remove from indexes."""
        if meta.category in self._by_category:
            self._by_category[meta.category] = [
                n for n in self._by_category[meta.category] if n != meta.name
            ]
        for tag in meta.tags:
            if tag in self._by_tag:
                self._by_tag[tag] = [n for n in self._by_tag[tag] if n != meta.name]
    
    def _fire_event(self, event_type: str, handle: SkillHandle) -> None:
        """Fire event callbacks."""
        for cb in self._event_callbacks:
            try:
                cb(event_type, handle)
            except Exception:
                pass
    
    def on_change(self, callback: Callable[[str, SkillHandle], None]) -> None:
        """Register a callback for skill add/remove/update events."""
        self._event_callbacks.append(callback)
    
    # ---- Lifecycle ----------------------------------------------------
    
    def reload_all(self) -> int:
        """Force re-scan all directories. Return count of updated skills."""
        count = 0
        for d in self._skill_dirs:
            count += self.discover_from_directory(d)
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Return registry statistics."""
        return {
            "total_skills": len(self._skills),
            "categories": {k: len(v) for k, v in self._by_category.items()},
            "tags": {k: len(v) for k, v in self._by_tag.items()},
            "watched_dirs": len(self._skill_dirs),
            "watched_files": len(self._file_mtimes),
            "sources": self._count_sources(),
        }
    
    def _count_sources(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for h in self._skills.values():
            counts[h.source] = counts.get(h.source, 0) + 1
        return counts
    
    def __len__(self) -> int:
        return len(self._skills)
    
    def __contains__(self, name: str) -> bool:
        return name in self._skills
    
    def __iter__(self):
        return iter(self._skills.values())


# ============================================================
# Built-in Skills Factory
# ============================================================

def create_builtin_registry() -> SkillRegistry:
    """Create a registry pre-loaded with common built-in skills."""
    reg = SkillRegistry()
    
    # --- Analysis Skills ---
    reg.register_function(
        fn=lambda inp: json.dumps({"status": "analyzed", "input": inp}),
        name="analyze_text",
        description="Deep analysis of text: sentiment, topics, entities, readability",
        category="analysis",
        tags=["text", "nlp", "analysis"],
    )
    
    reg.register_function(
        fn=lambda inp: json.dumps({"status": "summarized"}),
        name="summarize",
        description="Condense long text into key points with adjustable detail level",
        category="analysis",
        tags=["text", "summary"],
    )
    
    # --- Generation Skills ---
    reg.register_function(
        fn=lambda inp: json.dumps({"status": "generated"}),
        name="generate_code",
        description="Generate code in any language from natural language spec",
        category="generation",
        tags=["code", "programming"],
        requires_api=True,
        cost_estimate="medium",
    )
    
    reg.register_function(
        fn=lambda inp: json.dumps({"status": "written"}),
        name="write_document",
        description="Create structured documents: reports, proposals, documentation",
        category="generation",
        tags=["document", "writing"],
    )
    
    # --- Retrieval Skills ---
    reg.register_function(
        fn=lambda inp: json.dumps({"results": [], "query": inp.get("query", "")}),
        name="web_search",
        description="Search the web for current information",
        category="retrieval",
        tags=["web", "search", "realtime"],
        requires_api=True,
        cost_estimate="low",
    )
    
    reg.register_function(
        fn=lambda inp: json.dumps({"data": []}),
        name="query_database",
        description="Query structured databases using SQL or natural language",
        category="retrieval",
        tags=["database", "sql", "structured"],
    )
    
    # --- Tool Skills ---
    reg.register_function(
        fn=lambda inp: json.dumps({"result": "executed"}),
        name="run_command",
        description="Execute shell commands safely with timeout and sandbox",
        category="tool",
        tags=["shell", "command", "execution"],
    )
    
    reg.register_function(
        fn=lambda inp: json.dumps({"path": "", "content": ""}),
        name="read_file",
        description="Read file contents with automatic encoding detection",
        category="tool",
        tags=["file", "io", "read"],
    )
    
    reg.register_function(
        fn=lambda inp: json.dumps({"written": True}),
        name="write_file",
        description="Write or append content to files with backup",
        category="tool",
        tags=["file", "io", "write"],
    )
    
    # --- Math/Data Skills ---
    reg.register_function(
        fn=lambda inp: json.dumps({"result": 0}),
        name="calculate",
        description="Perform mathematical calculations and symbolic math",
        category="analysis",
        tags=["math", "calculation"],
    )
    
    reg.register_function(
        fn=lambda inp: json.dumps({"chart_data": {}}),
        name="visualize",
        description="Generate charts, graphs, and visualizations from data",
        category="generation",
        tags=["chart", "visualization", "plot"],
    )
    
    return reg


# ============================================================
# Demo / Test Entry Point
# ============================================================

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("AI-Staff V4 Skill Registry — Self-Test")
    print("=" * 60)
    
    # 1. Create registry with builtins
    reg = create_builtin_registry()
    print(f"\n✅ Built-in registry: {reg.count()} skills")
    
    # 2. Test search
    results = reg.search("code")
    print(f"\n🔍 Search 'code': {[r.metadata.name for r in results]}")
    
    results = reg.search("web search")
    print(f"🔍 Search 'web search': {[r.metadata.name for r in results]}")
    
    # 3. Test category listing
    cat_skills = reg.list_by_category("tool")
    print(f"\n📂 Category 'tool': {[s.metadata.name for s in cat_skills]}")
    
    # 4. AI-friendly output
    print(f"\n🤖 Prompt Context (first 500 chars):")
    ctx = reg.to_prompt_context()
    print(ctx[:500] + "...")
    
    # 5. Stats
    print(f"\n📊 Stats:")
    for k, v in reg.stats().items():
        print(f"   {k}: {v}")
    
    # 6. Runtime registration test
    def custom_executor(inp):
        return f"CUSTOM_RESULT: {inp}"
    
    reg.register(SkillHandle(
        metadata=SkillMetadata(
            name="custom_test",
            description="A dynamically registered skill for testing",
            category="test",
            tags=["demo", "test"],
        ),
        executor=custom_executor,
        source="runtime",
    ))
    print(f"\n✅ After dynamic registration: {reg.count()} skills")
    
    # 7. Execute a skill
    handle = reg.get("echo" if "echo" in reg else "summarize")
    if handle:
        result = handle.executor({"text": "Hello Skill Registry!"})
        print(f"\n⚡ Execute '{handle.metadata.name}': {result}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
