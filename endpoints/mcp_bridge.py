"""
MCP Bridge for AI-Staff V4.

Provides Model Context Protocol (MCP) compatibility layer.
Allows MCP-compatible clients to access AI-Staff capabilities as tools/resources.

This is a lightweight implementation following the MCP spec.
For production use with Claude Desktop or similar, consider using the official MCP SDK.

MCP Concepts mapped to AI-Staff:
  - tools → Skills + Expert agents
  - resources → Memory entries, expert configs, task history
  - prompts → Task templates, system prompts
"""

from __future__ import annotations

import json
import time
import uuid
import traceback
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass, field


# ============================================================
# MCP Protocol Types
# ============================================================

@dataclass 
class MCPTool:
    """MCP Tool description."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass  
class MCPResource:
    """MCP Resource descriptor."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "application/json"
    
    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPResult:
    """Result of an MCP tool call."""
    content: List[Dict]  # [{type: "text", text: "..."}]
    is_error: bool = False
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "isError": self.is_error,
        }


@dataclass
class MCPPromptMessage:
    """MCP prompt message."""
    role: str  # user | assistant
    content: str
    
    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class MCPPromptTemplate:
    """MCP prompt template."""
    name: str
    description: str = ""
    arguments: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


# ============================================================
# MCP Bridge — Maps AI-Staff internals to MCP protocol
# ============================================================

class MCPBridge:
    """
    Bridge between AI-Staff V4 and MCP protocol.
    
    Converts:
      - SkillRegistry → MCP tools
      - MemorySystem → MCP resources
      - ExpertRegistry → MCP tools (per-expert)
      - AIStaff methods → callable tools
    
    Usage:
        bridge = MCPBridge(staff=staff, registry=skill_reg)
        
        # Get all tools for MCP client
        tools = bridge.list_tools()
        
        # Call a tool
        result = bridge.call_tool("chat", {"prompt": "Hello"})
        
        # Get resources  
        resources = bridge.list_resources()
        content = bridge.read_resource("memory://preferences")
    """
    
    def __init__(self, staff=None, skill_registry=None, memory_system=None):
        self.staff = staff
        self.skill_registry = skill_registry
        self.memory_system = memory_system
        self._custom_tools: Dict[str, Callable] = {}
        self._session_id = str(uuid.uuid4())[:8]
    
    # ---- Tools --------------------------------------------------------
    
    def list_tools(self) -> List[MCPTool]:
        """List all available tools (skills + built-in)."""
        tools = []
        
        # Core AI-Staff tools (always available)
        tools.extend(self._get_core_tools())
        
        # Skill registry tools
        if self.skill_registry:
            for skill in self.skill_registry.all_skills():
                tools.append(self._skill_to_mcp_tool(skill))
        
        # Custom registered tools
        for name in self._custom_tools:
            tools.append(MCPTool(
                name=name,
                description=f"Custom tool: {name}",
                input_schema={"type": "object", "properties": {}},
            ))
        
        return tools
    
    def _get_core_tools(self) -> List[MCPTool]:
        """Define core AI-Staff tools."""
        return [
            MCPTool(
                name="chat",
                description="Send a message to AI and get a response",
                input_schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "User message"},
                        "mode": {"type": "string", "enum": ["direct","code","research","decision"], "description": "Execution mode"},
                        "model": {"type": "string", "description": "Specific model override"},
                    },
                    "required": ["prompt"],
                },
            ),
            MCPTool(
                name="auto_run",
                description="Smart execution: auto-detects task type and routes optimally",
                input_schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Task description"},
                        "options": {"type": "array", "items": {"type": "string"}, "description": "Options for decision tasks"},
                    },
                    "required": ["prompt"],
                },
            ),
            MCPTool(
                name="code",
                description="Generate, debug, or review code with AI assistance",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Code task description"},
                        "language": {"type": "string", "description": "Programming language"},
                    },
                    "required": ["task"],
                },
            ),
            MCPTool(
                name="research",
                description="Deep research on a topic using multi-step agent workflow",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Research question"},
                        "depth": {"type": "string", "enum": ["quick","standard","deep"], "default": "standard"},
                    },
                    "required": ["query"],
                },
            ),
            MCPTool(
                name="decision",
                description="Get AI analysis and recommendation for a decision",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Decision question"},
                        "options": {"type": "array", "items": {"type": "string"}, "description": "Options to choose from"},
                        "context": {"type": "string", "description": "Additional context"},
                    },
                    "required": ["question", "options"],
                },
            ),
            MCPTool(
                name="improve",
                description="Trigger AI self-improvement cycle to enhance performance",
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "enum": ["all","prompts","strategy"], "default": "all"},
                    },
                },
            ),
            MCPTool(
                name="list_skills",
                description="List all available skills and their descriptions",
                input_schema={"type": "object", "properties": {}},
            ),
            MCPTool(
                name="status",
                description="Get current system status and statistics",
                input_schema={"type": "object", "properties": {}},
            ),
        ]
    
    def _skill_to_mcp_tool(self, skill) -> MCPTool:
        """Convert a SkillHandle to MCPTool."""
        props = {}
        required = []
        for fld in skill.metadata.input_fields:
            prop_def: Dict[str, Any] = {"type": fld.type, "description": fld.description}
            if fld.enum:
                prop_def["enum"] = fld.enum
            elif fld.default is not None:
                prop_def["default"] = fld.default
            if fld.required:
                required.append(fld.name)
            props[fld.name] = prop_def
        
        return MCPTool(
            name=f"skill_{skill.metadata.name}",
            description=skill.metadata.description,
            input_schema={"type": "object", "properties": props, "required": required},
        )
    
    # ---- Tool Execution ------------------------------------------------
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Execute a tool by name with given arguments."""
        try:
            # Custom tool?
            if name in self._custom_tools:
                result = self._custom_tools[name](**arguments)
                return MCPResult(content=[{"type": "text", "text": json.dumps(result, ensure_ascii=False)}])
            
            # Core tools
            core_handlers = {
                "chat": self._tool_chat,
                "auto_run": self._tool_auto_run,
                "code": self._tool_code,
                "research": self._tool_research,
                "decision": self._tool_decision,
                "improve": self._tool_improve,
                "list_skills": self._tool_list_skills,
                "status": self._tool_status,
            }
            
            if name in core_handlers:
                result = core_handlers[name](arguments)
                return MCPResult(content=[{"type": "text", "text": result}])
            
            # Skill-prefixed tool?
            if name.startswith("skill_"):
                skill_name = name[6:]  # strip "skill_" prefix
                return self._call_skill_tool(skill_name, arguments)
            
            return MCPResult(
                content=[{"type": "text", "text": f"Unknown tool: {name}"}],
                is_error=True,
            )
            
        except Exception as e:
            return MCPResult(
                content=[{"type": "text", "text": f"Error executing {name}: {traceback.format_exc()}"}],
                is_error=True,
            )
    
    def _tool_chat(self, args: dict) -> str:
        if not self.staff:
            return "Error: AI-Staff not initialized"
        prompt = args.get("prompt", "")
        mode = args.get("mode", "direct")
        model = args.get("model", "")
        result = self.staff.chat(prompt, model=model) if mode == "direct" else self.staff.auto_run(prompt)
        return str(result)
    
    def _tool_auto_run(self, args: dict) -> str:
        if not self.staff:
            return "Error: AI-Staff not initialized"
        prompt = args.get("prompt", "")
        result = self.staff.auto_run(prompt)
        return str(result)
    
    def _tool_code(self, args: dict) -> str:
        if not self.staff:
            return "Error: AI-Staff not initialized"
        task = args.get("task", args.get("prompt", ""))
        result = self.staff.code(task)
        return str(result)
    
    def _tool_research(self, args: dict) -> str:
        if not self.staff:
            return "Error: AI-Staff not initialized"
        query = args.get("query", args.get("prompt", ""))
        result = self.staff.research(query)
        return str(result)
    
    def _tool_decision(self, args: dict) -> str:
        if not self.staff:
            return "Error: AI-Staff not initialized"
        question = args.get("question", args.get("prompt", ""))
        options = args.get("options", [])
        result = self.staff.decision(question, options=options)
        return str(result)
    
    def _tool_improve(self, args: dict) -> str:
        target = args.get("target", "all")
        if self.staff and hasattr(self.staff, 'self_improve'):
            report = self.staff.self_improve.run_cycle(target)
            return json.dumps(report, ensure_ascii=False, indent=2)
        return "Self-improvement not available"
    
    def _tool_list_skills(self, args: dict) -> str:
        if self.skill_registry:
            return self.skill_registry.to_table()
        return "No skill registry"
    
    def _tool_status(self, args: dict) -> dict:
        info = {
            "version": "4.0.0",
            "session_id": self._session_id,
            "tools_count": len(self.list_tools()),
            "staff_loaded": self.staff is not None,
            "skills_count": len(self.skill_registry) if self.skill_registry else 0,
        }
        if self.staff and hasattr(self.staff, 'budget'):
            info["budget"] = self.staff.budget.summary()
        return json.dumps(info, ensure_ascii=False, indent=2)
    
    def _call_skill_tool(self, skill_name: str, args: dict) -> MCPResult:
        """Execute a skill from the registry."""
        if not self.skill_registry:
            return MCPResult(content=[{"type": "text", "text": "No skill registry"}], is_error=True)
        
        handle = self.skill_registry.get(skill_name)
        if not handle:
            return MCPResult(content=[{"type": "text", "text": f"Skill '{skill_name}' not found"}], is_error=True)
        
        try:
            result = handle.executor(args)
            return MCPResult(content=[{"type": "text", "text": json.dumps(result, ensure_ascii=False)}])
        except Exception as e:
            return MCPResult(content=[{"type": "text", "text": f"Skill error: {e}"}], is_error=True)
    
    # ---- Resources -----------------------------------------------------
    
    def list_resources(self) -> List[MCPResource]:
        """List available resources."""
        resources = []
        
        # Memory-based resources
        if self.memory_system:
            resources.extend([
                MCPResource(uri="memory://preferences", name="User Preferences", description="Learned user preferences"),
                MCPResource(uri="memory://summaries", name="Conversation Summaries", description="Past conversation summaries"),
                MCPResource(uri="memory://history", name="Task History", description="Completed tasks log"),
            ])
        
        # Expert configs as resources
        if self.staff and hasattr(self.staff, 'expert_registry'):
            for exp_id in self.staff.expert_registry._experts:
                resources.append(MCPResource(
                    uri=f"expert://{exp_id}",
                    name=f"Expert: {exp_id}",
                    description="Expert role configuration",
                ))
        
        return resources
    
    def read_resource(self, uri: str) -> MCPResult:
        """Read a resource by URI."""
        try:
            if uri.startswith("memory://"):
                return self._read_memory_resource(uri)
            elif uri.startswith("expert://"):
                return self._read_expert_resource(uri)
            else:
                return MCPResult(content=[{"type": "text", "text": f"Unknown resource: {uri}"}], is_error=True)
        except Exception as e:
            return MCPResult(content=[{"type": "text", "text": f"Resource error: {e}"}], is_error=True)
    
    def _read_memory_resource(self, uri: str) -> MCPResult:
        if not self.memory_system:
            return MCPResult(content=[{"type": "text", "text": "Memory not available"}], is_error=True)
        
        path = uri[len("memory://"):]
        if path == "preferences":
            prefs = self.memory_system.get_preferences(top_k=20)
            return MCPResult(content=[{"type": "text", "text": json.dumps(prefs, ensure_ascii=False, indent=2)}])
        elif path == "summaries":
            summaries = self.memory_system.get_summaries("default")
            return MCPResult(content=[{"type": "text", "text": json.dumps(summaries, ensure_ascii=False, indent=2)}])
        elif path == "history":
            history = self.memory_system.get_history("default", limit=50)
            return MCPResult(content=[{"type": "text", "text": json.dumps(history, ensure_ascii=False, indent=2)}])
        else:
            return MCPResult(content=[{"type": "text", "text": f"Unknown memory path: {path}"}], is_error=True)
    
    def _read_expert_resource(self, uri: str) -> MCPResult:
        exp_id = uri[len("expert://"):]
        if not self.staff or not hasattr(self.staff, 'expert_registry'):
            return MCPResult(content=[{"type": "text", "text": "Experts not available"}], is_error=True)
        
        expert = self.staff.expert_registry._experts.get(exp_id)
        if not expert:
            return MCPResult(content=[{"type": "text", "text": f"Expert '{exp_id}' not found"}], is_error=True)
        
        return MCPResult(content=[{
            "type": "text",
            "text": json.dumps({
                "id": expert.id,
                "name": expert.name,
                "description": expert.description,
                "domain_tags": expert.domain_tags,
                "tools": expert.tools,
                "system_prompt": expert.system_prompt[:500],
            }, ensure_ascii=False, indent=2),
        }])
    
    # ---- Prompts ------------------------------------------------------
    
    def list_prompts(self) -> List[MCPPromptTemplate]:
        """Available prompt templates."""
        templates = [
            MCPPromptTemplate(
                name="analyze_task",
                description="Analyze a task and determine optimal execution strategy",
            ),
            MCPPromptTemplate(
                name="code_review",
                description="Review code for bugs, style, and improvements",
            ),
            MCPPromptTemplate(
                name="research_report",
                description="Generate a comprehensive research report on a topic",
            ),
            MCPPromptTemplate(
                name="decision_analysis",
                description="Analyze a decision with pros/cons and recommendation",
            ),
        ]
        return templates
    
    # ---- Custom Tool Registration --------------------------------------
    
    def register_tool(self, name: str, fn: Callable, description: str = ""):
        """Register a custom tool."""
        self._custom_tools[name] = fn


# Fix typo
def MPPromptTemplate(*a, **k):
    return MCPPromptTemplate(*a, **k)


# Demo / Test
if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    
    print("=" * 60)
    print("AI-Staff V4 MCP Bridge — Self-Test")
    print("=" * 60)
    
    # Create bridge without full staff instance (test infrastructure)
    from ..skills.registry import create_builtin_registry
    
    bridge = MCPBridge(
        staff=None,
        skill_registry=create_builtin_registry(),
        memory_system=None,
    )
    
    # Test 1: List tools
    tools = bridge.list_tools()
    print(f"\n1. Tools: {len(tools)} available")
    for t in tools[:5]:
        print(f"   - {t.name}: {t.description[:60]}")
    
    # Test 2: List skills via tool
    result = bridge.call_tool("list_skills", {})
    print(f"\n2. list_skills tool result length: {len(result.content[0]['text'])} chars")
    
    # Test 3: Status tool
    result = bridge.call_tool("status", {})
    data = json.loads(result.content[0]["text"])
    print(f"\n3. Status:")
    for k, v in data.items():
        print(f"   {k}: {v}")
    
    # Test 4: Resources
    resources = bridge.list_resources()
    print(f"\n4. Resources: {len(resources)}")
    for r in resources:
        print(f"   - {r.uri}: {r.name}")
    
    # Test 5: Error handling
    result = bridge.call_tool("nonexistent_tool", {})
    print(f'\n5. Error handling: is_error={result.is_error}')
    
    print("\n" + "=" * 60)
    print("MCP Bridge self-test complete!")
