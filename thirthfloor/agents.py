from dataclasses import dataclass
import json, re, logging

@dataclass
class Tool:
    name: str
    description: str
    fn: callable

def tool(fn):
    """Decorator: wraps a function as a Tool using its name and docstring."""
    return Tool(name=fn.__name__, description=fn.__doc__ or "", fn=fn)

def run_agent(engine, alias, user_message, tools=None, system=None, max_turns=10):
    """
    Run a ReAct loop. Returns the final response string.
    Model signals tool calls with JSON anywhere in its response:
      {"tool": "name", "args": {"key": "value"}}
    """
    tools = tools or []
    tool_map = {t.name: t for t in tools}
    tools_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    sys = system or ""
    if tools:
        sys += f"\n\nAvailable tools:\n{tools_desc}"
        sys += '\nTo call a tool, include {"tool":"name","args":{...}} in your response.'
    messages = []
    if sys:
        messages.append({"role":"system","content":sys})
    messages.append({"role":"user","content":user_message})
    for _ in range(max_turns):
        content = engine.chat(alias, messages)
        messages.append({"role":"assistant","content":content})
        match = re.search(r'\{"tool":\s*"([^"]+)",\s*"args":\s*(\{[^}]*\})\}', content)
        if match and match.group(1) in tool_map:
            try:
                result = tool_map[match.group(1)].fn(**json.loads(match.group(2)))
            except Exception as e:
                result = f"Error: {e}"
            messages.append({"role":"user","content":f"Tool result: {result}"})
        else:
            return content
    return content
