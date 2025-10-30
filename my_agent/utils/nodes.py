from langchain.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from my_agent.utils.tools import add, multiply, divide
from my_agent.utils.state import MessagesState
from typing import Literal
from langgraph.graph import END
import os

load_dotenv()

# Modelo con OpenRouter (compatible con quickstart)
model = ChatOpenAI(
    model="openai/gpt-oss-20b:free",  # Cambia por otro si quieres (ej. "meta-llama/llama-3.1-8b-instruct:free")
    temperature=0,
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

def llm_call(state):
    """LLM decide tool o responde."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state):
    """Ejecuta tools."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if "tool_calls" in last_message and last_message["tool_calls"]:
        return "tool_node"
    return END


