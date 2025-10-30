from langgraph.graph import StateGraph, START, END
from typing import Literal
from my_agent.utils.state import MessagesState
from my_agent.utils.nodes import llm_call, tool_node, should_continue
from dotenv import load_dotenv

load_dotenv()

# Construye grafo
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compila y exporta
agent = agent_builder.compile()

