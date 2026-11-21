
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    make_retrieve_node,
    grade_node,
    rewrite_node,
    generate_node,
    fallback_node,
    self_check_node,
    regenerate_node
)


def build_agent(vector_store):

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("retrieve", make_retrieve_node(vector_store))
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("self_check", self_check_node)
    workflow.add_node("regenerate", regenerate_node)

    # Initial workflow
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")

    # Document relevance decision
    def decide_retrieval(state):

        if state["relevant"]:
            return "generate"

        if state["retries"] < 2:
            return "rewrite"

        return "fallback"

    workflow.add_conditional_edges(
        "grade",
        decide_retrieval,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback"
        }
    )

    # Retrieval rewrite loop
    workflow.add_edge("rewrite", "retrieve")

    # Answer generation and checking
    workflow.add_edge("generate", "self_check")

    def decide_answer(state):

        if state["answer_supported"]:
            return "end"

        if state["answer_retries"] < 1:
            return "regenerate"

        return "fallback"

    workflow.add_conditional_edges(
        "self_check",
        decide_answer,
        {
            "end": END,
            "regenerate": "regenerate",
            "fallback": "fallback"
        }
    )

    workflow.add_edge("regenerate", "self_check")

    workflow.add_edge("fallback", END)

    return workflow.compile()
