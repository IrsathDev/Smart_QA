# agent.py
# Step 3: Agentic RAG with LangGraph — Router + Memory + Chitchat

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from utils import ask_nvidia
from rag_utils import hybrid_search


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STATE — the shared object that flows through every node
# ═══════════════════════════════════════════════════════════════════════════════
# TypedDict is like a Python dict with typed keys.
# LangGraph passes this state object between every node automatically.
# Each node can READ any key and WRITE (update) any key.

class AgentState(TypedDict):
    question:   str          # the user's current question
    history:    list[dict]   # last N chat turns [{"role":..,"content":..}]
    route:      str          # "rag" or "chitchat" — decided by router node
    context:    str          # retrieved document chunks (only for RAG route)
    answer:     str          # final answer from the LLM


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NODE FUNCTIONS — each node receives the full state, returns partial update
# ═══════════════════════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> dict:
    """
    Decides whether the question needs document retrieval (RAG)
    or is just a general/greeting message (chitchat).

    How: asks the LLM to classify the question with a strict prompt.
    Returns: {"route": "rag"} or {"route": "chitchat"}
    """
    question = state["question"]

    classify_prompt = f"""Classify this user question into exactly one category.
Reply with ONLY the word: rag   OR   chitchat

- rag      → needs information from an uploaded document
- chitchat → greeting, thanks, general conversation, small talk

Question: {question}
Answer:"""

    result = ask_nvidia(classify_prompt, []).strip().lower()

    # Safety fallback — if LLM gives unexpected output, default to rag
    route = "rag" if "rag" in result else "chitchat"
    print(f"🔀 Router decision: '{route}' for question: '{question[:60]}'")

    return {"route": route}


def rag_node(state: AgentState, chroma_collection, bm25_index, chunks_store) -> dict:
    """
    Runs hybrid search (BM25 + Vector + RRF) to find relevant chunks,
    then calls the LLM with those chunks as context PLUS the chat history.

    This is where Step 2's hybrid search is actually used inside the agent.
    Returns: {"context": ..., "answer": ...}
    """
    question = state["question"]
    history  = state["history"]

    # ── Retrieve relevant chunks ───────────────────────────────────────────────
    chunks = hybrid_search(
        query=question,
        all_chunks=chunks_store,
        chroma_collection=chroma_collection,
        bm25_index=bm25_index,
        top_k=5
    )
    context = "\n\n".join(chunks)

    # ── Build system message ───────────────────────────────────────────────────
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer using ONLY the document context below.\n"
            "Mention page numbers like (Page 2) if available.\n"
            "If not found, say: 'Not available in the document.'\n\n"
            f"Context:\n{context}"
        )
    }

    # ── Inject history so the LLM remembers previous turns ────────────────────
    # Format: [system, ...history turns..., current question]
    messages = [system_msg] + history + [{"role": "user", "content": question}]

    answer = ask_nvidia("", messages)   # prompt is inside messages, not separate
    return {"context": context, "answer": answer}


def chitchat_node(state: AgentState) -> dict:
    """
    Handles greetings, thanks, and general conversation.
    No document retrieval needed — just history-aware LLM reply.
    Returns: {"answer": ...}
    """
    question = state["question"]
    history  = state["history"]

    system_msg = {
        "role": "system",
        "content": (
            "You are a friendly, helpful assistant. "
            "Answer conversationally. Keep it short and warm."
        )
    }

    messages = [system_msg] + history + [{"role": "user", "content": question}]
    answer = ask_nvidia("", messages)

    return {"context": "", "answer": answer}


def memory_node(state: AgentState) -> dict:
    """
    This node doesn't call the LLM — it just returns the state unchanged.
    The actual saving to SQLite happens in app.py (after graph runs).
    Its purpose here is to be a clean final checkpoint in the graph
    where we could add memory trimming, summarisation, etc. in future steps.
    """
    # LangGraph requires a node to write to at least one state key.
    return {"answer": state.get("answer", "")}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ROUTING FUNCTION — tells LangGraph which edge to follow
# ═══════════════════════════════════════════════════════════════════════════════

def route_decision(state: AgentState) -> str:
    """
    Called by LangGraph after router_node runs.
    Returns the name of the next node to execute.
    """
    return state["route"]   # "rag" or "chitchat"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BUILD GRAPH — wire nodes and edges together
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent(chroma_collection, bm25_index, chunks_store):
    """
    Builds and compiles the LangGraph agent.

    Graph structure:
        router → (conditional) → rag_node     → memory → END
                              → chitchat_node → memory → END

    Returns a compiled LangGraph app you can call with .invoke(state).
    """

    # ── Wrap nodes that need extra args (closures) ────────────────────────────
    def _rag(state):
        return rag_node(state, chroma_collection, bm25_index, chunks_store)

    # ── Create graph with our state schema ────────────────────────────────────
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("router",   router_node)
    graph.add_node("rag",      _rag)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("memory",   memory_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional edge: router → rag OR chitchat ────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_decision,                      # function that reads state["route"]
        {"rag": "rag", "chitchat": "chitchat"}  # mapping return value → node name
    )

    # ── Both branches converge at memory, then END ────────────────────────────
    graph.add_edge("rag",      "memory")
    graph.add_edge("chitchat", "memory")
    graph.add_edge("memory",   END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RUN AGENT — single entry point called from app.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(
    question: str,
    history: list[dict],
    chroma_collection,
    bm25_index,
    chunks_store: list[str]
) -> tuple[str, str]:
    """
    Main function called by app.py for every user question.

    Builds the agent fresh each call (stateless — state lives in SQLite).
    Returns: (answer_text, route_taken)
    """
    agent = build_agent(chroma_collection, bm25_index, chunks_store)

    initial_state: AgentState = {
        "question": question,
        "history":  history,
        "route":    "",
        "context":  "",
        "answer":   ""
    }

    final_state = agent.invoke(initial_state)

    return final_state["answer"], final_state["route"]