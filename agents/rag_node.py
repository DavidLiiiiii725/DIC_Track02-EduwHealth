from typing import Dict, Any, List, Optional
import re

def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def rag_retrieve_node(
    state: Dict[str, Any],
    memory,
    *,
    k: int = 6,
    depth: int = 2,
    budget_chars: int = 2200,
    seed_top_n: int = 1,
) -> Dict[str, Any]:
    """
    General RAG node:
    - Choose KG seed concept automatically (no hard-coded domains).
    - Retrieve from vector store + KG, then build a compact context.
    - Return *partial updates only* to avoid LangGraph concurrent update issues.

    Expected memory interface:
      memory.retrieve(query, concept=None, k=..., depth=...) -> {"semantic": [...], "structured": [...]}
    Optional memory interface (recommended for better generality):
      memory.pick_concepts(query, top_n=...) -> ["ConceptA", "ConceptB", ...]
    """

    user_q = _normalize(state["user_input"])

    # 1) Pick concept seeds (general)
    concepts: List[str] = []
    if hasattr(memory, "pick_concepts") and callable(getattr(memory, "pick_concepts")):
        try:
            concepts = memory.pick_concepts(user_q, top_n=seed_top_n) or []
        except Exception:
            concepts = []
    concept: Optional[str] = concepts[0] if concepts else None

    # 2) Retrieve hybrid evidence
    retrieved = memory.retrieve(query=user_q, concept=concept, k=k, depth=depth)
    semantic = retrieved.get("semantic", []) or []
    structured = retrieved.get("structured", []) or []

    # 3) Build a structured evidence pack (general, explainable)
    evidence = {
        "query": user_q,
        "concept_seed": concept,
        "vector_hits": [{"text": _normalize(t)} for t in semantic if _normalize(t)],
        "kg_evidence": [{"text": _normalize(t)} for t in structured if _normalize(t)],
    }

    # 4) Build a compact context with a budget (chars-based; easy + model-agnostic)
    lines: List[str] = []
    lines.append("You are given retrieved evidence to ground your answer.")
    lines.append("Use it as primary support; if insufficient, ask one clarifying question.\n")

    if concept:
        lines.append(f"[KG seed concept] {concept}\n")

    if evidence["vector_hits"]:
        lines.append("## Retrieved Notes (Vector Store)")
        for item in evidence["vector_hits"]:
            lines.append(f"- {item['text']}")

    if evidence["kg_evidence"]:
        lines.append("\n## Retrieved Relations (Knowledge Graph)")
        for item in evidence["kg_evidence"]:
            lines.append(f"- {item['text']}")

    rag_context = "\n".join(lines).strip()

    # enforce budget
    if len(rag_context) > budget_chars:
        rag_context = rag_context[:budget_chars].rsplit("\n", 1)[0] + "\n...(truncated)"

    # 5) Return partial update only (LangGraph-safe)
    return {
        "rag_context": rag_context,
        "rag_evidence": evidence,       # 结构化 evidence（推荐保留）
        "rag_semantic": semantic,       # 兼容你旧字段
        "rag_structured": structured,   # 兼容你旧字段
    }
