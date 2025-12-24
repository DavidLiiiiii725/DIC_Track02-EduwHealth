# core/orchestrator.py
from pathlib import Path

from agents.graph import build_graph
from safety.escalation import HumanEscalation
from config import RISK_THRESHOLD

from memory.vector_store import VectorStore
from memory.knowledge_graph import KnowledgeGraph
from memory.hybrid_memory import HybridMemory


class TutorOrchestrator:
    def __init__(self, kb_store_dir: str = "kb_store"):
        # 1) Load Vector KB
        kb_store = Path(kb_store_dir)
        index_path = kb_store / "vector.index"
        texts_path = kb_store / "vector_texts.jsonl"

        if not index_path.exists() or not texts_path.exists():
            raise RuntimeError(
                "Vector KB not built yet.\n"
                f"Expected files: {index_path} and {texts_path}"
            )

        vs = VectorStore.load(str(kb_store))

        # 2) Optional KG
        kg = KnowledgeGraph()

        # 3) HybridMemory
        memory = HybridMemory(kg, vs)

        # 4) Build LangGraph
        self.app = build_graph(memory)

        # 5) Safety module
        self.hem = HumanEscalation()

    def handle(self, user_input: str):
        state = self.app.invoke({"user_input": user_input})

        risk = state.get("risk_score", 0.0)
        escalation = self.hem.check(risk) if risk > RISK_THRESHOLD else "OK"

        return {
            "response": state.get("final_response", ""),
            "emotion": state.get("emotion", {}),
            "risk": risk,
            "escalation": escalation,
            # debug: verify RAG really happened
            "rag_context": state.get("rag_context", ""),
        }
