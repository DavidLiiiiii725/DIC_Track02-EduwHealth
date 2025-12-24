class HybridMemory:
    def __init__(self, kg, vector_store):
        self.kg = kg
        self.vs = vector_store

    def retrieve(self, query, concept=None, k=5, depth=2):
        semantic = self.vs.search(query, k=k)
        structured_nodes = self.kg.query(concept, depth=depth) if concept else []
        structured = [str(x) for x in structured_nodes]
        return {
            "semantic": semantic,
            "structured": structured
        }

    def pick_concepts(self, query: str, top_n: int = 1):
        """
        General concept linking: pick best matching KG node(s) for the query.
        """
        if not hasattr(self.kg, "nodes"):
            return []

        q = query.lower()
        candidates = []
        for n in self.kg.graph.nodes:
            name = str(n)
            score = 0
            if name.lower() in q:
                score += 2
            # 简单词重叠
            overlap = len(set(name.lower().split()) & set(q.split()))
            score += overlap
            if score > 0:
                candidates.append((score, name))

        candidates.sort(reverse=True)
        return [name for _, name in candidates[:top_n]]