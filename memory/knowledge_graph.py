import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triplet(self, head, relation, tail):
        self.graph.add_edge(head, tail, relation=relation)

    def query(self, node, depth=2):
        results = set()
        for target in nx.single_source_shortest_path(self.graph, node, cutoff=depth):
            results.add(target)
        return list(results)
