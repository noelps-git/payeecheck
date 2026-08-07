"""
l6_graph.py — Level 6: Graph-Augmented Entity Resolution

Builds an attribute graph (VPA, account, mobile, name) and resolves
entities by shared attributes, not just name similarity. Seeded with
real known merchant VPA patterns from corpus.py plus illustrative
mule-pattern examples for the demo.
See: PayeeCheck Engineering Playbook, Level 6.
"""
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
try:
    from l5_siamese import match as name_match
    from corpus import KNOWN_MERCHANT_VPAS
except ImportError:
    from matchers.l5_siamese import match as name_match
    from matchers.corpus import KNOWN_MERCHANT_VPAS

G = nx.Graph()


def register_entity(entity_id: str, attributes: dict):
    G.add_node(entity_id, type="entity")
    for attr_type, value in attributes.items():
        if not value:
            continue
        attr_node = f"{attr_type}:{value}"
        G.add_node(attr_node, type=attr_type, value=value)
        G.add_edge(entity_id, attr_node, weight=1.0)


def _seed_graph():
    """Seed with real known merchant VPAs + illustrative mule-pattern entities."""
    # Real merchant VPAs (from corpus.py — public knowledge)
    for name, vpa, psp in KNOWN_MERCHANT_VPAS:
        register_entity(f"MERCHANT_{name.replace(' ', '_').upper()}", {
            "vpa": vpa, "name": name, "psp": psp,
        })

    # Illustrative fraud-ring example for demonstration (NOT real data —
    # these are constructed to show how shared-attribute resolution works)
    register_entity("ENT_DEMO_001", {
        "vpa": "krishna.ent@ybl", "account": "KKBK0004891",
        "mobile": "9000000001", "name": "Krishna Enterprises",
    })
    register_entity("ENT_DEMO_002", {
        "vpa": "k.enterprises@oksbi",
        "mobile": "9000000001",  # shared mobile -> same fraud ring
        "name": "K. Enterprises",
    })
    register_entity("ENT_DEMO_003", {
        "vpa": "krishna.traders@axl",
        "mobile": "9000000001",  # same mobile again -> ring of 3
        "name": "Krishna Traders",
    })


_seed_graph()


def find_shared_attributes(attrs_query: dict) -> dict:
    matches = {}
    for attr_type, value in attrs_query.items():
        if not value:
            continue
        attr_node = f"{attr_type}:{value}"
        if not G.has_node(attr_node):
            continue
        for n in G.neighbors(attr_node):
            if G.nodes[n].get("type") == "entity":
                matches.setdefault(n, []).append({"attr": attr_type, "value": value})
    return matches


def resolve(query: dict) -> dict:
    graph_hits = find_shared_attributes(query)
    results = []
    for entity_id, shared in graph_hits.items():
        known_name = next(
            (G.nodes[n]["value"] for n in G.neighbors(entity_id)
             if G.nodes[n].get("type") == "name"),
            entity_id,
        )
        name_r = name_match(query.get("name", ""), known_name)
        attr_score = min(len(shared) * 0.30, 1.0)
        combined = round(0.40 * name_r["score"] + 0.60 * attr_score, 2)
        results.append({
            "entity_id": entity_id,
            "known_name": known_name,
            "shared_attrs": shared,
            "name_score": name_r["score"],
            "attribute_score": round(attr_score, 2),
            "combined_confidence": combined,
            "resolution": "SAME_ENTITY" if combined >= 0.7 else "POSSIBLE_MATCH",
        })

    results.sort(key=lambda x: x["combined_confidence"], reverse=True)

    return {
        "query": query,
        "candidates": results,
        "resolved": results[0] if results else None,
        "ring_size": len(results),
        "level": 6,
        "algorithm": "graph_entity_resolution + siamese_name_match",
    }


def find_rings(min_ring_size: int = 2) -> list:
    """
    Scan the whole graph for connected entity clusters sharing any
    attribute — this is how fraud rings get surfaced even when no
    individual VPA looks suspicious on its own.
    """
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "entity"]
    seen = set()
    rings = []
    for node in entity_nodes:
        if node in seen:
            continue
        # 2-hop neighbourhood: entity -> attribute -> other entities
        cluster = set()
        for attr in G.neighbors(node):
            for other_entity in G.neighbors(attr):
                if G.nodes[other_entity].get("type") == "entity":
                    cluster.add(other_entity)
        if len(cluster) >= min_ring_size:
            rings.append(sorted(cluster))
            seen |= cluster
    return rings


if __name__ == "__main__":
    # Weak name match, but shared mobile collapses to same entity
    r = resolve({
        "name": "K. Ent.",
        "vpa": "k.enterprises@oksbi",
        "mobile": "9000000001",
        "account": None,
    })
    print("Resolution result:")
    print(r["resolved"])

    print("\nFraud rings detected in graph:")
    for ring in find_rings():
        print(f"  Ring of {len(ring)}: {ring}")
