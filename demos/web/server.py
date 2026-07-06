#!/usr/bin/env python3
"""Thinking Dust v2 — Web Server

Serves the demo UI and provides API endpoints for TD v2.
Run: arch -arm64 .venv-arm64/bin/python demos/web/server.py
Open: http://localhost:5000
"""

import sys
import os
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
from td.thinking import GenericThinkingDust
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

# ── Initialize TD v2 ────────────────────────────────────────
print("Initializing Thinking Dust v2...")
vocab = build_default_vocabulary(dim=10000)
mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
td = GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10000, pure_mode=True)
print(f"Ready. {len(td.kg.triples)} triples loaded.")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/teach", methods=["POST"])
def teach():
    """Teach a fact to TD v2."""
    data = request.json
    text = data.get("text", "").strip()
    answer = data.get("answer", "").strip() or None

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = td.teach(text, answer or text)
        return jsonify({
            "success": True,
            "message": result.get("message", "Taught"),
            "triples": len(td.kg.triples),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    """Ask TD v2 a question."""
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = td.think(text)
        solution = result.solution or {}
        return jsonify({
            "type": solution.get("type", "unknown"),
            "answer": solution.get("formatted", ""),
            "confidence": solution.get("confidence", 0),
            "method": solution.get("method", ""),
            "similarity": solution.get("similarity", 0),
            "trace": result.trace if hasattr(result, "trace") else [],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/teach_relation", methods=["POST"])
def teach_relation():
    """Teach a relation property."""
    data = request.json
    name = data.get("name", "").strip()
    prop = data.get("property", "").strip()

    if not name or not prop:
        return jsonify({"error": "Need name and property"}), 400

    try:
        td.teach_relation(name, prop)
        return jsonify({"success": True, "message": f"{name} → {prop}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Get KG statistics."""
    relations = set()
    for t in td.kg.triples:
        relations.add(t.relation)

    return jsonify({
        "triples": len(td.kg.triples),
        "relations": len(relations),
        "relation_names": sorted(relations),
        "properties": len(td.kg.relation_properties),
        "thinks": td.total_thinks,
    })


@app.route("/api/kg", methods=["GET"])
def knowledge_graph():
    """Get full KG as nodes and edges for visualization."""
    nodes = {}
    edges = []

    for t in td.kg.triples:
        s_key = t.subject.lower().replace(" ", "_")
        o_key = t.object.lower().replace(" ", "_")

        if s_key not in nodes:
            nodes[s_key] = {"id": s_key, "label": t.subject, "derived": t.source == "derived"}
        if o_key not in nodes:
            nodes[o_key] = {"id": o_key, "label": t.object, "derived": t.source == "derived"}

        edges.append({
            "source": s_key,
            "target": o_key,
            "label": t.relation.replace("_", " "),
            "derived": t.source == "derived",
        })

    return jsonify({
        "nodes": list(nodes.values()),
        "edges": edges,
    })


if __name__ == "__main__":
    print("\n✦ Thinking Dust v2 — Web Server")
    print(f"  → http://localhost:5000")
    print(f"  → {len(td.kg.triples)} facts in KG")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
