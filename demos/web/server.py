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

# Enable CORS for all routes
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

# ── Initialize TD v2 ────────────────────────────────────────
print("Initializing Thinking Dust v2...")
vocab = build_default_vocabulary(dim=10000)
mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
td = GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10000, pure_mode=True)

# Load existing knowledge from RDF store
store_path = os.path.join(PROJECT_ROOT, "data", "td_store")
if os.path.exists(store_path):
    td.kg.load(store_path)
    print(f"Loaded {len(td.kg.triples)} triples from RDF store.")
else:
    print("No existing store found. Starting fresh.")


def save_kg():
    """Save KG to RDF store after teaching."""
    try:
        td.kg.save(store_path)
    except Exception as e:
        print(f"Warning: Could not save KG: {e}")


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
        save_kg()  # Persist after teaching
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


@app.route("/api/teach_triples", methods=["POST"])
def teach_triples():
    """Teach validated triples from LLM output."""
    data = request.json
    triples = data.get("triples", [])

    if not triples:
        return jsonify({"error": "No triples provided"}), 400

    taught = 0
    skipped = 0
    errors = []

    for t in triples:
        s = str(t.get("subject", "")).strip()
        r = str(t.get("relation", "")).strip().lower().replace(" ", "_")
        o = str(t.get("object", "")).strip()

        # Validate: subject and object must be 1-5 words, non-empty
        if not s or not o or not r:
            skipped += 1
            continue
        if len(s.split()) > 5 or len(o.split()) > 5:
            skipped += 1
            continue
        # Skip if subject == object
        if s.lower() == o.lower():
            skipped += 1
            continue
        # Skip garbage relations
        if len(r) < 2:
            skipped += 1
            continue

        try:
            fact = f"{s} {r.replace('_', ' ')} {o}"
            td.teach(fact, fact)
            taught += 1
        except Exception as e:
            errors.append(f"{s} {r} {o}: {str(e)[:50]}")

    save_kg()

    return jsonify({
        "success": True,
        "taught": taught,
        "skipped": skipped,
        "errors": errors[:5],
        "total_triples": len(td.kg.triples),
    })
def teach_relation():
    """Teach a relation property."""
    data = request.json
    name = data.get("name", "").strip()
    prop = data.get("property", "").strip()

    if not name or not prop:
        return jsonify({"error": "Need name and property"}), 400

    try:
        td.teach_relation(name, prop)
        save_kg()  # Persist after teaching
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


@app.route("/api/fetch_url", methods=["POST"])
def fetch_url():
    """Fetch URL content for LLM simplification."""
    data = request.json
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        import urllib.request
        import re as regex
        import ssl

        # Create SSL context that handles certificate issues
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Strip HTML tags — aggressive cleanup before LLM
        # Remove scripts, styles, nav, footer, headers
        html = regex.sub(r"<script[^>]*>.*?</script>", "", html, flags=regex.DOTALL | regex.IGNORECASE)
        html = regex.sub(r"<style[^>]*>.*?</style>", "", html, flags=regex.DOTALL | regex.IGNORECASE)
        html = regex.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=regex.DOTALL | regex.IGNORECASE)
        html = regex.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=regex.DOTALL | regex.IGNORECASE)
        html = regex.sub(r"<header[^>]*>.*?</header>", "", html, flags=regex.DOTALL | regex.IGNORECASE)
        # Remove tags
        text = regex.sub(r"<[^>]+>", " ", html)
        # Clean whitespace and special chars
        text = regex.sub(r"\s+", " ", text).strip()
        # Remove common wiki boilerplate
        text = regex.sub(r"(?i)(navigation menu|personal tools|namespaces|views|search|navigation|toolbox|print/export|languages|what links here|related changes|upload file|special pages|permanent link|page information|cite this page|get shortened URL|wikidata item|download as PDF|printable version|content is available under|privacy policy|about|disclaimers)", "", text)

        if not text:
            return jsonify({"error": "No text content found"}), 400

        return jsonify({"text": text[:5000], "url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n✦ Thinking Dust v2 — Web Server")
    print(f"  → http://localhost:5000")
    print(f"  → {len(td.kg.triples)} facts in KG")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
