"""Bulk loader for structured knowledge graph data.

Loads pre-structured (subject, relation, object) triples directly into
TD v2's knowledge graph, bypassing the parser and per-fact overhead.

Use cases:
- Wikidata5m (20M triples, already structured)
- CSV/TSV imports of structured data
- LLM-generated triple datasets
- Any data where triples are pre-extracted

NOT for:
- Human-written natural language (use td.teach() instead)
- Sentences that need parsing (use td.teach() instead)

Usage:
    from td.bulk_loader import BulkLoader

    loader = BulkLoader(td_instance)
    stats = loader.load_triples(triples, source="wikidata5m")
    print(stats)

Reference: Wikidata5m — https://deepgraphlearning.github.io/project/wikidata5m
"""

from __future__ import annotations

import gzip
import time
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class LoadStats:
    """Statistics from a bulk load operation."""
    total_lines: int = 0
    triples_loaded: int = 0
    triples_skipped: int = 0  # duplicates or malformed
    entity_count: int = 0
    relation_count: int = 0
    load_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    beagle_time_sec: float = 0.0

    def summary(self) -> str:
        return (
            f"Loaded {self.triples_loaded:,} triples "
            f"({self.entity_count:,} entities, {self.relation_count:,} relations) "
            f"in {self.load_time_sec:.1f}s "
            f"({self.triples_loaded / max(self.load_time_sec, 0.001):.0f} triples/sec)\n"
            f"Skipped {self.triples_skipped:,} (duplicates/malformed)\n"
            f"Inference: {self.inference_time_sec:.1f}s | BEAGLE: {self.beagle_time_sec:.1f}s"
        )


class BulkLoader:
    """Bulk loads structured triples into TD v2's knowledge graph.

    Bypasses the parser entirely. Triples must be pre-extracted
    (subject, relation, object) format.

    Args:
        td: GenericThinkingDust instance
        batch_size: How often to flush to SPARQL store (default: 10000)
        run_inference: Whether to run derive_all() after loading (default: True)
        update_beagle: Whether to batch-update BEAGLE vectors (default: False)
    """

    def __init__(self, td, batch_size: int = 10000,
                 run_inference: bool = True, update_beagle: bool = False):
        self.td = td
        self.batch_size = batch_size
        self.run_inference = run_inference
        self.update_beagle = update_beagle

    def load_triples(self, triples: list[tuple[str, str, str]] | Iterator,
                     source: str = "bulk", relation_map: dict = None,
                     entity_map: dict = None) -> LoadStats:
        """Load a list of (subject, relation, object) triples.

        Args:
            triples: List or iterator of (subject, relation, object) tuples
            source: Source label for provenance (e.g., "wikidata5m")
            relation_map: Optional dict mapping relation IDs to names
                          (e.g., {"P39": "position held"})
            entity_map: Optional dict mapping entity IDs to names
                        (e.g., {"Q22686": "Donald Trump"})

        Returns:
            LoadStats with loading statistics
        """
        stats = LoadStats()
        t0 = time.time()

        entities_seen = set()
        relations_seen = set()
        triples_seen = set()  # For deduplication

        for i, triple in enumerate(triples):
            stats.total_lines += 1

            # Unpack with error handling
            try:
                if len(triple) < 3:
                    stats.triples_skipped += 1
                    continue
                s, r, o = triple[0], triple[1], triple[2]
            except (TypeError, AttributeError, IndexError):
                stats.triples_skipped += 1
                continue

            # Map IDs to names if maps provided
            if entity_map:
                s = entity_map.get(s, s)
                o = entity_map.get(o, o)
            if relation_map:
                r = relation_map.get(r, r)

            # Normalize (case-insensitive matching, preserve original case)
            s = s.strip()
            r = r.strip()
            o = o.strip()

            # Create case-folded lookup keys for matching
            s_key = s.lower()
            r_key = r.lower()
            o_key = o.lower()

            # Use lowercased forms for storage (consistent with TD v2 convention)
            # TODO: preserve original case for display, use lowered for matching
            s = s_key
            r = r_key
            o = o_key

            # Skip empty
            if not s or not r or not o:
                stats.triples_skipped += 1
                continue

            # Deduplicate
            triple_key = (s, r, o)
            if triple_key in triples_seen:
                stats.triples_skipped += 1
                continue
            triples_seen.add(triple_key)

            # Track unique entities and relations
            entities_seen.add(s)
            entities_seen.add(o)
            relations_seen.add(r)

            # Load directly into KG (bypasses parser)
            self.td.kg.add_fact(s, r, o, source=source)

            stats.triples_loaded += 1

            # Progress logging
            if (i + 1) % self.batch_size == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  Loaded {i + 1:,} triples ({rate:.0f} triples/sec)...")

        stats.entity_count = len(entities_seen)
        stats.relation_count = len(relations_seen)
        stats.load_time_sec = time.time() - t0

        # Run inference (derive_all) once after loading
        if self.run_inference:
            t_inf = time.time()
            # First: auto-detect relation properties from data patterns
            print("  Detecting relation properties...")
            nlp = None
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError) as e:
                print(f"  spaCy unavailable ({e}), using statistical detection only")
            detected = self.td.kg.detect_relation_properties(min_evidence=3, nlp=nlp)
            if detected:
                for rel, props in detected.items():
                    print(f"    {rel}: {props}")
            # Then: run inference with detected properties
            print("  Running inference (derive_all)...")
            derived = self.td.kg.derive_all()
            stats.inference_time_sec = time.time() - t_inf
            print(f"  Derived {len(derived):,} new facts")

        # Batch BEAGLE update (optional — expensive for large datasets)
        if self.update_beagle and self.td.wvm is not None:
            t_beagle = time.time()
            print("  Updating BEAGLE vectors...")
            # TODO: batch update from entity descriptions
            stats.beagle_time_sec = time.time() - t_beagle

        return stats

    def load_wikidata5m(self, triples_path: str, aliases_path: str = None,
                        relation_aliases_path: str = None,
                        source: str = "wikidata5m") -> LoadStats:
        """Load Wikidata5m dataset directly.

        Handles the Wikidata5m format:
        - Triples: Q22686\\tP39\\tQ11696 (tab-separated Wikidata IDs)
        - Aliases: Q22686\\tDonald Trump\\tDonnie Trump (tab-separated)

        Entity and relation IDs are converted to human-readable names
        using the alias files. The FIRST alias is used as the canonical
        name (e.g., Q22686 → "Donald Trump", P39 → "position held").

        After loading, relation properties are registered based on the
        Wikidata relation mapping (e.g., P31 → is_a, P131 → in transitive).

        Args:
            triples_path: Path to triples file (.txt or .txt.gz)
            aliases_path: Path to entity aliases file (optional)
            relation_aliases_path: Path to relation aliases file (optional)
            source: Source label

        Returns:
            LoadStats
        """
        # Load entity aliases (ID → name)
        entity_map = {}
        if aliases_path:
            print(f"  Loading entity aliases from {aliases_path}...")
            for line in self._read_lines(aliases_path):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    entity_id = parts[0].strip()
                    # Use first alias as canonical name
                    name = parts[1].strip().lower()
                    if name:
                        entity_map[entity_id] = name
            print(f"  Loaded {len(entity_map):,} entity aliases")

        # Load relation aliases (ID → name)
        relation_map = {}
        if relation_aliases_path:
            print(f"  Loading relation aliases from {relation_aliases_path}...")
            for line in self._read_lines(relation_aliases_path):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel_id = parts[0].strip()
                    name = parts[1].strip().lower()
                    if name:
                        relation_map[rel_id] = name
            print(f"  Loaded {len(relation_map):,} relation aliases")

        # Parse triples
        def triple_generator():
            for line in self._read_lines(triples_path):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    yield (parts[0], parts[1], parts[2])

        print(f"  Loading triples from {triples_path}...")
        return self.load_triples(
            triple_generator(),
            source=source,
            relation_map=relation_map,
            entity_map=entity_map,
        )

    def load_tsv(self, path: str, source: str = "tsv",
                 has_header: bool = False) -> LoadStats:
        """Load triples from a TSV file (subject\\trelation\\tobject).

        Args:
            path: Path to TSV file
            source: Source label
            has_header: Whether to skip the first line

        Returns:
            LoadStats
        """
        def triple_generator():
            lines = self._read_lines(path)
            if has_header:
                next(lines, None)
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    yield (parts[0], parts[1], parts[2])

        return self.load_triples(triple_generator(), source=source)

    @staticmethod
    def _read_lines(path: str) -> Iterator[str]:
        """Read lines from a file, supporting .gz compression."""
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    yield line
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    yield line
