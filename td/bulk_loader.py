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

        for i, triple in enumerate(triples):
            stats.total_lines += 1

            # Unpack
            if len(triple) < 3:
                stats.triples_skipped += 1
                continue

            s, r, o = triple[0], triple[1], triple[2]

            # Map IDs to names if maps provided
            if entity_map:
                s = entity_map.get(s, s)
                o = entity_map.get(o, o)
            if relation_map:
                r = relation_map.get(r, r)

            # Normalize
            s = s.strip().lower()
            r = r.strip().lower()
            o = o.strip().lower()

            # Skip empty
            if not s or not r or not o:
                stats.triples_skipped += 1
                continue

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

        # Register Wikidata relation properties after loading
        # This maps common Wikidata properties to TD v2 inference rules.
        # Reference: https://www.wikidata.org/wiki/Wikidata:List_of_properties
        self._register_wikidata_properties(relation_map)

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

    def _register_wikidata_properties(self, relation_map: dict):
        """Register TD v2 inference properties for common Wikidata relations.

        Maps Wikidata property IDs to TD v2 relation properties (transitive,
        symmetric, functional, inverse). This enables inference on loaded data.

        Reference: https://www.wikidata.org/wiki/Wikidata:List_of_properties
        """
        # Transitive relations (R(X,Y) ∧ R(Y,Z) → R(X,Z))
        transitive = {
            "P131",  # located in administrative territory
            "P17",   # country
            "P27",   # country of citizenship
            "P463",  # member of
            "P361",  # part of
            "P279",  # subclass of
            "P706",  # located in/on physical feature
            "P150",  # contains administrative territory
        }

        # Symmetric relations (R(X,Y) → R(Y,X))
        symmetric = {
            "P3373",  # sibling
            "P26",    # spouse
            "P47",    # shares border with
        }

        # Functional relations (R(X,Y) ∧ R(X,Z) → Y=Z)
        functional = {
            "P36",   # capital (each country has ONE capital)
            "P35",   # head of state
        }
        # NOT functional: P39 (position held) — multiple people hold same position
        # NOT functional: P106 (occupation) — many people share same occupation

        # Inverse pairs (R1(X,Y) → R2(Y,X))
        inverse_pairs = {
            ("P361", "P527"),  # part_of ↔ has_part
            ("P150", "P131"),  # contains ↔ located_in
            ("P127", "P355"),  # owned_by ↔ has_subsidiary
        }

        # Resolve relation IDs to human-readable names and register
        registered = 0
        for rel_id in transitive:
            name = relation_map.get(rel_id)
            if name:
                self.td.kg.set_relation_property(name, "transitive")
                registered += 1

        for rel_id in symmetric:
            name = relation_map.get(rel_id)
            if name:
                self.td.kg.set_relation_property(name, "symmetric")
                registered += 1

        for rel_id in functional:
            name = relation_map.get(rel_id)
            if name:
                self.td.kg.set_relation_property(name, "functional")
                registered += 1

        for rel1_id, rel2_id in inverse_pairs:
            name1 = relation_map.get(rel1_id)
            name2 = relation_map.get(rel2_id)
            if name1 and name2:
                self.td.kg.set_relation_property(name1, "inverse", inverse=name2)
                registered += 1

        # Also register "is_a" and common relations that aren't in Wikidata
        # but emerge from the data
        self.td.kg.set_relation_property("is_a", "transitive")
        self.td.kg.set_relation_property("instance of", "transitive")
        self.td.kg.set_relation_property("subclass of", "transitive")

        print(f"  Registered {registered} Wikidata relation properties")

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
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    yield line
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line
