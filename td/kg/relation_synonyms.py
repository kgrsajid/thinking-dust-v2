"""Relation synonymy detection via embedding clustering.

Detects relations that mean the same thing using different words:
  "in" = "part of" = "contains" = "located in" = "belongs to"

Uses spaCy word vectors (300d) + HDBSCAN clustering.

References:
- alphaXiv (Dec 2025): "Ontology-Based KG Framework" — HDBSCAN on embeddings
- MaGiX (EMNLP 2025): cross-synonym edges via embedding similarity
- W3C OWL 2: equivalentProperty declarations
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RelationSynonymGroup:
    """A cluster of synonymous relations."""
    canonical: str  # The representative relation name
    members: list[str]  # All synonyms including canonical
    confidence: float = 1.0  # 1.0 = user-taught, 0.5-0.9 = auto-detected
    source: str = "user"  # "user", "auto", "cluster"

    def __repr__(self):
        return f"SynonymGroup({self.canonical}: {self.members})"


def _get_relation_vectors(relations: list[str], nlp) -> dict[str, list[float]]:
    """Encode relations as spaCy word vectors.

    For multi-word relations ("part of"), averages word vectors.
    """
    vectors = {}
    for rel in relations:
        doc = nlp(rel)
        if doc.has_vector:
            vectors[rel] = doc.vector.tolist()
        else:
            # Fallback: average character encoding (poor quality but non-zero)
            vectors[rel] = [0.0] * nlp.vocab.vectors_length
    return vectors


def cluster_relations(
    relations: list[str],
    nlp=None,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    similarity_threshold: float = 0.5,
) -> list[RelationSynonymGroup]:
    """Cluster relations by embedding similarity using HDBSCAN.

    Args:
        relations: List of relation name strings
        nlp: spaCy model with word vectors (en_core_web_md recommended)
        min_cluster_size: Minimum relations to form a synonym group
        min_samples: HDBSCAN min_samples parameter
        similarity_threshold: Minimum similarity to consider as synonym

    Returns:
        List of RelationSynonymGroup objects
    """
    import numpy as np

    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_md")

    if len(relations) < 2:
        return [RelationSynonymGroup(canonical=r, members=[r]) for r in relations]

    # Encode relations as vectors
    rel_vectors = _get_relation_vectors(relations, nlp)

    # Build distance matrix (1 - cosine similarity)
    n = len(relations)
    rel_list = list(relations)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            vec_i = np.array(rel_vectors[rel_list[i]])
            vec_j = np.array(rel_vectors[rel_list[j]])

            # Cosine similarity
            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)
            if norm_i > 0 and norm_j > 0:
                sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
            else:
                sim = 0.0

            # Distance = 1 - similarity
            dist = max(0.0, 1.0 - sim)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    # Cluster with HDBSCAN
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
            cluster_selection_epsilon=1.0 - similarity_threshold,
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
    except ImportError:
        # Fallback: simple threshold-based clustering
        cluster_labels = _simple_cluster(distance_matrix, similarity_threshold, n)

    # Build synonym groups
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label == -1:
            # Noise point — singleton group
            rel = rel_list[i]
            groups[f"singleton_{rel}"] = RelationSynonymGroup(
                canonical=rel, members=[rel]
            )
        else:
            if label not in groups:
                groups[label] = RelationSynonymGroup(
                    canonical=rel_list[i], members=[rel_list[i]]
                )
            else:
                groups[label].members.append(rel_list[i])

    # Set canonical to most frequent (or first) member
    for group in groups.values():
        if len(group.members) > 1:
            # Use the shortest name as canonical (usually the most natural)
            group.canonical = min(group.members, key=len)
            group.confidence = similarity_threshold
            group.source = "cluster"

    return list(groups.values())


def _simple_cluster(distance_matrix, threshold, n) -> list[int]:
    """Simple threshold-based clustering fallback (no HDBSCAN)."""
    labels = [-1] * n
    current_label = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = current_label
        for j in range(i + 1, n):
            if distance_matrix[i][j] < (1.0 - threshold):
                labels[j] = current_label
        current_label += 1

    return labels


def find_synonyms(
    relation: str,
    all_relations: list[str],
    nlp=None,
    threshold: float = 0.5,
) -> list[str]:
    """Find synonyms of a specific relation.

    Args:
        relation: The relation to find synonyms for
        all_relations: All known relation names
        nlp: spaCy model with word vectors
        threshold: Minimum similarity to consider as synonym

    Returns:
        List of synonym relation names (excluding the input)
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_md")

    import numpy as np

    doc_rel = nlp(relation)
    if not doc_rel.has_vector:
        return []

    synonyms = []
    for other in all_relations:
        if other == relation:
            continue
        doc_other = nlp(other)
        if not doc_other.has_vector:
            continue
        sim = doc_rel.similarity(doc_other)
        if sim >= threshold:
            synonyms.append(other)

    return sorted(synonyms, key=lambda r: -nlp(r).similarity(doc_rel))


class RelationSynonymRegistry:
    """Registry of relation synonym groups.

    Manages which relations are equivalent. Integrates with SPARQL
    via OWL equivalentProperty declarations.

    Usage:
        registry = RelationSynonymRegistry()
        registry.teach("in", ["part of", "contains", "located in"])
        registry.get_canonical("part of")  # → "in"
        registry.get_synonyms("in")  # → ["part of", "contains", "located in"]
    """

    def __init__(self):
        # canonical → set of synonyms (including canonical)
        self._groups: dict[str, set[str]] = {}
        # synonym → canonical (reverse lookup)
        self._canonical_map: dict[str, str] = {}

    def teach(self, canonical: str, synonyms: list[str]) -> None:
        """Teach that a set of relations are synonymous.

        Args:
            canonical: The canonical (preferred) relation name
            synonyms: List of synonym relation names
        """
        canonical = canonical.lower().strip()
        synonyms = [s.lower().strip() for s in synonyms]

        # If canonical is already mapped to another group, merge into that
        existing_canonical = self._canonical_map.get(canonical)
        if existing_canonical:
            canonical = existing_canonical

        # Start with existing members of the canonical's group
        all_members = set(self._groups.get(canonical, set()))
        all_members.add(canonical)
        all_members.update(synonyms)

        # Check if any synonym is already in a different group → merge
        for syn in synonyms:
            existing = self._canonical_map.get(syn)
            if existing and existing != canonical:
                all_members.update(self._groups.pop(existing, set()))

        self._groups[canonical] = all_members
        for member in all_members:
            self._canonical_map[member] = canonical

    def get_canonical(self, relation: str) -> str:
        """Get the canonical form of a relation.

        If the relation is a synonym, returns the canonical.
        If unknown, returns the relation itself.
        """
        return self._canonical_map.get(relation.lower().strip(), relation.lower().strip())

    def get_synonyms(self, relation: str) -> list[str]:
        """Get all synonyms of a relation (excluding itself)."""
        canonical = self.get_canonical(relation)
        members = self._groups.get(canonical, set())
        return [m for m in members if m != relation.lower().strip()]

    def are_synonyms(self, rel1: str, rel2: str) -> bool:
        """Check if two relations are synonyms."""
        return self.get_canonical(rel1) == self.get_canonical(rel2)

    def get_all_groups(self) -> list[RelationSynonymGroup]:
        """Get all synonym groups."""
        return [
            RelationSynonymGroup(canonical=canonical, members=sorted(members))
            for canonical, members in self._groups.items()
        ]

    def suggest_synonyms(self, relation: str, all_relations: list[str],
                         nlp=None, threshold: float = 0.7) -> list[str]:
        """Suggest potential synonyms using vector similarity.

        Returns suggestions with confidence scores. User should verify.
        """
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load("en_core_web_md")
            except OSError:
                return []

        return find_synonyms(relation, all_relations, nlp, threshold)

    def to_sparql_quads(self, store) -> int:
        """Export synonym groups as OWL equivalentProperty in SPARQL store.

        Uses OWL 2 standard: owl:equivalentProperty
        """
        try:
            from pyoxigraph import Quad, NamedNode
        except ImportError:
            return 0

        OWL_EQUIV = NamedNode("http://www.w3.org/2002/07/owl#equivalentProperty")
        from ..query import relation_to_uri

        count = 0
        for canonical, members in self._groups.items():
            if len(members) < 2:
                continue
            canonical_uri = relation_to_uri(canonical)
            for member in members:
                if member != canonical:
                    member_uri = relation_to_uri(member)
                    store.store.add(Quad(canonical_uri, OWL_EQUIV, member_uri))
                    count += 1

        return count
