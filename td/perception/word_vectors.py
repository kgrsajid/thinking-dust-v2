#!/usr/bin/env python3
"""BEAGLE-style word vector trainer for TD v2.

Based on Jones & Mewhort (2007) "Bound Encoding of the Aggregate Language Environment"

Architecture:
    Each word has THREE vectors:
    1. Environmental vector (e_word): static random identity vector, never changes
    2. Context vector (c_word): accumulates co-occurrence info from sentences
    3. Memory vector (m_word): final vector = normalize(e_word + c_word)

    Phase 1 (this file): context only (no order/convolution)
    Phase 2 will add order vectors with circular convolution

Algorithm (context part):
    For each sentence in corpus:
        content_words = non-stop-words in sentence
        For each word w in content_words:
            context = all OTHER content words in sentence
            c_word += sum of environmental vectors of context words
            (high-frequency function words are removed before counting)

    After training: m_word = normalize(e_word + c_word)

    Words appearing in similar contexts get similar context vectors,
    making their memory vectors similar. This is pure co-occurrence semantics.

Usage:
    from td.perception.word_vectors import WordVectorModel
    wvm = WordVectorModel(dim=10000)
    wvm.train(corpus_sentences)
    wvm.save("data/word_vectors.pkl")

    # At runtime:
    vec = wvm.get("capital")  # memory vector with semantics
    sim = wvm.similarity("capital", "city")  # should be >0.3 after training
    query_hv = wvm.encode_query("capital of france?")  # bag of content words
"""

from __future__ import annotations

import pickle
import time
import re
import numpy as np
from collections import defaultdict

from .hdc import generate_hypervector, bundle, similarity, normalize_hdc


# Minimal English stop words (function words removed before co-occurrence counting)
STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "shall", "it", "its", "this",
    "that", "these", "those", "there", "their", "they", "them", "of", "to",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "about", "against", "between", "under",
    "then", "once", "here", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "but",
    "if", "because", "until", "while", "down", "out", "off", "over",
    "i", "you", "he", "she", "we", "me", "him", "her", "us",
    "what", "which", "who", "whom", "whose",
})


def tokenize(text: str) -> list[str]:
    """Minimal tokenization: lowercase, strip punctuation, split."""
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return [t.strip("-'") for t in text.split() if t.strip()]


def content_words(tokens: list[str]) -> list[str]:
    """Return only content words (non-stop-words with alnum chars)."""
    return [t for t in tokens if t not in STOP_WORDS and any(c.isalnum() for c in t)]


class WordVectorModel:
    """BEAGLE-style word vectors for TD v2.

    Each word has:
        - environmental vector (static random identity)
        - context vector (accumulated co-occurrence from corpus)
        - memory vector (environmental + context, normalized)

    After training, words in similar contexts have similar memory vectors.

    Properties:
        - Pure HDC (bundle = addition + sign, no neural network)
        - Online learning (can add sentences incrementally)
        - O(1) lookup at runtime
        - ~10KB per word (10000-dim int8)

    WSD (Word Sense Disambiguation):
        - sense_clusters: per-word list of (context_vec, count, example_sentence)
        - New contexts assigned to best-matching cluster or create new one
        - Enables context-dependent sense routing for polysemous words

    References:
        Jones & Mewhort (2007), Psychological Review 114(1): 1-37.
            BEAGLE context vectors encode co-occurrence.
        Ruas et al. (2020), Expert Systems with Applications.
            "Sentence-level context expands graph connectivity beyond word-level."
        AlMousa et al. (2022), ACM TALLIP.
            "Captures maximum sentence context while maintaining term order."
        Melamud et al. (2016), CoNLL.
            Context-dependent embeddings outperform context-independent for WSD.
    """

    # Similarity threshold for assigning to existing sense cluster.
    # Below this, a new cluster is created.
    # Tuned: 0.15 is BEAGLE's effective semantic threshold (Jones & Mewhort, 2007).
    SENSE_CLUSTER_NEW_THRESHOLD = 0.15

    # Similarity threshold for merging two clusters (anti-fragmentation).
    # Above this, clusters are too similar and should be merged.
    SENSE_CLUSTER_MERGE_THRESHOLD = 0.95

    def __init__(self, dim: int = 10000):
        self.dim = dim

        # Environmental vectors: static random identity per word (never changes)
        self.env_hvs: dict[str, np.ndarray] = {}

        # Context vectors: accumulated co-occurrence (float, not bipolar)
        self.ctx_hvs: dict[str, np.ndarray] = {}

        # Word frequency
        self.word_freq: dict[str, int] = defaultdict(int)

        self.trained_sentences = 0

        # Cache for memory vectors (invalidated on update)
        self._mem_cache: dict[str, np.ndarray] = {}
        self._cache_dirty = True

        # ── WSD: Per-word sense clusters ──────────────────────────────
        # Each entry: list of (context_vec, count, example_sentence)
        # context_vec: accumulated environmental vectors for this sense
        # count: number of sentences assigned to this cluster
        # example_sentence: first sentence that created this cluster
        #
        # This is the Tier 1 WSD mechanism from the WSD_MILESTONE_SPEC.
        # Instead of one muddled context vector per word, we maintain
        # separate accumulators per sense. New contexts are assigned to
        # the best-matching cluster (or create a new one).
        #
        # Cost: ~24KB per extra sense (10K-dim float32).
        # Most words have 1 sense → no increase.
        self.sense_clusters: dict[str, list[tuple[np.ndarray, int, str]]] = {}

    def _get_or_create_env(self, word: str) -> np.ndarray:
        """Get or create the environmental (static identity) vector for a word."""
        if word not in self.env_hvs:
            self.env_hvs[word] = generate_hypervector(self.dim)
            # Initialize empty context accumulator
            self.ctx_hvs[word] = np.zeros(self.dim, dtype=np.float32)
        return self.env_hvs[word]

    def train_sentence(self, sentence: str) -> None:
        """Train on a single sentence (BEAGLE context accumulation).

        For each content word w in the sentence:
            c_w += sum of environmental vectors of all OTHER content words

        This is the exact BEAGLE context update from Jones & Mewhort (2007).
        Function words are removed before counting. The environmental vector
        stays fixed; only the context vector accumulates.
        """
        tokens = tokenize(sentence)
        words = content_words(tokens)

        if len(words) < 2:
            return  # Need at least 2 content words for co-occurrence

        # Ensure all words have environmental + context vectors
        for w in words:
            self._get_or_create_env(w)
            self.word_freq[w] += 1

        # Sum of ALL environmental vectors (for subtraction later)
        total_env = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            total_env += self.env_hvs[w].astype(np.float32)

        # For each word, add context (sum of OTHER words' environmental vectors)
        for w in words:
            # Context for w = total - w's own environmental vector
            context_sum = total_env - self.env_hvs[w].astype(np.float32)
            self.ctx_hvs[w] += context_sum

        self.trained_sentences += 1
        self._cache_dirty = True

        # Update sense clusters (WSD Tier 1)
        # Uses full sentence context per Ruas et al. (2020) and AlMousa et al. (2022)
        for w in words:
            ctx = self._get_sentence_context_vector(sentence, w)
            if ctx is not None:
                self._assign_to_cluster(w, ctx, sentence)

    def train(self, sentences: list[str], verbose: bool = True) -> None:
        """Train on a corpus of sentences."""
        t0 = time.perf_counter()
        for i, sentence in enumerate(sentences):
            self.train_sentence(sentence)
            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Trained {i+1}/{len(sentences)} sentences "
                      f"({elapsed:.1f}s, {len(self.env_hvs)} words)")

        # Finalize memory vectors
        self._build_mem_cache()

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"Training complete: {len(sentences)} sentences, "
                  f"{len(self.env_hvs)} unique words, {elapsed:.1f}s")

    def _build_mem_cache(self) -> None:
        """Build cached memory vectors: normalize(environmental + context)."""
        self._mem_cache = {}
        for word in self.env_hvs:
            env = self.env_hvs[word].astype(np.float32)
            ctx = self.ctx_hvs[word]
            # Memory = environmental + context, then bipolarize
            mem = env + ctx
            self._mem_cache[word] = normalize_hdc(mem)
        self._cache_dirty = False

    def get(self, word: str) -> np.ndarray | None:
        """Get trained memory vector for a word, or None if unknown.

        Memory vector = normalize(environmental + context).
        This vector has semantic information from corpus co-occurrence.
        """
        if self._cache_dirty:
            self._build_mem_cache()
        return self._mem_cache.get(word.lower())

    def get_or_random(self, word: str) -> np.ndarray:
        """Get memory vector, or random if unknown (graceful fallback)."""
        vec = self.get(word)
        if vec is not None:
            return vec
        # Unknown word: return its environmental vector (random, but consistent)
        return self._get_or_create_env(word.lower())

    def train_incremental(self, sentence: str) -> None:
        """Online learning: update word vectors from a single sentence.

        Called from teach() to continuously improve word semantics.
        Sense clusters are updated inside train_sentence() (WSD Tier 1).
        """
        self.train_sentence(sentence)
        # Rebuild cache for affected words only (optimization for production)
        self._build_mem_cache()

    def _get_sentence_context_vector(self, sentence: str, target_word: str,
                                      exclude_words: frozenset[str] = None) -> np.ndarray | None:
        """Compute context vector for a target word from a full sentence.

        This is the BEAGLE context computation from Jones & Mewhort (2007):
        context(w) = sum of environmental vectors of all OTHER content words in sentence.

        Key insight from Ruas et al. (2020) and AlMousa et al. (2022):
        Use the FULL SENTENCE context, not just the extracted triple.
        Sentence-level context captures maximum disambiguation signal.

        For WSD teach routing, exclude_words can strip the shared frame
        (entity name, copula, etc.) so only distinguishing content remains.

        Args:
            sentence: The full sentence text
            target_word: The word to compute context for
            exclude_words: Additional words to exclude from context (e.g., shared frame)

        Returns:
            Context vector (float32), or None if sentence has <2 content words
        """
        tokens = tokenize(sentence)
        words = content_words(tokens)
        if len(words) < 2:
            return None

        # Ensure all words have environmental vectors
        for w in words:
            self._get_or_create_env(w)

        # Context for target_word = sum of OTHER content words' env vectors
        # Exclude target word AND any additional exclude_words
        target = target_word.lower()
        exclude = exclude_words or frozenset()
        context = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            if w != target and w not in exclude:
                context += self.env_hvs[w].astype(np.float32)
        return context

    def get_dependency_context_vector(self, sentence: str, target_word: str,
                                       nlp=None) -> np.ndarray | None:
        """Compute context vector using SpaCy dependency-based neighbor extraction.

        Instead of ALL content words (noisy), extract only syntactically
        connected words: the target's head, the head's subject/object,
        and the target's children. This is more precise than bag-of-words
        and matches the "neighbour word analysis" from Sumanathilaka et al. (2026).

        Algorithm:
            1. Find the target word in the SpaCy dependency tree
            2. Get its head word (the word it depends on)
            3. Get the head's subject (nsubj) — WHO does the action
            4. Get the head's other objects — WHAT is affected
            5. Get the target's children (modifiers, determiners)
            6. Compute BEAGLE context from these syntactic neighbors only

        Example:
            "the prisoner was locked in the cell overnight"
            target = "cell"
            → head = "locked" (via "in")
            → head's subject = "prisoner"
            → context = env(locked) + env(prisoner) + env(overnight)

            "the cell contains organelles"
            target = "cell"
            → head = "contains"
            → head's object = "organelles"
            → context = env(contains) + env(organelles)

        Reference: Sumanathilaka et al. (2026), 'EAD Framework':
            'neighbour word analysis is the critical disambiguation signal'
            'a fixed-size context window of up to 10 tokens on each side'

        Args:
            sentence: The full sentence text
            target_word: The word to compute context for
            nlp: SpaCy model (optional, uses lazy loading if None)

        Returns:
            Context vector (float32), or None if SpaCy unavailable or
            target not found in dependency tree
        """
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                return None

        doc = nlp(sentence)
        target = target_word.lower()

        # Find the target token in the dependency tree
        target_token = None
        for token in doc:
            if token.text.lower() == target:
                target_token = token
                break

        if target_token is None:
            return None

        # Collect syntactic neighbors
        neighbor_words = set()

        # 1. Head word (what the target depends on)
        head = target_token.head
        if head.text.lower() != target:
            neighbor_words.add(head.text.lower())

        # 2. Head's subject (WHO does the action)
        for child in head.children:
            if child.dep_ in ("nsubj", "nsubjpass") and child.text.lower() != target:
                neighbor_words.add(child.text.lower())

        # 3. Head's objects (WHAT is affected)
        for child in head.children:
            if child.dep_ in ("dobj", "attr", "acomp") and child.text.lower() != target:
                neighbor_words.add(child.text.lower())

        # 4. Preposition chain (for "locked in the cell" → "locked")
        if target_token.dep_ == "pobj":
            prep = target_token.head  # the preposition
            if prep.dep_ == "prep":
                verb = prep.head  # the verb
                neighbor_words.add(verb.text.lower())
                # Get verb's subject
                for child in verb.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child.text.lower() != target:
                        neighbor_words.add(child.text.lower())

        # 5. Target's children (modifiers, adjectives)
        for child in target_token.children:
            if child.dep_ in ("amod", "compound", "nummod") and child.text.lower() != target:
                neighbor_words.add(child.text.lower())

        # Remove stopwords and the target itself
        neighbor_words.discard(target)
        neighbor_words = {w for w in neighbor_words if w not in STOP_WORDS}

        if not neighbor_words:
            return None

        # Ensure all words have environmental vectors
        for w in neighbor_words:
            self._get_or_create_env(w)

        # Compute context from syntactic neighbors only
        context = np.zeros(self.dim, dtype=np.float32)
        for w in neighbor_words:
            context += self.env_hvs[w].astype(np.float32)

        return context

    def _assign_to_cluster(self, word: str, context_vec: np.ndarray,
                           sentence: str) -> int:
        """Assign a context vector to a sense cluster for a word.

        Algorithm (from CRHCL 2025, Sumanathilaka et al. 2026, Melamud et al. 2016):
        1. Compare context_vec with all existing clusters' context vectors
        2. If best_sim > SENSE_CLUSTER_NEW_THRESHOLD → assign to existing cluster
        3. Otherwise → create new cluster

        Anti-fragmentation (from ART / Grossberg 1976):
        After assignment, check if any two clusters are too similar (>MERGE_THRESHOLD).
        If so, merge them.

        Args:
            word: The word being disambiguated
            context_vec: The sentence-level context vector
            sentence: The original sentence (stored as example)

        Returns:
            Index of the assigned cluster
        """
        word = word.lower()
        if word not in self.sense_clusters:
            self.sense_clusters[word] = []

        clusters = self.sense_clusters[word]

        if not clusters:
            # First sense: create first cluster
            clusters.append((context_vec.copy(), 1, sentence))
            return 0

        # Find best-matching cluster via cosine similarity
        best_sim = -1.0
        best_idx = -1
        ctx_norm = context_vec / (np.linalg.norm(context_vec) + 1e-10)

        for i, (cluster_vec, count, _) in enumerate(clusters):
            cluster_norm = cluster_vec / (np.linalg.norm(cluster_vec) + 1e-10)
            sim = float(np.dot(ctx_norm, cluster_norm))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.SENSE_CLUSTER_NEW_THRESHOLD:
            # Assign to existing cluster: running average update
            # Weighted average preserves the cluster centroid while
            # incorporating new evidence (Jones & Mewhort, 2007)
            old_vec, count, example = clusters[best_idx]
            new_count = count + 1
            # Running average: new_vec = (old_vec * count + context_vec) / (count + 1)
            updated_vec = (old_vec * count + context_vec) / new_count
            clusters[best_idx] = (updated_vec, new_count, example)
            return best_idx
        else:
            # New sense: create new cluster
            clusters.append((context_vec.copy(), 1, sentence))
            # Anti-fragmentation: check if any clusters should be merged
            self._check_sense_merge(word)
            return len(clusters) - 1

    def _check_sense_merge(self, word: str) -> None:
        """Merge clusters that are too similar (anti-fragmentation).

        If two clusters have cosine similarity > SENSE_CLUSTER_MERGE_THRESHOLD,
        they represent the same sense and should be merged.

        This prevents over-fragmentation from slight context variations.
        Reference: Adaptive Resonance Theory (Grossberg, 1976) —
        vigilance parameter controls cluster granularity.
        """
        clusters = self.sense_clusters.get(word, [])
        if len(clusters) < 2:
            return

        # Check all pairs (small N, typically 2-5 clusters per word)
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                vec_i, count_i, ex_i = clusters[i]
                vec_j, count_j, ex_j = clusters[j]

                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i < 1e-10 or norm_j < 1e-10:
                    j += 1
                    continue

                sim = float(np.dot(vec_i / norm_i, vec_j / norm_j))
                if sim > self.SENSE_CLUSTER_MERGE_THRESHOLD:
                    # Merge: weighted average of centroids, sum counts
                    total = count_i + count_j
                    merged_vec = (vec_i * count_i + vec_j * count_j) / total
                    clusters[i] = (merged_vec, total, ex_i)
                    clusters.pop(j)
                    # Don't increment j — it was removed
                else:
                    j += 1
            i += 1

    def get_sense(self, word: str, context_sentence: str) -> int:
        """Get the sense cluster index for a word given its context.

        This is the WSD routing function. Given a word and its full sentence
        context, returns which sense cluster it belongs to.

        Args:
            word: The polysemous word
            context_sentence: The full sentence containing the word

        Returns:
            Sense cluster index (0 = first/default sense)
        """
        word = word.lower()
        if word not in self.sense_clusters:
            return 0  # No clusters yet, default sense

        clusters = self.sense_clusters[word]
        if len(clusters) <= 1:
            return 0  # Only one sense, use it

        # Compute context vector for this word in this sentence
        ctx = self._get_sentence_context_vector(context_sentence, word)
        if ctx is None:
            return 0  # Can't compute context, use default

        # Find best-matching cluster
        ctx_norm = ctx / (np.linalg.norm(ctx) + 1e-10)
        best_sim = -1.0
        best_idx = 0

        for i, (cluster_vec, _, _) in enumerate(clusters):
            norm = np.linalg.norm(cluster_vec)
            if norm < 1e-10:
                continue
            cluster_norm = cluster_vec / norm
            sim = float(np.dot(ctx_norm, cluster_norm))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx

    def get_sense_count(self, word: str) -> int:
        """Get the number of sense clusters for a word."""
        return len(self.sense_clusters.get(word.lower(), []))

    def get_sense_info(self, word: str) -> list[dict]:
        """Get information about all sense clusters for a word.

        Returns list of dicts with cluster metadata.
        """
        word = word.lower()
        clusters = self.sense_clusters.get(word, [])
        return [
            {"index": i, "count": count, "example": example[:80]}
            for i, (_, count, example) in enumerate(clusters)
        ]

    def similarity(self, word1: str, word2: str) -> float:
        """Cosine similarity between two words' memory vectors."""
        v1 = self.get(word1)
        v2 = self.get(word2)
        if v1 is None or v2 is None:
            return 0.0
        return similarity(v1, v2)

    def nearest_neighbors(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find nearest neighbors of a word by HDC similarity."""
        vec = self.get(word)
        if vec is None:
            return []
        results = []
        for w in self._mem_cache:
            if w == word:
                continue
            results.append((w, similarity(vec, self._mem_cache[w])))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def encode_query(self, text: str) -> np.ndarray:
        """Encode text as position-independent bag of content word memory vectors.

        This is the MHN storage/retrieval key. Paraphrases with the same
        content words produce the same vector regardless of word order.

        "what is the capital of france" → bundle(capital_mem, france_mem)
        "capital of france?"             → bundle(capital_mem, france_mem)
        "france's capital"               → bundle(capital_mem, france_mem)
        """
        tokens = tokenize(text)
        words = content_words(tokens)
        if not words:
            return generate_hypervector(self.dim)

        vectors = [self.get_or_random(w) for w in words]
        return normalize_hdc(bundle(*vectors))

    def save(self, path: str) -> None:
        """Save model to file (includes sense clusters for WSD)."""
        if self._cache_dirty:
            self._build_mem_cache()
        data = {
            "dim": self.dim,
            "env_hvs": self.env_hvs,
            "ctx_hvs": self.ctx_hvs,
            "word_freq": dict(self.word_freq),
            "trained_sentences": self.trained_sentences,
            "sense_clusters": self.sense_clusters,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load model from file (includes sense clusters for WSD)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.dim = data["dim"]
        self.env_hvs = data["env_hvs"]
        self.ctx_hvs = data["ctx_hvs"]
        self.word_freq = defaultdict(int, data["word_freq"])
        self.trained_sentences = data["trained_sentences"]
        # Load sense clusters (backward compatible: missing key = empty)
        self.sense_clusters = data.get("sense_clusters", {})
        self._cache_dirty = True
        self._build_mem_cache()

    def stats(self) -> dict:
        return {
            "dim": self.dim,
            "vocab_size": len(self.env_hvs),
            "trained_sentences": self.trained_sentences,
            "memory_kb": len(self.env_hvs) * self.dim // 1024,
        }
