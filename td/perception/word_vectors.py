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
    """

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
        """
        self.train_sentence(sentence)
        # Rebuild cache for affected words only (optimization for production)
        self._build_mem_cache()

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
        """Save model to file."""
        if self._cache_dirty:
            self._build_mem_cache()
        data = {
            "dim": self.dim,
            "env_hvs": self.env_hvs,
            "ctx_hvs": self.ctx_hvs,
            "word_freq": dict(self.word_freq),
            "trained_sentences": self.trained_sentences,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.dim = data["dim"]
        self.env_hvs = data["env_hvs"]
        self.ctx_hvs = data["ctx_hvs"]
        self.word_freq = defaultdict(int, data["word_freq"])
        self.trained_sentences = data["trained_sentences"]
        self._cache_dirty = True
        self._build_mem_cache()

    def stats(self) -> dict:
        return {
            "dim": self.dim,
            "vocab_size": len(self.env_hvs),
            "trained_sentences": self.trained_sentences,
            "memory_kb": len(self.env_hvs) * self.dim // 1024,
        }
