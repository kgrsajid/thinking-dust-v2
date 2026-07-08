"""Lesk-style Word Sense Disambiguation for TD v2.

Zero-parameter WSD using sense definitions built from teach() interactions.
When a user teaches facts about a sense, the teach sentences become the
"gloss" (definition) for that sense. New facts are routed to the best-
matching sense by comparing context words against sense glosses.

This is the Extended Lesk algorithm adapted for TD v2's teach-from-zero
architecture. No WordNet, no pretraining, no parameters.

Algorithm (Simplified Lesk, Vasilescu et al. 2004):
    for each sense of the word:
        signature = set of content words in sense's gloss + examples
        overlap = |context ∩ signature|
    return sense with max overlap

TD v2 adaptation:
    - "gloss" = content words from teach() sentences for each sense
    - "context" = content words from the new teach() sentence
    - Extended: also use dependency neighbors from SpaCy

References:
    - Lesk (1986), "Automatic Sense Disambiguation Using Machine Readable Dictionaries"
    - Banerjee & Pedersen (2002), "Extended Gloss Overlaps for WSD"
    - Vasilescu et al. (2004), "Simplified Lesk with smart default"
    - Kilgarriff & Rosensweig (2000), "English SENSEVAL: The First Evaluation"
"""

from __future__ import annotations

from collections import Counter
from td.perception.word_vectors import tokenize, content_words


def _lemmatize(word: str) -> str:
    """Simple lemmatization: strip common suffixes.

    Not as good as spaCy, but zero-dependency and fast.
    For production: use spaCy token.lemma_.
    """
    w = word.lower()
    # Strip common suffixes (ordered by length, longest first)
    for suffix in ("tion", "sion", "ment", "ness", "ance", "ence",
                   "ing", "ied", "ies", "ers", "est", "ity",
                   "ed", "er", "es", "ly", "al", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


class LeskWSD:
    """Lesk-style Word Sense Disambiguation using teach() glosses.

    Each sense has a "gloss" — a set of content words accumulated from
    teach() sentences that were assigned to that sense. When a new fact
    comes in, its context words are compared against each sense's gloss.
    The sense with the highest overlap wins.

    Zero parameters. Zero training. Works cold-start.
    """

    def __init__(self):
        # entity → sense_idx → Counter of content words (the gloss)
        self.sense_glosses: dict[str, list[Counter]] = {}

    def add_sense_example(self, entity: str, sense_idx: int, sentence: str):
        """Add a teach() sentence as a gloss example for a sense.

        Called when a fact is assigned to a sense. The content words
        from the sentence become part of that sense's gloss.
        Words are lemmatized for morphological matching.

        Args:
            entity: The polysemous word (e.g., "cell")
            sense_idx: Which sense (0 = first/default, 1 = second, etc.)
            sentence: The teach() sentence
        """
        entity = entity.lower()
        words = self._extract_context_words(sentence, entity)

        if entity not in self.sense_glosses:
            self.sense_glosses[entity] = []

        # Extend list if needed
        while len(self.sense_glosses[entity]) <= sense_idx:
            self.sense_glosses[entity].append(Counter())

        # Add lemmatized words to this sense's gloss
        self.sense_glosses[entity][sense_idx].update(words)

    def resolve_sense(self, entity: str, sentence: str,
                      wvm=None, similarity_threshold: float = 0.3) -> int:
        """Resolve which sense a new sentence belongs to.

        Hybrid approach:
        1. Try Lesk (exact word overlap) — high precision when it fires
        2. If Lesk has zero overlap for ALL senses, return -1 (no signal)
           The caller should fall back to sense_clusters or default.

        Args:
            entity: The polysemous word
            sentence: The new teach() sentence
            wvm: WordVectorModel for BEAGLE similarity fallback (optional)
            similarity_threshold: BEAGLE threshold (high = conservative)

        Returns:
            Sense index (0 = first/default), or -1 if no signal
        """
        entity = entity.lower()
        if entity not in self.sense_glosses:
            return 0

        glosses = self.sense_glosses[entity]
        if len(glosses) <= 1:
            return 0

        context_words = self._extract_context_words(sentence, entity)
        if not context_words:
            return 0

        context_set = set(context_words)

        best_sense = 0
        best_score = -1
        any_overlap = False

        for idx, gloss_counter in enumerate(glosses):
            gloss_set = set(gloss_counter.keys())

            # Exact overlap (Simplified Lesk)
            exact_overlap = len(context_set & gloss_set)
            weighted = sum(
                min(context_words.count(w), gloss_counter[w])
                for w in context_set & gloss_set
            )
            score = exact_overlap + weighted * 0.5

            if score > 0:
                any_overlap = True

            # BEAGLE similarity fallback (Extended Lesk)
            # Only used when exact overlap is zero, with HIGH threshold
            # to avoid false matches from noisy BEAGLE vectors
            if score == 0 and wvm is not None:
                sim_score = 0.0
                for ctx_word in context_set:
                    for gloss_word in gloss_set:
                        sim = wvm.similarity(ctx_word, gloss_word)
                        if sim >= similarity_threshold:
                            sim_score += sim
                if sim_score > 0:
                    score = sim_score
                    any_overlap = True

            if score > best_score:
                best_score = score
                best_sense = idx

        if not any_overlap:
            return -1  # No signal — caller should use fallback

        return best_sense

    def get_gloss_words(self, entity: str, sense_idx: int) -> set[str]:
        """Get the gloss words for a specific sense."""
        entity = entity.lower()
        if entity not in self.sense_glosses:
            return set()
        if sense_idx >= len(self.sense_glosses[entity]):
            return set()
        return set(self.sense_glosses[entity][sense_idx].keys())

    def get_sense_count(self, entity: str) -> int:
        """Get the number of senses for an entity."""
        return len(self.sense_glosses.get(entity.lower(), []))

    def _extract_context_words(self, sentence: str, exclude: str) -> list[str]:
        """Extract content words from a sentence, excluding the target entity.

        Uses lemmatization to handle morphological variants:
        "prisoners" → "prison", "cells" → "cell", "escaped" → "escap"
        """
        tokens = tokenize(sentence)
        words = content_words(tokens)
        # Exclude the entity itself and common frame words
        exclude_set = {exclude, "is", "are", "was", "were", "be", "been",
                       "the", "a", "an", "has", "have", "had", "do", "does"}
        # Lemmatize for matching
        return [_lemmatize(w) for w in words if w not in exclude_set]

    def save(self, path: str) -> None:
        """Persist sense glosses to file."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.sense_glosses, f)

    def load(self, path: str) -> None:
        """Load sense glosses from file."""
        import pickle
        with open(path, "rb") as f:
            self.sense_glosses = pickle.load(f)
