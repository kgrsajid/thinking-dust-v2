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

        The FULL sentence is used (not just the triple), giving richer
        gloss words for later matching. This is the Extended Lesk
        approach (Banerjee & Pedersen, 2002).

        Args:
            entity: The polysemous word (e.g., "cell")
            sense_idx: Which sense (0 = first/default, 1 = second, etc.)
            sentence: The teach() sentence (full, not triple-form)
        """
        entity = entity.lower()
        words = self._extract_context_words(sentence, entity)

        if entity not in self.sense_glosses:
            self.sense_glosses[entity] = []

        # Extend list if needed
        while len(self.sense_glosses[entity]) <= sense_idx:
            self.sense_glosses[entity].append(Counter())

        # Add lemmatized words AND raw words (for better matching)
        self.sense_glosses[entity][sense_idx].update(words)
        # Also add raw content words (handles cases where lemmatization
        # strips useful suffixes like "prisoner" → "prison")
        tokens = tokenize(sentence)
        raw_words = content_words(tokens)
        frame = {entity, "is", "are", "was", "were", "be", "been",
                 "the", "a", "an", "has", "have", "had", "do", "does",
                 "is_a", "part", "part_of", "type", "kind", "sort"}
        raw = [w for w in raw_words if w not in frame]
        self.sense_glosses[entity][sense_idx].update(raw)

    def resolve_sense_with_fact(self, entity: str, sentence: str,
                                relation: str, obj: str,
                                wvm=None, similarity_threshold: float = 0.3) -> int:
        """Resolve sense using both sentence context AND the fact's object.

        The fact's object is often the strongest sense signal:
        "cell part_of prison" → "prison" is the key word.
        "cell is_a device" → "device" is the key word.

        Args:
            entity: The polysemous word
            sentence: The full teach sentence
            relation: The relation (e.g., "is_a", "part_of")
            obj: The object of the triple (e.g., "prison", "device")
            wvm: WordVectorModel for BEAGLE similarity
            similarity_threshold: BEAGLE threshold

        Returns:
            Sense index, or -1 if no signal
        """
        entity = entity.lower()
        obj = obj.lower()
        if entity not in self.sense_glosses:
            return 0

        glosses = self.sense_glosses[entity]
        if len(glosses) <= 1:
            return 0

        # Build context: sentence words + fact's object
        context_words = self._extract_context_words(sentence, entity)
        # The object is the STRONGEST signal — add it explicitly
        context_words.extend(self._extract_context_words(obj, entity))
        if not context_words:
            return 0

        context_set = set(context_words)

        best_sense = 0
        best_score = -1.0
        any_signal = False

        for idx, gloss_counter in enumerate(glosses):
            gloss_set = set(gloss_counter.keys())

            # Exact overlap
            exact = len(context_set & gloss_set)
            weighted = sum(
                min(context_words.count(w), gloss_counter[w])
                for w in context_set & gloss_set
            )
            score = float(exact + weighted * 0.5)

            if score > 0:
                any_signal = True

            # BEAGLE fallback
            if score == 0 and wvm is not None:
                sim_score = 0.0
                for ctx_word in context_set:
                    if ctx_word not in wvm.env_hvs:
                        continue
                    for gloss_word in gloss_set:
                        if gloss_word not in wvm.env_hvs:
                            continue
                        sim = wvm.similarity(ctx_word, gloss_word)
                        if sim >= similarity_threshold:
                            sim_score += sim
                if sim_score > 0:
                    score = sim_score
                    any_signal = True

            if score > best_score:
                best_score = score
                best_sense = idx

        if not any_signal:
            return -1
        return best_sense

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

        Uses lemmatization to handle morphological variants.
        Also excludes common structural/frame words that are NOT
        sense discriminators (is_a, part, type, etc.)
        """
        tokens = tokenize(sentence)
        words = content_words(tokens)
        # Exclude entity + frame words (not sense-discriminating)
        frame_words = {exclude, "is", "are", "was", "were", "be", "been",
                       "the", "a", "an", "has", "have", "had", "do", "does",
                       # Structural words that appear in ALL senses
                       "is_a", "part", "part_of", "type", "kind", "sort"}
        return [_lemmatize(w) for w in words if w not in frame_words]

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
