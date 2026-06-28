"""Online learning from execution outcomes and user corrections.

Two learning modes:
    1. Outcome-based: Action executed → observed outcome → store attractor
    2. Correction-based: User says "no, do Y instead" → store corrected attractor

Both modes add new patterns to MHN. No gradient updates. No retraining.
"""

from __future__ import annotations

import numpy as np

from ..perception.hdc import ConceptVocabulary, bind, bundle, generate_hypervector
from ..memory.mhn import ModernHopfieldNetwork


class OnlineLearner:
    """Learns from execution outcomes and user corrections.

    The learner stores new patterns in the MHN as attractors:
    - Successful actions → positive attractors (deepened in energy landscape)
    - Failed actions → marked as inactive (avoided in future retrievals)
    - Corrections → old pattern superseded, new pattern stored

    No gradient descent. No backpropagation. Just MHN attractor storage.
    Learning is O(dim) per example — effectively instant.
    """

    def __init__(self, mhn: ModernHopfieldNetwork,
                 vocabulary: ConceptVocabulary):
        """Initialize online learner.

        Args:
            mhn: Modern Hopfield Network to store patterns in.
            vocabulary: Concept vocabulary for encoding outcomes.
        """
        self.mhn = mhn
        self.vocab = vocabulary

    def learn_from_outcome(
        self,
        situation_hdc: np.ndarray,
        action_hdc: np.ndarray,
        outcome: str,
        metadata: dict | None = None,
    ) -> int:
        """Store execution outcome as MHN attractor.

        If success: store (situation, action, "success") as positive attractor.
        If failure: store (situation, action, "failure") as negative attractor
                    (marked inactive so it won't be retrieved, but remains in
                    memory for debugging).

        Args:
            situation_hdc: HDC vector of the situation/context.
            action_hdc: HDC vector of the action taken.
            outcome: "success", "failure", or "partial".
            metadata: Additional metadata.

        Returns:
            Pattern index in MHN.
        """
        meta = metadata or {}
        meta["outcome"] = outcome

        # Store the pattern
        idx = self.mhn.store(situation_hdc, action_hdc, meta)

        if outcome == "failure":
            # Mark failures as inactive so they're not retrieved
            self.mhn.patterns[idx].active = False
            self.mhn._dirty = True

        return idx

    def learn_from_correction(
        self,
        situation_hdc: np.ndarray,
        wrong_action_hdc: np.ndarray,
        correct_action_hdc: np.ndarray,
        metadata: dict | None = None,
    ) -> int:
        """Learn from user correction.

        1. Find the closest stored pattern matching (situation, wrong_action).
        2. Mark it as inactive (superseded).
        3. Store new pattern (situation, correct_action) as positive attractor.

        Args:
            situation_hdc: HDC vector of the situation.
            wrong_action_hdc: HDC vector of the wrong action.
            correct_action_hdc: HDC vector of the correct action.
            metadata: Additional metadata.

        Returns:
            New pattern index in MHN.
        """
        meta = metadata or {}
        meta["corrected"] = True
        meta["outcome"] = "success"

        # Try to find and deactivate the old pattern
        results = self.mhn.retrieve(situation_hdc, top_k=5)
        for value_vec, sim, pat_meta in results:
            # Check if this retrieved value matches the wrong action
            from ..perception.hdc import similarity
            action_sim = similarity(value_vec, wrong_action_hdc)
            if action_sim > 0.5:
                # Found the old pattern — deactivate it
                # We need to find the actual pattern index
                for i, p in enumerate(self.mhn.patterns):
                    if p.active and similarity(p.key, situation_hdc) > 0.5:
                        from ..perception.hdc import similarity as sim_fn
                        if sim_fn(p.value, wrong_action_hdc) > 0.5:
                            self.mhn.patterns[i].active = False
                            self.mhn._dirty = True
                            break

        # Store the corrected pattern
        idx = self.mhn.store(situation_hdc, correct_action_hdc, meta)
        return idx

    def encode_action_plan(self, actions: list[dict]) -> np.ndarray:
        """Encode an action plan as an HDC vector.

        Each action is encoded as a record, then all are bundled.

        Args:
            actions: List of action dicts like:
                [{"action": "click", "target": "submit_button"}]

        Returns:
            HDC vector representing the plan.
        """
        if not actions:
            return generate_hypervector(self.vocab.dim)

        parts = []
        for a in actions:
            parts.append(self.vocab.encode_record(**a))

        if len(parts) == 1:
            return parts[0]

        result = parts[0]
        for p in parts[1:]:
            result = bundle(result, p)
        return result
