# TD v2 — Contextualized Embeddings Comparative Tech Spec

**Date:** 2026-07-08
**Status:** REVISED — 4 modules (HDC binding dropped) + Lesk hybrid baseline
**Purpose:** Find the best lightweight contextualized embedding approach for TD v2's WSD

---

## 0. Current Status

### What's Working Now

| Component | Status | Evidence |
|-----------|--------|----------|
| **Lesk WSD (standalone)** | ✅ 100% precision | 32/32 on 55-instance benchmark, 0 false positives |
| **WSD Benchmark** | ✅ 55 instances | 5 words, 15 senses, natural language |
| **Sense clusters (BEAGLE)** | ✅ Working | Per-word context clustering |
| **is_a object routing** | ✅ Working | cell(3), bank(2), apple(2), mercury(2), python(2) |
| **Lesk + teach() integration** | ⚠️ Gloss contamination | First few teaches go to base entity before senses created → gloss for sense 0 has words from all senses |

### Known Limitation: Teach() Gloss Contamination

When teaching multiple senses of a word, the first few facts go to the base entity (no senses yet). When senses are created later, the Lesk gloss for sense 0 contains words from ALL early teaches. The `_rebuild_lesk_glosses()` method mitigates this by rebuilding from triples after sense creation, but the routing for the first few facts is still affected.

**Impact:** First 2-3 teaches per entity may route incorrectly. After senses are established, routing works correctly.

**Fix needed:** Pre-create senses before teaching facts (require `is_a` declaration first), or use a two-pass approach (collect all facts, then route).

### What's Planned (4 Approaches + Lesk)

| # | Approach | Status | Params |
|---|----------|--------|--------|
| A | AERC (with dim reduction) | Planned | ~50K |
| B | NG-RC (with dim reduction) | Planned | ~57K |
| C | Pseudo-BiLM (real backward LSTM) | Planned | ~42K |
| D | Stacked BiLSTM (targeted attention) | Planned | ~85K |
| E | ~~HDC Binding~~ | **DROPPED** | — |
| Baseline | **Lesk WSD** | ✅ Working | 0 |

---

## 1. Problem Statement

BEAGLE (Jones & Mewhort, 2007) is a **static** word vector model — one vector per word regardless of context. For WSD, we need **contextualized** representations: "cell" should have different vectors in "prisoner locked in cell" vs "cell contains organelles" vs "cell phone has touchscreen."

BERT gives contextualized embeddings but requires 110M+ params and GPU. TD v2's constraints: **<100K params, CPU-only, no pretraining, no GPU.**

We implement 5 approaches as separate modules, benchmark them, and pick the winner.

---

## 2. The 5 Approaches

| # | Approach | Paper | Year | Params | CPU? | Key Idea |
|---|----------|-------|------|--------|------|----------|
| A | **AERC** (Attention-Enhanced Reservoir Computing) | Köster & Ito | 2025 | 15K–150K | ✅ | Fixed random reservoir + trained attention readout |
| B | **NG-RC** (Next Generation Reservoir Computing) | Gauthier et al. (Nature Comms) | 2021 | ~5K–50K | ✅ | No reservoir — nonlinear vector autoregression directly |
| C | **Pseudo-BiLM** (Pseudo-Bidirectional LM) | Goto et al. | 2024 | ~10K–30K | ✅ | Small backward LM concatenated with frozen forward |
| D | **Stacked BiLSTM + Attention** | Laatar et al. | 2023 | ~20K–80K | ✅ | Bidirectional LSTM with self-attention for WSD |
| E | **HDC Context Binding** | TD v2 original | — | ~0 | ✅ | Bind word with position via HDC algebra |

---

## 3. Approach A: AERC (Attention-Enhanced Reservoir Computing)

**Paper:** Köster & Ito (2025), "Reservoir Computing as a Language Model"
**Venue:** arXiv:2507.15779

### 3.1 Core Formula

**Reservoir state evolution:**
```
r_t = tanh(r_{t-1} · W_res + x_t · W_in)
```
Where:
- `r_t` ∈ ℝ^N: reservoir state at time t
- `W_res` ∈ ℝ^(N×N): fixed random recurrent matrix
- `W_in` ∈ ℝ^(N×d): fixed random input matrix
- `x_t` ∈ ℝ^d: input embedding at time t
- N = reservoir size, d = embedding dimension

**Attention-enhanced readout:**
```
W_att,l = F(W_net, r_l)         # Attention weights from reservoir state
r_o_l = W_att,l · r_l           # Project reservoir state to hidden
y_l = W_out · r_o_l             # Map to output logits
```

Where:
- `F` = single hidden layer neural network with ReLU
- `W_net` = trainable parameters (the ONLY trained weights)
- `H_o` = hidden dimension of attention layer

### 3.2 Trainable Parameters

For reservoir size N and attention hidden dim H:
```
params = N × H + H × H + H + H_o × V
```

Example configurations (from paper):

| N | H | V | Params |
|---|---|---|--------|
| 256 | 64 | 59 | ~20K |
| 512 | 128 | 59 | ~75K |
| 1024 | 256 | 59 | ~300K |

### 3.3 TD v2 Adaptation

```python
# td/contextual/aerc.py

class AERCContextualizer:
    """Attention-Enhanced Reservoir Computing for contextualized word vectors."""

    def __init__(self, dim=10000, reservoir_size=512, attention_hidden=128):
        self.dim = dim
        self.N = reservoir_size
        self.H = attention_hidden

        # Fixed random reservoir (NOT trained)
        self.W_res = np.random.randn(reservoir_size, reservoir_size) * 0.1
        self.W_in = np.random.randn(reservoir_size, dim) * 0.1

        # Trained attention layer
        self.W_net = np.random.randn(attention_hidden, reservoir_size) * 0.01
        self.W_out = np.random.randn(dim, attention_hidden) * 0.01

    def get_context_vector(self, sentence: str, target_word: str,
                           wvm: WordVectorModel) -> np.ndarray:
        """Get contextualized vector for target_word in sentence."""
        tokens = tokenize(sentence)
        states = []

        # Run reservoir through sentence
        r = np.zeros(self.N)
        for token in tokens:
            x = wvm.get_or_random(token)
            r = np.tanh(r @ self.W_res + x @ self.W_in)
            states.append(r.copy())

        # Find target word position
        target_idx = None
        for i, t in enumerate(tokens):
            if t == target_word.lower():
                target_idx = i
                break
        if target_idx is None:
            return wvm.get_or_random(target_word)

        # Apply attention at target position
        r_target = states[target_idx]
        r_hidden = np.maximum(0, self.W_net @ r_target)  # ReLU
        context_vec = self.W_out @ r_hidden
        return context_vec / (np.linalg.norm(context_vec) + 1e-10)

    def train_step(self, sentence: str, target_word: str,
                   correct_sense: int, wvm: WordVectorModel, lr=0.001):
        """Train attention layer on a single (sentence, word, sense) triple."""
        # Forward pass
        context_vec = self.get_context_vector(sentence, target_word, wvm)

        # Loss: cosine similarity with sense centroid
        # (simplified — full implementation uses cross-entropy)
        # Backprop through W_out and W_net only
        pass  # Implementation details in Phase 1
```

### 3.4 Why It Might Work for TD v2

- **Existing infrastructure:** TD v2 already has a CA reservoir (Rule 90, 64-bit, 16 steps)
- **Minimal new params:** Only the attention layer is trained (~5K–30K)
- **CPU-only:** Matrix multiply + tanh, no GPU needed
- **Proven:** Paper shows AERC approaches transformer performance on language modeling

### 3.5 Risk

- CA reservoir (Rule 90) may not be as effective as the tanh reservoir in the paper
- Need to tune reservoir size and attention hidden dim
- Training requires labeled sense data (from teach() interactions)

---

## 4. Approach B: NG-RC (Next Generation Reservoir Computing)

**Paper:** Gauthier et al. (2021), "Next generation reservoir computing"
**Venue:** Nature Communications, 12, 5581

### 4.1 Core Formula

**No reservoir needed.** The feature vector is built directly from time-delay observations:

**Linear features:**
```
O_lin,i = X_i ⊕ X_{i-s} ⊕ X_{i-2s} ⊕ ... ⊕ X_{i-(k-1)s}
```

Where:
- `X_i` = input at time step i
- k = number of time-delay steps (typically 2–5)
- s = skip size (typically 1)

**Nonlinear features (quadratic):**
```
O_nonlin = O_lin ⊗ O_lin  (outer product)
```

**Total feature vector:**
```
O_total = [1] ⊕ O_lin ⊕ O_nonlin
```

**Output:**
```
Y_{i+1} = W_out · O_total
```

**Training (ridge regression):**
```
W_out = Y_d · O_total^T · (O_total · O_total^T + α·I)^{-1}
```

### 4.2 Trainable Parameters

```
params = d × k + (d × k)^2  (quadratic features)
```

For d=100 (reduced word vector dim), k=3:
```
params = 300 + 90000 = ~90K
```

For d=50, k=2:
```
params = 100 + 10000 = ~10K
```

### 4.3 TD v2 Adaptation

```python
# td/contextual/ng_rc.py

class NGRCContextualizer:
    """Next Generation Reservoir Computing for contextualized word vectors."""

    def __init__(self, dim=100, k=3, s=1):
        self.dim = dim      # Reduced dimension for word vectors
        self.k = k          # Number of time-delay steps
        self.s = s          # Skip size
        self.W_out = None   # Trained via ridge regression
        self.alpha = 0.01   # Regularization parameter

    def _build_feature_vector(self, word_vectors: list[np.ndarray]) -> np.ndarray:
        """Build feature vector from sequence of word vectors."""
        # Linear features: concatenate k time-delayed vectors
        lin_parts = []
        for i in range(self.k):
            idx = -(i * self.s)  # Most recent first
            if abs(idx) <= len(word_vectors):
                lin_parts.append(word_vectors[idx])
            else:
                lin_parts.append(np.zeros(self.dim))
        O_lin = np.concatenate(lin_parts)

        # Nonlinear features: quadratic (outer product upper triangle)
        O_nonlin = np.outer(O_lin, O_lin)[np.triu_indices(len(O_lin))]

        return np.concatenate([[1.0], O_lin, O_nonlin])

    def get_context_vector(self, sentence: str, target_word: str,
                           wvm: WordVectorModel) -> np.ndarray:
        """Get contextualized vector for target_word in sentence."""
        tokens = tokenize(sentence)
        word_vecs = [wvm.get_or_random(t) for t in tokens]

        # Find target position
        target_idx = None
        for i, t in enumerate(tokens):
            if t == target_word.lower():
                target_idx = i
                break
        if target_idx is None:
            return wvm.get_or_random(target_word)

        # Build feature vector from context window around target
        start = max(0, target_idx - self.k * self.s)
        context_vecs = word_vecs[start:target_idx + 1]
        features = self._build_feature_vector(context_vecs)

        # Apply trained weights
        if self.W_out is not None:
            return self.W_out @ features
        return features[:self.dim]  # Fallback: linear features only

    def train(self, training_data: list[tuple[str, str, int]]):
        """Train via ridge regression on (sentence, word, sense) triples."""
        X = []  # Feature vectors
        Y = []  # Target sense vectors

        for sentence, word, sense_idx in training_data:
            tokens = tokenize(sentence)
            word_vecs = [wvm.get_or_random(t) for t in tokens]
            for i, t in enumerate(tokens):
                if t == word.lower():
                    features = self._build_feature_vector(word_vecs[:i+1])
                    X.append(features)
                    Y.append(sense_idx)  # One-hot or sense centroid

        X = np.array(X)
        Y = np.array(Y)

        # Ridge regression: W_out = Y · X^T · (X · X^T + α·I)^{-1}
        self.W_out = Y.T @ X.T @ np.linalg.inv(X @ X.T + self.alpha * np.eye(X.shape[0]))
```

### 4.4 Why It Might Work for TD v2

- **No random matrices:** Eliminates the "good vs bad reservoir" problem
- **Smallest parameter count:** ~10K for typical configurations
- **Ridge regression training:** Closed-form solution, no gradient descent needed
- **Fastest training:** Matrix inversion, not iterative optimization
- **Nature Communications validation:** Proven on chaotic systems benchmarks

### 4.5 Risk

- Designed for time-series forecasting, not NLP — may need adaptation
- Quadratic features scale as O(d²k²) — may be large for high-dim vectors
- Warm-up period needed (k time steps before first prediction)

---

## 5. Approach C: Pseudo-BiLM (Pseudo-Bidirectional Language Model)

**Paper:** Goto et al. (2024), "Pseudo-Bidirectional Representation from Large Unidirectional Language Models"
**Key finding:** Concatenating a small backward LM's hidden states to a frozen forward LM yields +10 F1 on NER, surpassing BERT in few-shot settings.

### 5.1 Core Formula

**Forward pass (existing BEAGLE):**
```
h_fwd = BEAGLE_context(word, left_context)
```

**Backward pass (small LSTM):**
```
h_bwd = LSTM_reverse(word, right_context)
```

**Pseudo-bidirectional representation:**
```
h_pseudo = [h_fwd ; h_bwd]  (concatenation)
```

### 5.2 Trainable Parameters

Only the backward LSTM is trained:
```
params = 4 × (d × h + h × h + h)  # LSTM gates
```

For d=100 (input dim), h=64 (hidden dim):
```
params = 4 × (100×64 + 64×64 + 64) = 4 × (6400 + 4096 + 64) = ~42K
```

### 5.3 TD v2 Adaptation

```python
# td/contextual/pseudo_bilm.py

class PseudoBiLMContextualizer:
    """Pseudo-Bidirectional LM for contextualized word vectors.

    Forward: existing BEAGLE context vector (frozen)
    Backward: small LSTM processing sentence in reverse (trained)
    Output: concatenation of both

    Reference: Goto et al. (2024) — +10 F1 over forward-only, surpasses BERT
    """

    def __init__(self, dim=100, hidden_dim=64):
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Backward LSTM parameters (trained)
        self.W_ih = np.random.randn(4 * hidden_dim, dim) * 0.01
        self.W_hh = np.random.randn(4 * hidden_dim, hidden_dim) * 0.01
        self.b = np.zeros(4 * hidden_dim)

    def _lstm_backward(self, word_vecs: list[np.ndarray]) -> list[np.ndarray]:
        """Run LSTM backward through sentence."""
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        states = []

        for x in reversed(word_vecs):
            gates = self.W_ih @ x + self.W_hh @ h + self.b
            i, f, g, o = np.split(gates, 4)
            i = 1 / (1 + np.exp(-i))    # sigmoid
            f = 1 / (1 + np.exp(-f))
            o = 1 / (1 + np.exp(-o))
            g = np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
            states.append(h.copy())

        states.reverse()  # Back to forward order
        return states

    def get_context_vector(self, sentence: str, target_word: str,
                           wvm: WordVectorModel) -> np.ndarray:
        """Get pseudo-bidirectional context vector."""
        tokens = tokenize(sentence)
        word_vecs = [wvm.get_or_random(t) for t in tokens]

        # Forward: BEAGLE context (existing, frozen)
        h_fwd = wvm._get_sentence_context_vector(sentence, target_word)
        if h_fwd is None:
            h_fwd = np.zeros(self.dim)

        # Backward: LSTM (trained)
        bwd_states = self._lstm_backward(word_vecs)

        # Find target position
        target_idx = None
        for i, t in enumerate(tokens):
            if t == target_word.lower():
                target_idx = i
                break
        if target_idx is None:
            h_bwd = np.zeros(self.hidden_dim)
        else:
            h_bwd = bwd_states[target_idx]

        # Concatenate forward + backward
        h_pseudo = np.concatenate([h_fwd[:self.dim], h_bwd])
        return h_pseudo / (np.linalg.norm(h_pseudo) + 1e-10)

    def train_step(self, sentence: str, target_word: str,
                   target_sense: int, lr=0.001):
        """Train backward LSTM on a single example."""
        # Backprop through LSTM gates
        # (simplified — full implementation in Phase 1)
        pass
```

### 5.4 Why It Might Work for TD v2

- **Proven to surpass BERT:** +10 F1 on NER with just a small backward LM
- **Forward pass is free:** Uses existing BEAGLE vectors (no retraining)
- **Only backward LSTM is trained:** ~42K params
- **Few-shot friendly:** Works well with limited training data
- **CPU-friendly:** LSTM inference is sequential but fast for short sentences

### 5.5 Risk

- LSTM is sequential — slower than reservoir for long sentences
- Backward LSTM needs training data (from teach() interactions)
- Concatenation doubles the vector dimension

---

## 6. Approach D: Stacked BiLSTM + Attention

**Paper:** Laatar et al. (2023), "Attention-based Stacked BiLSTM for WSD"
**Key finding:** Stacked bidirectional LSTM with self-attention outperforms other WSD methods on SemEval-2007.

### 6.1 Core Formula

**Bidirectional LSTM:**
```
h_fwd_t = LSTM_fwd(x_t, h_fwd_{t-1})
h_bwd_t = LSTM_bwd(x_t, h_bwd_{t+1})
h_t = [h_fwd_t ; h_bwd_t]  (concatenation)
```

**Stacked (2 layers):**
```
h^(1)_t = BiLSTM_layer1(x_t)
h^(2)_t = BiLSTM_layer2(h^(1)_t)
```

**Self-attention:**
```
α_t = softmax(W_attn · h^(2)_t + b_attn)
context = Σ_t (α_t · h^(2)_t)
```

### 6.2 Trainable Parameters

For 2-layer BiLSTM with hidden dim h and attention:
```
params = 2 × 4 × (d × h + h × h + h) + h × 1 + 1
```

For d=100, h=64:
```
params = 2 × 4 × (6400 + 4096 + 64) + 64 + 1 = ~85K
```

### 6.3 TD v2 Adaptation

```python
# td/contextual/stacked_bilstm.py

class StackedBiLSTMContextualizer:
    """Stacked BiLSTM with self-attention for WSD.

    Reference: Laatar et al. (2023) — state-of-the-art on SemEval-2007 WSD
    """

    def __init__(self, dim=100, hidden_dim=64, n_layers=2):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Layer 1: forward + backward LSTM
        self.fwd_lstm1 = LSTMLayer(dim, hidden_dim)
        self.bwd_lstm1 = LSTMLayer(dim, hidden_dim)

        # Layer 2: forward + backward LSTM
        self.fwd_lstm2 = LSTMLayer(2 * hidden_dim, hidden_dim)
        self.bwd_lstm2 = LSTMLayer(2 * hidden_dim, hidden_dim)

        # Self-attention
        self.W_attn = np.random.randn(2 * hidden_dim) * 0.01
        self.b_attn = 0.0

    def get_context_vector(self, sentence: str, target_word: str,
                           wvm: WordVectorModel) -> np.ndarray:
        """Get attention-weighted contextualized vector."""
        tokens = tokenize(sentence)
        word_vecs = [wvm.get_or_random(t) for t in tokens]

        # Layer 1: BiLSTM
        fwd1 = self.fwd_lstm1.forward(word_vecs)
        bwd1 = self.bwd_lstm1.backward(word_vecs)
        h1 = [np.concatenate([f, b]) for f, b in zip(fwd1, bwd1)]

        # Layer 2: BiLSTM
        fwd2 = self.fwd_lstm2.forward(h1)
        bwd2 = self.bwd_lstm2.backward(h1)
        h2 = [np.concatenate([f, b]) for f, b in zip(fwd2, bwd2)]

        # Self-attention weights
        attn_weights = np.array([np.dot(self.W_attn, h) + self.b_attn for h in h2])
        attn_weights = np.exp(attn_weights - np.max(attn_weights))
        attn_weights /= attn_weights.sum()

        # Weighted sum
        context = sum(a * h for a, h in zip(attn_weights, h2))
        return context / (np.linalg.norm(context) + 1e-10)
```

### 6.4 Why It Might Work for TD v2

- **Purpose-built for WSD:** The paper specifically targets word sense disambiguation
- **Attention highlights relevant context:** Self-attention learns which context words matter
- **Stacked layers capture syntax + semantics:** Lower layers = syntax, higher = semantics (like ELMo)
- **Interpretable:** Attention weights show which words influenced the disambiguation

### 6.5 Risk

- Highest parameter count (~85K) — still under 100K but tight
- Sequential LSTM inference — slower than reservoir approaches
- Needs more training data than reservoir approaches

---

## 7. Approach E: HDC Context Binding

**Based on:** TD v2's existing HDC infrastructure + McInnes et al. (2012) BSC-WSD

### 7.1 Core Formula

**Position-aware context encoding:**
```
C(word, pos) = E(word) ⊗ P(pos)
```

Where:
- `E(word)` = environmental vector for the word
- `P(pos)` = position vector (random, one per position)
- `⊗` = HDC bind (element-wise multiply)

**Sentence context vector:**
```
S = Σ_t C(word_t, pos_t)  (bundle all position-bound words)
```

**Sense extraction via unbinding:**
```
sense(word) = S ⊗ P(pos)^{-1} ⊗ E(word)^{-1}
```

### 7.2 Trainable Parameters

**Zero.** All vectors are random or pre-existing. The only "training" is accumulating context vectors per sense.

### 7.3 TD v2 Adaptation

```python
# td/contextual/hdc_binding.py

class HDCContextBinder:
    """HDC context binding for contextualized word vectors.

    Uses existing TD v2 infrastructure (bind, bundle, permute).
    Zero new parameters — all vectors are random or pre-existing.

    Reference: McInnes et al. (2012) — BSC-WSD, 94.55% accuracy
    Reference: Kanerva (2009) — HDC algebra
    """

    def __init__(self, dim=10000, max_positions=64):
        self.dim = dim
        # Position vectors (random, fixed)
        self.pos_vectors = [generate_hypervector(dim) for _ in range(max_positions)]

    def get_context_vector(self, sentence: str, target_word: str,
                           wvm: WordVectorModel) -> np.ndarray:
        """Get HDC context vector for target word in sentence."""
        tokens = tokenize(sentence)

        # Bind each word with its position
        bound_vectors = []
        for i, token in enumerate(tokens):
            word_vec = wvm.get_or_random(token)
            pos_vec = self.pos_vectors[i % len(self.pos_vectors)]
            bound = bind(word_vec, pos_vec)  # Element-wise multiply
            bound_vectors.append(bound)

        # Bundle all position-bound words
        context = bundle(*bound_vectors)  # Majority vote

        return context

    def get_sense_from_context(self, context: np.ndarray, target_word: str,
                               wvm: WordVectorModel) -> np.ndarray:
        """Extract sense-specific vector by unbinding the target word."""
        word_vec = wvm.get_or_random(target_word)
        # Unbind: context ⊗ word_vec^{-1} = rest of context
        sense = bind(context, word_vec)  # Self-inverse for bipolar
        return sense

    def train_step(self, *args, **kwargs):
        """No training needed — all vectors are random."""
        pass
```

### 7.4 Why It Might Work for TD v2

- **Zero new parameters:** Uses existing HDC infrastructure
- **Fastest:** Element-wise multiply + majority vote, O(dim)
- **BSC-WSD validated:** 94.55% accuracy on clinical WSD (McInnes et al., 2012)
- **No training needed:** Vectors are random, context is accumulated
- **Already integrated:** TD v2 has bind, bundle, permute

### 7.5 Risk

- Position encoding may not capture long-range dependencies
- HDC capacity limits: ~3000 superposed bindings before noise dominates (Kanerva, 2009)
- No learning — can't adapt to specific disambiguation patterns

---

## 8. Comparative Analysis

### 8.1 Parameter Budget

| Approach | Params | % of 100K Budget | Trainable? |
|----------|--------|------------------|------------|
| A. AERC | 5K–30K | 5–30% | Attention only |
| B. NG-RC | 10K–90K | 10–90% | Ridge regression |
| C. Pseudo-BiLM | ~42K | 42% | Backward LSTM |
| D. Stacked BiLSTM | ~85K | 85% | Full model |
| E. HDC Binding | 0 | 0% | None |

### 8.2 Computational Cost (per sentence)

| Approach | Training | Inference | CPU-Friendly? |
|----------|----------|-----------|---------------|
| A. AERC | Backprop through attention | Matrix multiply + tanh | ✅ Fast |
| B. NG-RC | Ridge regression (one-time) | Matrix multiply | ✅ Fastest |
| C. Pseudo-BiLM | LSTM backprop | LSTM forward (sequential) | ✅ Medium |
| D. Stacked BiLSTM | Full backprop | 2× LSTM forward (sequential) | ✅ Slower |
| E. HDC Binding | None | Element-wise multiply | ✅ Fastest |

### 8.3 Context Quality (expected)

| Approach | Left Context | Right Context | Long-Range | Syntactic |
|----------|-------------|---------------|------------|-----------|
| A. AERC | ✅ Full | ✅ Full | ⚠️ Reservoir memory | ⚠️ Limited |
| B. NG-RC | ✅ k steps | ❌ None | ❌ Window only | ❌ No |
| C. Pseudo-BiLM | ✅ BEAGLE | ✅ LSTM | ⚠️ LSTM memory | ⚠️ Limited |
| D. Stacked BiLSTM | ✅ Full | ✅ Full | ⚠️ LSTM memory | ✅ Best |
| E. HDC Binding | ✅ Full | ✅ Full | ❌ Position-bound | ❌ No |

### 8.4 Training Data Requirement

| Approach | Min Examples | Cold Start | Online Learning |
|----------|-------------|------------|-----------------|
| A. AERC | ~50 | ❌ Needs warm-up | ✅ Per-teach |
| B. NG-RC | ~20 | ✅ Few-shot | ⚠️ Ridge recompute |
| C. Pseudo-BiLM | ~100 | ❌ Needs warm-up | ✅ Per-teach |
| D. Stacked BiLSTM | ~200 | ❌ Needs warm-up | ✅ Per-teach |
| E. HDC Binding | 0 | ✅ Instant | ✅ Accumulate |

### 8.5 Integration Complexity

| Approach | New Files | Changes to Existing | Risk |
|----------|-----------|---------------------|------|
| A. AERC | 1 | word_vectors.py | Low |
| B. NG-RC | 1 | word_vectors.py | Low |
| C. Pseudo-BiLM | 1 | word_vectors.py | Low |
| D. Stacked BiLSTM | 2 | word_vectors.py | Medium |
| E. HDC Binding | 1 | None (uses existing) | Lowest |

---

## 9. Implementation Plan

### Phase 1: Module Interface (Day 1)

Create `td/contextual/` directory with a common interface:

```python
# td/contextual/__init__.py

class Contextualizer:
    """Base class for contextualized word vector providers.

    Design decisions (from Gemini code review):
    - Tokenization is the CALLER's responsibility (SRP)
    - WVM injected via constructor (Dependency Injection)
    - Training supports both online (train_step) and batch (finalize)
    - Factory pattern for runtime switching
    """

    def __init__(self, wvm: WordVectorModel, dim: int = 10000):
        self.wvm = wvm
        self.dim = dim

    def get_context_vector(self, tokens: list[str], target_idx: int) -> np.ndarray:
        """Return a contextualized vector for tokens[target_idx].

        Args:
            tokens: Pre-tokenized sentence (caller handles tokenization)
            target_idx: Index of the target word in tokens

        Returns:
            Contextualized vector (dim,)
        """
        raise NotImplementedError

    def train_step(self, tokens: list[str], target_idx: int, sense_label: int):
        """Online training: accumulate a single example.

        For batch methods (NG-RC): accumulates internally.
        Call finalize_training() after all train_steps.
        """
        pass

    def finalize_training(self):
        """Batch training: run optimization on accumulated examples.
        For online methods (AERC, BiLSTM): no-op.
        For batch methods (NG-RC): runs ridge regression.
        """
        pass

    def save(self, path: str): ...
    def load(self, path: str): ...


# Factory pattern for runtime switching
CONTEXTUALIZERS = {}

def register_contextualizer(name: str, cls: type):
    CONTEXTUALIZERS[name] = cls

def create_contextualizer(name: str, wvm: WordVectorModel, **kwargs) -> Contextualizer:
    return CONTEXTUALIZERS[name](wvm=wvm, **kwargs)
```

### Phase 2: Implement All 5 (Days 2–6)

| Day | Module | Lines (est.) |
|-----|--------|-------------|
| 2 | `td/contextual/aerc.py` | ~150 |
| 3 | `td/contextual/ng_rc.py` | ~120 |
| 4 | `td/contextual/pseudo_bilm.py` | ~130 |
| 5 | `td/contextual/stacked_bilstm.py` | ~180 |
| 6 | `td/contextual/hdc_binding.py` | ~80 |

### Phase 3: Benchmark Suite (Day 7)

Create `tests/test_contextualized.py`:

```python
# Test: same word in different contexts → different vectors
# Test: same word in same context → same vector
# Test: different words in same context → different vectors
# Test: WSD accuracy on cell/bank/apple/mercury/python
# Test: training convergence
# Test: inference latency
# Test: parameter count
```

### Phase 4: Integration (Day 8)

Wire the winning contextualizer into `GenericThinkingDust`:

```python
# In thinking.py __init__:
self.contextualizer = AERCContextualizer(...)  # or whichever wins

# In teach() and ask():
ctx_vec = self.contextualizer.get_context_vector(sentence, word, self.wvm)
```

### Phase 5: Documentation (Day 9)

Update ARCHITECTURE.md, DEVELOPMENT.md with:
- Which approach won and why
- Benchmark results
- Integration guide

---

## 10. Benchmark Metrics

| Metric | How to Measure | Target |
|--------|---------------|--------|
| **WSD Accuracy** | % correct sense assignment on test set | >80% |
| **Context Discrimination** | cosine(context_bio, context_prison) vs cosine(context_bio, context_bio) | Separation >0.1 |
| **Inference Latency** | ms per sentence on CPU | <10ms |
| **Parameter Count** | total trainable params | <100K |
| **Training Time** | seconds to converge on 100 examples | <30s |
| **Memory** | MB for model + vectors | <50MB |

---

## 11. Success Criteria

After benchmarking all 5 approaches:

1. **Winner:** The approach with best WSD accuracy × speed tradeoff
2. **Must have:** >70% WSD accuracy on cell/bank/apple/mercury/python
3. **Must have:** <10ms inference latency on CPU
4. **Must have:** <100K trainable parameters
5. **Nice to have:** Online learning (update per teach())
6. **Nice to have:** Zero new parameters (HDC binding)

---

## 12. References

| # | Paper | Year | Venue | Approach |
|---|-------|------|-------|----------|
| 1 | Köster & Ito, "Reservoir Computing as a Language Model" | 2025 | arXiv:2507.15779 | AERC |
| 2 | Gauthier et al., "Next generation reservoir computing" | 2021 | Nature Communications 12:5581 | NG-RC |
| 3 | Goto et al., "Pseudo-Bidirectional Representation" | 2024 | arXiv | Pseudo-BiLM |
| 4 | Laatar et al., "Attention-based Stacked BiLSTM for WSD" | 2023 | ResearchGate | Stacked BiLSTM |
| 5 | McInnes et al., "HDC Approach to WSD" | 2012 | AMIA Symposium | BSC-WSD |
| 6 | Kanerva, "Hyperdimensional Computing" | 2009 | IEEE CIM | HDC algebra |
| 7 | Jones & Mewhort, "BEAGLE" | 2007 | Psychological Review | Static vectors |
| 8 | Peters et al., "ELMo" | 2018 | NAACL | BiLSTM contextualization |

---

## 13. Critical Issues & Fixes (Code Review Findings)

### 13.1 Architecture Fixes (from Gemini review)

| Issue | Fix |
|-------|-----|
| Each contextualizer handles its own tokenization | Move to caller. Interface: `get_context_vector(tokens, target_idx)` |
| Training interface inconsistency (online vs batch) | Add `finalize_training()` for batch methods. `train_step()` accumulates. |
| WVM passed to every call | Inject via constructor (DI) |
| Hardcoded winner in Phase 4 | Factory Pattern: `create_contextualizer("aerc", wvm)` |
| No shape assertions | Add `assert r.shape == (self.N,)` after every matrix op |
| HDC position limit = 64 | Generate position vectors dynamically, cache them |

### 13.2 Technical Risks (from self-review)

| Risk | Impact | Mitigation |
|------|--------|------------|
| **NG-RC quadratic features explode for 10K-dim vectors** | `O(d²k²)` = 100M params for d=10K, k=3 | Dimensionality reduction: random project 10K→100 before NG-RC. Features become 10K instead of 100M. |
| **AERC uses tanh reservoir, TD v2 uses CA (Rule 90)** | Different dynamics, may not transfer | Test both: (a) use tanh reservoir for AERC, (b) adapt CA to AERC-style attention readout |
| **Pseudo-BiLM's forward is static** | BEAGLE doesn't change per-sentence | Accept limitation: backward LSTM provides the contextualization. Forward is "background knowledge." |
| **Cold-start: not enough labeled data** | Early teach() won't have enough (sentence, word, sense) triples | Generate synthetic WSD data from 10K corpus. Label words appearing in multiple domains. |
| **NG-RC designed for time-series, not NLP** | "Time-delay" = context window, but semantics differ | Adapt: use word positions as time steps, not actual time |

### 13.3 New Ideas

| Idea | Source | Description |
|------|--------|-------------|
| **Dimensionality reduction for NG-RC** | Self | Random project 10K→100 before NG-RC. Reduces quadratic features from 100M to 10K. |
| **Hybrid: HDC position + AERC context** | Self | Use HDC bind for position encoding, AERC for context. Best of both. |
| **Synthetic WSD data from corpus** | Self | Label words in 10K corpus that appear in multiple domains (cell, bank, python, mercury). Generate (sentence, word, sense_idx) triples automatically. |
| **Ensemble: vote across approaches** | Gemini | Don't pick one winner — ensemble the top 2-3 for robustness. |

---

## 14. Updated Implementation Plan

### Phase 0: Data Preparation (Day 0)

Generate synthetic WSD training data from the 10K corpus:

```python
# Identify polysemous words in corpus
# Label each occurrence with its domain sense
# Output: list of (sentence, word, sense_idx) triples
```

### Phase 1: Module Interface (Day 1)

Create `td/contextual/` with:
- `__init__.py` — base class + factory
- Common tokenization (caller-side)
- Shape assertion decorator

### Phase 2: Implement All 5 (Days 2–6)

Each module implements:
- `__init__(wvm, dim, **kwargs)` — constructor with DI
- `get_context_vector(tokens, target_idx)` — main method
- `train_step(tokens, target_idx, sense_label)` — online training
- `finalize_training()` — batch finalization (if needed)
- `save(path)` / `load(path)` — persistence

### Phase 3: Benchmark Suite (Day 7)

```python
# Baselines:
#   (a) Static BEAGLE (current)
#   (b) Random context vectors

# Test set:
#   cell (3 senses), bank (2), apple (2), mercury (2), python (2)

# Metrics:
#   - WSD accuracy (% correct sense assignment)
#   - Context discrimination (cosine separation)
#   - Inference latency (ms per sentence)
#   - Parameter count
#   - Training convergence
```

### Phase 4: Integration (Day 8)

```python
# Factory pattern — no hardcoding
self.contextualizer = create_contextualizer(
    config.context_model_type,  # "aerc", "ng_rc", "pseudo_bilm", etc.
    wvm=self.wvm,
    dim=self.dim,
)
```

### Phase 5: Documentation (Day 9)

Update ARCHITECTURE.md, DEVELOPMENT.md with:
- Winner and why
- Benchmark results table
- Integration guide
- Known limitations

---

*"Five roads diverged in a wood, and we — we took the one with best F1."*
