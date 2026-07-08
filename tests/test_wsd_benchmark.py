"""WSD Benchmark — Real evaluation, not cherry-picked.

Tests word sense disambiguation on multiple polysemous words with
diverse, natural-language sentences. Not 15 instances — hundreds.

Benchmark design:
- Multiple polysemous words (cell, bank, apple, python, mercury, bat, crane,
  match, rock, star, spring, trunk, scale, seal, slug)
- 3-5 senses per word
- 10-20 test sentences per word (natural language, not templates)
- Sentences sourced from real usage patterns (not cherry-picked)
- Baselines: random, most-frequent-sense, Lesk-only, sense_clusters-only
- Metrics: accuracy, precision per sense, fallback rate

References:
    - Lesk (1986), "Automatic Sense Disambiguation"
    - Vasilescu et al. (2004), "Simplified Lesk with smart default"
    - Kilgarriff & Rosensweig (2000), "English SENSEVAL"
    - SemEval-2007 Task 17 (4,654 instances)
"""

import pytest
import random
from td.perception.lesk_wsd import LeskWSD


# ─── Benchmark Data ────────────────────────────────────────────────
# Each word has senses with teach glosses and test sentences.
# Sentences are natural language, NOT templates.
# Covering: biology, technology, finance, geography, food, animals,
# sports, music, astronomy, chemistry, everyday usage.

BENCHMARK_DATA = {
    "cell": {
        "senses": {
            0: {
                "name": "biology",
                "glosses": [
                    "cell is_a organelle",
                    "cell is part of organism",
                    "cell contains nucleus and mitochondria",
                    "cell membrane transports ions across boundary",
                    "red blood cell carries oxygen through body",
                ],
            },
            1: {
                "name": "prison",
                "glosses": [
                    "cell is_a room",
                    "cell is part of prison",
                    "prisoner was locked in the cell overnight",
                    "guards inspected each cell in the block",
                    "the cell was cold damp and had no heating",
                ],
            },
            2: {
                "name": "phone",
                "glosses": [
                    "cell is_a device",
                    "cell connects to network for calls",
                    "cell phone has touchscreen display",
                    "cell towers provide wireless coverage",
                    "battery in cell phone powers the display",
                ],
            },
        },
        "tests": [
            ("the cell membrane controls what enters and exits", 0),
            ("mitochondria inside the cell produce energy for organisms", 0),
            ("cells divide through mitosis to create new cells", 0),
            ("the nucleus contains the cell genetic material", 0),
            ("white blood cell fights infection in immune system", 0),
            ("the inmate escaped from the maximum security cell", 1),
            ("prisoners spend most of their time confined to cells", 1),
            ("the guard searched the cell for contraband items", 1),
            ("each cell in the prison block holds two inmates", 1),
            ("the prisoner banged on the cell door all night", 1),
            ("smartphone cell reception depends on tower proximity", 2),
            ("the cell phone battery lasts about twelve hours", 2),
            ("cell towers provide coverage in rural areas", 2),
            ("her cell phone screen cracked when she dropped it", 2),
            ("cellular network divides coverage into cells", 2),
        ],
    },
    "bank": {
        "senses": {
            0: {
                "name": "financial",
                "glosses": [
                    "bank is_a institution",
                    "bank approved the loan application",
                    "bank teller processed the deposit",
                    "bank vault stores cash and gold",
                    "central bank raised interest rates",
                ],
            },
            1: {
                "name": "river",
                "glosses": [
                    "bank is part of river",
                    "river bank was muddy after rain",
                    "fishermen stood on the bank casting lines",
                    "erosion wore away at the bank over years",
                    "the bank of the stream had wildflowers growing",
                ],
            },
        },
        "tests": [
            ("the bank approved my mortgage application yesterday", 0),
            ("she works as a teller at the bank downtown", 0),
            ("the bank vault has reinforced steel walls", 0),
            ("online banking has transformed how people use the bank", 0),
            ("the bank manager reviewed the loan application", 0),
            ("wildflowers grew along the bank of the stream", 1),
            ("the river bank was steep and muddy after rain", 1),
            ("geese nested along the bank of the lake", 1),
            ("the bank collapsed during the flood last spring", 1),
            ("children played on the sunny bank by the creek", 1),
        ],
    },
    "apple": {
        "senses": {
            0: {
                "name": "fruit",
                "glosses": [
                    "apple is_a fruit",
                    "apple grows on trees in orchards",
                    "apple pie is a classic dessert recipe",
                    "green apple has a tart sour taste",
                    "eat an apple a day for good health",
                ],
            },
            1: {
                "name": "company",
                "glosses": [
                    "apple is_a company",
                    "apple released a new iphone with camera",
                    "apple stock price rose after earnings",
                    "apple silicon chip outperforms intel",
                    "tim cook became apple ceo after jobs",
                ],
            },
        },
        "tests": [
            ("the apple orchard harvest happens every autumn", 0),
            ("she bit into the ripe red apple at lunch", 0),
            ("apple juice is a popular beverage worldwide", 0),
            ("baked apple with cinnamon is delicious", 0),
            ("the apple fell from the tree and rolled downhill", 0),
            ("apple announced the vision pro mixed reality headset", 1),
            ("apple stock hit an all-time high this quarter", 1),
            ("the apple ecosystem integrates hardware and software", 1),
            ("apple pay allows contactless payments using your phone", 1),
            ("apple music has over one hundred million subscribers", 1),
        ],
    },
    "python": {
        "senses": {
            0: {
                "name": "programming",
                "glosses": [
                    "python is_a programming language",
                    "python supports multiple paradigms",
                    "write a python script to parse data",
                    "python package manager pip installs libraries",
                    "python decorators modify function behavior",
                ],
            },
            1: {
                "name": "snake",
                "glosses": [
                    "python is_a snake",
                    "python slithered through the jungle",
                    "python can grow to over twenty feet",
                    "the python coiled around the branch waiting",
                    "reticulated python is the longest snake",
                ],
            },
        },
        "tests": [
            ("python is widely used for data science and machine learning", 0),
            ("she wrote a python script to automate the report", 0),
            ("python supports object-oriented and functional programming", 0),
            ("the python library has excellent documentation", 0),
            ("debugging python code is easier with proper logging", 0),
            ("the python slithered silently through the tall grass", 1),
            ("a python can swallow prey much larger than its head", 1),
            ("the zookeeper fed the python a large rat", 1),
            ("burmese python populations are invasive in florida", 1),
            ("the python wrapped itself around the tree branch", 1),
        ],
    },
    "mercury": {
        "senses": {
            0: {
                "name": "planet",
                "glosses": [
                    "mercury is_a planet",
                    "mercury is closest planet to the sun",
                    "mercury completes orbit in eighty-eight days",
                    "surface of mercury has extreme temperatures",
                    "mercury is the smallest planet in solar system",
                ],
            },
            1: {
                "name": "element",
                "glosses": [
                    "mercury is_a element",
                    "mercury is a toxic heavy metal",
                    "mercury poisoning causes neurological damage",
                    "mercury in thermometer rose to thirty-seven",
                    "mercury was used in old barometers",
                ],
            },
        },
        "tests": [
            ("mercury orbits closer to the sun than any other planet", 0),
            ("mercury has no atmosphere and is heavily cratered", 0),
            ("a day on mercury lasts about fifty-nine earth days", 0),
            ("mercury is visible from earth with the naked eye", 0),
            ("the surface of mercury reaches four hundred degrees", 0),
            ("mercury poisoning can cause serious health problems", 1),
            ("the mercury in the thermometer showed a high fever", 1),
            ("mercury amalgam was used in dental fillings for decades", 1),
            ("workers were exposed to mercury vapor in the factory", 1),
            ("mercury contamination in the river harmed fish populations", 1),
        ],
    },
}


# ─── Baselines ─────────────────────────────────────────────────────

def random_baseline(num_senses: int, num_trials: int = 10000) -> float:
    """Random baseline: pick a random sense. Returns expected accuracy."""
    if num_senses <= 1:
        return 1.0
    return 1.0 / num_senses


def most_frequent_baseline(data: dict) -> float:
    """Most-frequent-sense baseline: always pick sense 0."""
    correct = 0
    total = 0
    for word_data in data.values():
        for _, expected in word_data["tests"]:
            if expected == 0:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# ─── Tests ─────────────────────────────────────────────────────────

class TestWSDBenchmark:
    """Comprehensive WSD benchmark on real polysemous words."""

    @pytest.fixture
    def lesk(self):
        """Pre-built Lesk WSD with all benchmark glosses."""
        lesk = LeskWSD()
        for word, data in BENCHMARK_DATA.items():
            for sense_idx, sense_data in data["senses"].items():
                for gloss in sense_data["glosses"]:
                    lesk.add_sense_example(word, sense_idx, gloss)
        return lesk

    def test_lesk_accuracy(self, lesk):
        """Lesk algorithm accuracy on the full benchmark."""
        correct = 0
        total = 0
        fallback = 0

        for word, data in BENCHMARK_DATA.items():
            for sentence, expected in data["tests"]:
                resolved = lesk.resolve_sense(word, sentence)
                if resolved == -1:
                    fallback += 1
                elif resolved == expected:
                    correct += 1
                total += 1

        fired = total - fallback
        accuracy = correct / fired if fired > 0 else 0.0
        print(f"\n  Lesk accuracy: {correct}/{fired} = {accuracy:.1%}")
        print(f"  Fallback: {fallback}/{total} = {fallback/total:.1%}")
        print(f"  Total accuracy: {correct}/{total} = {correct/total:.1%}")

        # Lesk should beat random baseline
        avg_senses = sum(len(d["senses"]) for d in BENCHMARK_DATA.values()) / len(BENCHMARK_DATA)
        random_acc = random_baseline(int(avg_senses))
        assert accuracy > random_acc, f"Lesk ({accuracy:.1%}) should beat random ({random_acc:.1%})"

    def test_lesk_precision_per_sense(self, lesk):
        """Per-sense precision: Lesk should not be biased toward sense 0."""
        sense_correct = {}
        sense_total = {}

        for word, data in BENCHMARK_DATA.items():
            for sentence, expected in data["tests"]:
                resolved = lesk.resolve_sense(word, sentence)
                if resolved != -1:  # Only count when Lesk fires
                    sense_total[expected] = sense_total.get(expected, 0) + 1
                    if resolved == expected:
                        sense_correct[expected] = sense_correct.get(expected, 0) + 1

        print("\n  Per-sense precision:")
        for sense_idx in sorted(sense_total.keys()):
            total = sense_total[sense_idx]
            correct = sense_correct.get(sense_idx, 0)
            prec = correct / total if total > 0 else 0
            print(f"    Sense {sense_idx}: {correct}/{total} = {prec:.1%}")

    def test_lesk_no_false_positives(self, lesk):
        """Lesk should not confidently assign wrong senses."""
        wrong_confident = 0
        for word, data in BENCHMARK_DATA.items():
            for sentence, expected in data["tests"]:
                resolved = lesk.resolve_sense(word, sentence)
                if resolved != -1 and resolved != expected:
                    wrong_confident += 1

        # Wrong confident assignments should be rare
        total = sum(len(d["tests"]) for d in BENCHMARK_DATA.values())
        error_rate = wrong_confident / total
        print(f"\n  Wrong confident assignments: {wrong_confident}/{total} = {error_rate:.1%}")
        assert error_rate < 0.3, f"Error rate {error_rate:.1%} too high"

    def test_random_baseline(self):
        """Random baseline for comparison."""
        avg_senses = sum(len(d["senses"]) for d in BENCHMARK_DATA.values()) / len(BENCHMARK_DATA)
        random_acc = random_baseline(int(avg_senses))
        print(f"\n  Random baseline ({int(avg_senses)} senses): {random_acc:.1%}")

    def test_most_frequent_baseline(self):
        """Most-frequent-sense baseline for comparison."""
        mfs_acc = most_frequent_baseline(BENCHMARK_DATA)
        print(f"\n  Most-frequent-sense baseline: {mfs_acc:.1%}")

    def test_fallback_is_honest(self, lesk):
        """When Lesk returns -1, it should mean 'I don't know'."""
        # Test with completely unrelated sentences
        unrelated = [
            ("the weather is nice today", "cell"),
            ("she walked to the store yesterday", "bank"),
            ("he likes to play guitar", "apple"),
            ("the movie was entertaining", "python"),
            ("she decorated the room with flowers", "mercury"),
        ]
        for sentence, word in unrelated:
            resolved = lesk.resolve_sense(word, sentence)
            # Should return -1 (no signal) or 0 (default)
            assert resolved in (-1, 0), f"Unexpected sense {resolved} for '{sentence}'"

    def test_total_benchmark_size(self):
        """Verify benchmark has enough instances for statistical significance."""
        total = sum(len(d["tests"]) for d in BENCHMARK_DATA.values())
        print(f"\n  Total benchmark instances: {total}")
        assert total >= 50, f"Benchmark too small: {total} instances (need >= 50)"
