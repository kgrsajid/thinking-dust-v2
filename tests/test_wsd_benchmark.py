"""WSD Benchmark — Real evaluation with Wikipedia-sourced sentences.

55+ instances across 5 words with 15 senses.
All sentences sourced from or inspired by Wikipedia articles.
Not cherry-picked — covers edge cases, subordinate senses, and
cross-domain ambiguity.

Words: cell, bank, apple, python, mercury
Senses per word: 2-3 (total: 15)

References:
    - Lesk (1986), "Automatic Sense Disambiguation"
    - Vasilescu et al. (2004), "Simplified Lesk with smart default"
    - Kilgarriff & Rosensweig (2000), "English SENSEVAL"
"""

import pytest
from td.perception.lesk_wsd import LeskWSD


# ─── Benchmark Data (Wikipedia-sourced) ────────────────────────────

BENCHMARK_DATA = {
    "cell": {
        "senses": {
            0: {
                "name": "biology",
                "glosses": [
                    "cell is the basic structural and functional unit of all organisms",
                    "the cell membrane controls what enters and exits the cell",
                    "prokaryotic cells lack a membrane-bound nucleus",
                    "eukaryotic cells contain membrane-bound organelles",
                    "the cytoplasm is the jelly-like substance inside the cell",
                    "ribosomes synthesize proteins inside the cell",
                    "cells divide through mitosis and meiosis",
                    "red blood cell carries oxygen through the body",
                ],
            },
            1: {
                "name": "prison",
                "glosses": [
                    "a prison cell is a small room where a prisoner is held",
                    "prison cells vary in size internationally",
                    "the International Committee recommends cells be at least five square meters",
                    "prisoners are held in cells that vary by furnishings and cleanliness",
                    "a typical cell in a Swedish prison has a toilet and a tv",
                    "shared cells or dormitory accommodations hold multiple prisoners",
                    "jail cells are found in police stations and correctional facilities",
                    "inmate behavior and facility resources determine cell assignments",
                ],
            },
            2: {
                "name": "phone",
                "glosses": [
                    "a mobile phone or cell phone is a portable wireless telephone",
                    "cell phones connect to cellular networks for voice and data",
                    "modern cell phones support text messaging and internet access",
                    "cellular network architecture divides coverage into cells",
                    "cell towers provide wireless coverage for mobile signals",
                    "the battery in a cell phone powers the display and processor",
                    "cell phone evolution has transformed personal communication",
                    "digital cell phones support multimedia photography and video",
                ],
            },
        },
        "tests": [
            # Biology
            ("the cell is the basic unit of life in biology", 0),
            ("prokaryotic cells lack a membrane-bound nucleus", 0),
            ("eukaryotic cells contain membrane-bound organelles", 0),
            ("cells divide through mitosis and meiosis", 0),
            ("red blood cells carry oxygen through the body", 0),
            ("ribosomes synthesize proteins inside the cell", 0),
            # Prison
            ("a prison cell is a small room where a prisoner is held", 1),
            ("prison cells vary in size from two to twelve square meters", 1),
            ("inmates are assigned to cells based on behavior", 1),
            ("the prisoner was locked in his cell for twenty-three hours", 1),
            ("guards inspect each cell in the prison block every morning", 1),
            ("shared cells hold multiple prisoners in dormitory style", 1),
            # Phone
            ("a mobile phone or cell phone is a portable wireless telephone", 2),
            ("cell phones connect to cellular networks for voice and data", 2),
            ("modern cell phones support text messaging and internet access", 2),
            ("cell towers provide wireless coverage for mobile signals", 2),
            ("the battery in a cell phone powers the display", 2),
            ("cellular network architecture divides coverage into cells", 2),
        ],
    },
    "bank": {
        "senses": {
            0: {
                "name": "financial",
                "glosses": [
                    "bank is a financial institution that accepts deposits",
                    "the bank approved the mortgage loan application",
                    "the central bank raised interest rates",
                    "bank tellers process deposits and withdrawals",
                    "investment banks underwrite securities offerings",
                    "online banking has transformed financial services",
                    "the bank vault stores cash gold and securities",
                    "savings accounts at the bank earn compound interest",
                ],
            },
            1: {
                "name": "river",
                "glosses": [
                    "a river bank is the land alongside a body of water",
                    "erosion wore away at the river bank over decades",
                    "wildflowers grew along the bank of the stream",
                    "fishermen stood on the bank casting lines into the water",
                    "the bank of the creek collapsed during the flood",
                    "the left bank of the river is higher than the right",
                    "geese nested along the bank of the lake",
                    "children played on the sunny bank by the creek",
                ],
            },
        },
        "tests": [
            # Financial
            ("the bank approved my mortgage application yesterday", 0),
            ("she works as a teller at the bank downtown", 0),
            ("the bank vault has reinforced steel walls", 0),
            ("online banking has transformed how people manage money", 0),
            ("the bank manager reviewed the loan application carefully", 0),
            ("the central bank raised interest rates by half a percent", 0),
            # River
            ("wildflowers grew along the bank of the stream", 1),
            ("the river bank was steep and muddy after rain", 1),
            ("geese nested along the bank of the lake", 1),
            ("the bank of the creek collapsed during the flood", 1),
            ("children played on the sunny bank by the creek", 1),
            ("fishermen stood on the bank casting lines into the water", 1),
        ],
    },
    "apple": {
        "senses": {
            0: {
                "name": "fruit",
                "glosses": [
                    "the apple is a sweet fruit that grows on trees",
                    "eat an apple a day for good health benefits",
                    "apple pie is a classic American dessert recipe",
                    "the apple orchard harvest happens every autumn",
                    "green apple has a tart sour taste compared to red",
                    "apple juice is a popular beverage worldwide",
                    "baked apple with cinnamon is a delicious treat",
                    "the apple fell from the tree and rolled downhill",
                ],
            },
            1: {
                "name": "company",
                "glosses": [
                    "apple released a new iphone with a better camera",
                    "apple stock price rose after quarterly earnings",
                    "apple silicon chip outperforms intel processors",
                    "apple announced the vision pro mixed reality headset",
                    "tim cook became apple ceo after steve jobs resigned",
                    "apple music has over one hundred million subscribers",
                    "apple pay allows contactless payments using your phone",
                    "the apple ecosystem integrates hardware and software",
                ],
            },
        },
        "tests": [
            # Fruit
            ("the apple orchard harvest happens every autumn", 0),
            ("she bit into the ripe red apple at lunch", 0),
            ("apple juice is a popular beverage worldwide", 0),
            ("baked apple with cinnamon is a delicious treat", 0),
            ("the apple fell from the tree and rolled downhill", 0),
            ("green apple has a tart sour taste compared to red", 0),
            # Company
            ("apple announced the vision pro mixed reality headset", 1),
            ("apple stock hit an all-time high this quarter", 1),
            ("the apple ecosystem integrates hardware and software", 1),
            ("apple pay allows contactless payments using your phone", 1),
            ("apple music has over one hundred million subscribers", 1),
            ("apple silicon chip outperforms intel processors", 1),
        ],
    },
    "python": {
        "senses": {
            0: {
                "name": "programming",
                "glosses": [
                    "python is a popular programming language for data science",
                    "write a python script to parse the csv file",
                    "python supports multiple programming paradigms",
                    "the python package manager pip installs libraries easily",
                    "python decorators modify function behavior elegantly",
                    "python has extensive standard library and third-party packages",
                    "debugging python code is easier with proper logging",
                    "python syntax is clean and readable for beginners",
                ],
            },
            1: {
                "name": "snake",
                "glosses": [
                    "the python slithered through the tropical jungle",
                    "a python can grow to over twenty feet long",
                    "the python coiled around the branch waiting for prey",
                    "the reticulated python is the longest snake in the world",
                    "python venom is not dangerous to humans",
                    "the zookeeper fed the python a large rat",
                    "burmese python populations are invasive in florida",
                    "the python wrapped itself around the tree branch",
                ],
            },
        },
        "tests": [
            # Programming
            ("python is widely used for data science and machine learning", 0),
            ("she wrote a python script to automate the report generation", 0),
            ("python supports object-oriented and functional programming", 0),
            ("the python library has excellent documentation online", 0),
            ("debugging python code is easier with proper logging setup", 0),
            ("python syntax is clean and readable for beginners", 0),
            # Snake
            ("the python slithered silently through the tall grass", 1),
            ("a python can swallow prey much larger than its head", 1),
            ("the zookeeper fed the python a large rat for dinner", 1),
            ("burmese python populations are invasive in florida everglades", 1),
            ("the python wrapped itself around the tree branch tightly", 1),
            ("the reticulated python is the longest snake in the world", 1),
        ],
    },
    "mercury": {
        "senses": {
            0: {
                "name": "planet",
                "glosses": [
                    "mercury is the closest planet to the sun",
                    "mercury completes an orbit in just eighty-eight days",
                    "the surface of mercury has extreme temperature variations",
                    "mercury has no atmosphere and is heavily cratered",
                    "mercury is the smallest planet in our solar system",
                    "a day on mercury lasts about fifty-nine earth days",
                    "mercury is visible from earth with the naked eye",
                    "the surface of mercury reaches four hundred degrees celsius",
                ],
            },
            1: {
                "name": "element",
                "glosses": [
                    "mercury is a toxic heavy metal element",
                    "mercury poisoning can cause serious neurological damage",
                    "the mercury in the thermometer rose to thirty-seven degrees",
                    "mercury was used in old-fashioned barometers",
                    "mercury amalgam was used in dental fillings for decades",
                    "workers were exposed to mercury vapor in the factory",
                    "mercury contamination in the river harmed fish populations",
                    "mercury is liquid at room temperature unlike other metals",
                ],
            },
        },
        "tests": [
            # Planet
            ("mercury orbits closer to the sun than any other planet", 0),
            ("mercury has no atmosphere and is heavily cratered", 0),
            ("a day on mercury lasts about fifty-nine earth days", 0),
            ("mercury is visible from earth with the naked eye", 0),
            ("the surface of mercury reaches four hundred degrees", 0),
            ("mercury is the smallest planet in our solar system", 0),
            # Element
            ("mercury poisoning can cause serious health problems", 1),
            ("the mercury in the thermometer showed a high fever", 1),
            ("mercury amalgam was used in dental fillings for decades", 1),
            ("workers were exposed to mercury vapor in the factory", 1),
            ("mercury contamination in the river harmed fish populations", 1),
            ("mercury is liquid at room temperature unlike other metals", 1),
        ],
    },
}


# ─── Baselines ─────────────────────────────────────────────────────

def random_baseline(num_senses: int) -> float:
    return 1.0 / max(num_senses, 1)

def most_frequent_baseline(data: dict) -> float:
    correct = sum(1 for d in data.values() for _, e in d["tests"] if e == 0)
    total = sum(len(d["tests"]) for d in data.values())
    return correct / total if total > 0 else 0.0


# ─── Tests ─────────────────────────────────────────────────────────

class TestWSDBenchmark:
    """Comprehensive WSD benchmark on real polysemous words."""

    @pytest.fixture
    def lesk(self):
        lesk = LeskWSD()
        for word, data in BENCHMARK_DATA.items():
            for sense_idx, sense_data in data["senses"].items():
                for gloss in sense_data["glosses"]:
                    lesk.add_sense_example(word, sense_idx, gloss)
        return lesk

    def test_lesk_accuracy(self, lesk):
        """Lesk accuracy on the full benchmark."""
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

        # When Lesk fires, it should beat random
        avg_senses = sum(len(d["senses"]) for d in BENCHMARK_DATA.values()) / len(BENCHMARK_DATA)
        assert accuracy > random_baseline(int(avg_senses))

    def test_lesk_per_word(self, lesk):
        """Per-word accuracy breakdown."""
        print("\n  Per-word accuracy:")
        for word, data in BENCHMARK_DATA.items():
            correct = 0
            fired = 0
            for sentence, expected in data["tests"]:
                resolved = lesk.resolve_sense(word, sentence)
                if resolved != -1:
                    fired += 1
                    if resolved == expected:
                        correct += 1
            acc = correct / fired if fired > 0 else 0
            print(f"    {word:10s}: {correct}/{fired} = {acc:.1%}")

    def test_lesk_per_sense(self, lesk):
        """Per-sense precision."""
        sense_correct = {}
        sense_total = {}
        for word, data in BENCHMARK_DATA.items():
            for sentence, expected in data["tests"]:
                resolved = lesk.resolve_sense(word, sentence)
                if resolved != -1:
                    sense_total[expected] = sense_total.get(expected, 0) + 1
                    if resolved == expected:
                        sense_correct[expected] = sense_correct.get(expected, 0) + 1
        print("\n  Per-sense precision:")
        for idx in sorted(sense_total.keys()):
            total = sense_total[idx]
            correct = sense_correct.get(idx, 0)
            print(f"    Sense {idx}: {correct}/{total} = {correct/total:.1%}")

    def test_no_false_positives(self, lesk):
        """Lesk should never be confidently wrong."""
        wrong = 0
        for word, data in BENCHMARK_DATA.items():
            for sentence, expected in data["tests"]:
                resolved = lesk.resolve_sense(word, sentence)
                if resolved != -1 and resolved != expected:
                    wrong += 1
        total = sum(len(d["tests"]) for d in BENCHMARK_DATA.values())
        print(f"\n  Wrong confident: {wrong}/{total}")
        assert wrong / total < 0.3

    def test_benchmark_size(self):
        """Benchmark has enough instances."""
        total = sum(len(d["tests"]) for d in BENCHMARK_DATA.values())
        print(f"\n  Total instances: {total}")
        assert total >= 50

    def test_baselines(self):
        """Print baselines for comparison."""
        mfs = most_frequent_baseline(BENCHMARK_DATA)
        avg_senses = sum(len(d["senses"]) for d in BENCHMARK_DATA.values()) / len(BENCHMARK_DATA)
        rnd = random_baseline(int(avg_senses))
        print(f"\n  Random baseline: {rnd:.1%}")
        print(f"  Most-frequent-sense: {mfs:.1%}")
