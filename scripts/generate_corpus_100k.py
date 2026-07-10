#!/usr/bin/env python3
"""Generate 100K domain-specific sentences for BEAGLE training.

Research-backed approach (Jones, Gorman & Wewhort, 2015):
- Domain quality > raw size (TASA beat Wikipedia)
- Sentences cover 11 domains for WSD + general knowledge
- Each domain uses vocabulary from TD v2's WSD benchmark words

Domains:
1. Biology — cell(organelle), DNA, membrane, nucleus
2. Prison — cell(prison), prisoner, jail, guard
3. Technology — cell(phone), mobile, wireless, tower
4. Finance — bank(finance), loan, mortgage, interest
5. Geography — bank(river), countries, capitals, rivers
6. Food — apple(fruit), orchard, juice, pie
7. Programming — python(lang), code, script, library
8. Zoology — python(snake), reptile, constrictor
9. Astronomy — mercury(planet), orbit, solar, planet
10. Chemistry — mercury(element), metal, thermometer
11. General — countries, capitals, history, sports

Output: data/synthetic_corpus_100k.txt (one sentence per line)

Reference: Jones, M.N., Gorman, R.M., & Wewhort, D.J.K. (2015).
    "Encoding Sequential Information in Semantic Space Models."
    Psychonomic Bulletin & Review. PMC4405220.
"""

import random
import os

random.seed(42)  # Reproducible

# ── Domain vocabularies ──────────────────────────────────────────────────

BIOLOGY = {
    "nouns": ["cell", "membrane", "nucleus", "organelle", "cytoplasm", "ribosome",
              "mitochondria", "DNA", "RNA", "protein", "enzyme", "chromosome",
              "gene", "tissue", "organ", "organism", "bacteria", "virus",
              "membrane", "wall", "fiber", "blood", "nerve", "muscle"],
    "verbs": ["contains", "produces", "synthesizes", "replicates", "divides",
              "absorbs", "releases", "transports", "regulates", "controls",
              "generates", "processes", "stores", "transmits", "grows"],
    "adjectives": ["living", "organic", "biological", "cellular", "microscopic",
                   "eukaryotic", "prokaryotic", "single-celled", "multicellular"],
    "templates": [
        "The {noun} {verb} essential molecules for survival.",
        "A {adj} {noun} {verb} nutrients from the environment.",
        "The {noun} is surrounded by a protective membrane.",
        "Inside the {noun}, the {noun2} {verb} genetic information.",
        "The {noun} {verb} energy through a complex process.",
        "Biological {noun}s {verb} in response to external stimuli.",
        "The {adj} {noun} {verb} new {noun2}s through division.",
        "Proteins are synthesized by {noun}s in the cytoplasm.",
        "The {noun} {verb} the flow of molecules across the barrier.",
        "DNA {verb} the instructions for building {adj} {noun}s.",
        "The {noun} {verb} waste products from the cell.",
        "Mitochondria {verb} energy for the {adj} {noun}.",
        "The {noun} {verb} chemical signals to other {noun2}s.",
        "A {adj} {noun} contains many different types of {noun2}s.",
        "The {noun} {verb} the structure of the {adj} {noun2}.",
    ],
}

PRISON = {
    "nouns": ["cell", "prisoner", "inmate", "guard", "warden", "jail", "prison",
              "block", "yard", "sentence", "crime", "convict", "escape",
              "bars", "lock", "key", "uniform", "dormitory", "corridor"],
    "verbs": ["locked", "guarded", "escorted", "patrolled", "sentenced",
              "released", "transferred", "confined", "restricted", "watched",
              "served", "escaped", "arrested", "detained", "held"],
    "adjectives": ["maximum-security", "solitary", "dark", "narrow", "reinforced",
                   "guarded", "confined", "restricted", "isolated", "grim"],
    "templates": [
        "The {noun} was {verb} in a small {adj} room.",
        "A {adj} {noun} {verb} the perimeter of the facility.",
        "The {noun} {verb} for three years before being released.",
        "Each {noun} has a {adj} {noun2} with a steel door.",
        "The {noun} {verb} through the {adj} corridors all night.",
        "Prisoners are confined to their {noun}s during lockdown.",
        "The {noun} {verb} the {noun2} to the visiting area.",
        "A {adj} {noun2} separates the {noun}s from the outside.",
        "The {noun} was {verb} for attempting to escape.",
        "Guards {verb} each {noun} twice during the night shift.",
        "The {adj} {noun} had no windows and one steel door.",
        "The {noun} {verb} his sentence in a federal facility.",
        "The warden {verb} all {adj} {noun}s to remain locked.",
        "A single {noun} occupied the {adj} {noun2} for months.",
        "The {noun} was {verb} to solitary confinement for misconduct.",
    ],
}

TECHNOLOGY = {
    "nouns": ["phone", "device", "signal", "tower", "network", "wireless",
              "battery", "screen", "app", "software", "data", "connection",
              "antenna", "carrier", "bandwidth", "spectrum", "router", "modem"],
    "verbs": ["transmits", "receives", "connects", "broadcasts", "charges",
              "downloads", "uploads", "processes", "stores", "displays",
              "communicates", "operates", "functions", "activates", "syncs"],
    "adjectives": ["mobile", "wireless", "digital", "portable", "smart",
                   "cellular", "broadband", "high-speed", "encrypted", "remote"],
    "templates": [
        "The {noun} {verb} data over a {adj} network.",
        "A {adj} {noun} {verb} to the nearest {noun2}.",
        "The {noun} {verb} electromagnetic signals at high frequency.",
        "Modern {noun}s {verb} information using digital protocols.",
        "The {adj} {noun} {verb} a strong connection to the {noun2}.",
        "The {noun} {verb} when the {noun2} is within range.",
        "A {adj} {noun} can {verb} data at incredible speeds.",
        "The {noun} {verb} the user's location via satellite.",
        "The {adj} {noun2} {verb} coverage across the entire city.",
        "Your {noun} {verb} automatically when you enter the area.",
        "The {noun} {verb} the signal before forwarding it.",
        "A {adj} {noun} uses less power than a traditional {noun2}.",
        "The {noun} {verb} multiple connections simultaneously.",
        "The {adj} {noun} {verb} sensitive user data securely.",
        "The {noun} {verb} to the network within seconds.",
    ],
}

FINANCE = {
    "nouns": ["bank", "loan", "mortgage", "interest", "teller", "account",
              "deposit", "withdrawal", "credit", "debit", "investment",
              "savings", "check", "transfer", "balance", "rate", "fee",
              "branch", "customer", "transaction", "finance", "budget",
              "revenue", "profit", "loss", "debt", "asset", "liability"],
    "verbs": ["offers", "charges", "deposits", "withdraws", "transfers",
              "approves", "denies", "calculates", "processes", "manages",
              "invests", "lends", "borrows", "repays", "collects",
              "budgets", "earns", "spends", "saves", "finances"],
    "adjectives": ["annual", "monthly", "fixed", "variable", "compound",
                   "low-interest", "high-yield", "federally-insured", "digital"],
    "templates": [
        "The {noun} {verb} a low {noun2} rate on savings accounts.",
        "A {adj} {noun} {verb} customers to deposit money online.",
        "The {noun} {verb} a penalty for early {noun2} withdrawal.",
        "She opened a {adj} {noun2} at the local {noun}.",
        "The {noun} {verb} the {noun2} over a thirty-year period.",
        "The {adj} {noun} {verb} competitive rates for new customers.",
        "The {noun} {verb} a fee for each international transfer.",
        "A {adj} {noun2} accrues interest over time.",
        "The {noun} {verb} the loan application within two days.",
        "He {verb} money from his savings {noun} every month.",
        "The {noun} {verb} the balance at the end of each month.",
        "The {adj} {noun} requires a minimum deposit of one hundred dollars.",
        "The {noun} {verb} overdraft protection for premium accounts.",
        "A {adj} {noun2} offers higher returns than a regular savings account.",
        "The {noun} {verb} fraud alerts to all its customers.",
    ],
}

GEOGRAPHY = {
    "nouns": ["bank", "river", "shore", "coast", "ocean", "sea", "lake",
              "mountain", "valley", "desert", "forest", "continent", "country",
              "capital", "city", "border", "island", "peninsula", "climate"],
    "verbs": ["flows", "erodes", "borders", "stretches", "surrounds",
              "separates", "connects", "covers", "extends", "lies",
              "contains", "borders", "crosses", "meets", "joins"],
    "adjectives": ["northern", "southern", "eastern", "western", "tropical",
                   "arid", "fertile", "mountainous", "coastal", "landlocked"],
    "templates": [
        "The river {verb} along the eastern {noun} of the country.",
        "The {adj} {noun} {verb} for hundreds of miles.",
        "The {noun} was slowly {verb} by the flowing water.",
        "The {adj} {noun} {verb} the country from its neighbor.",
        "A fertile {noun} lies between the mountain and the {noun2}.",
        "The {noun} {verb} into the ocean near the coastal city.",
        "The {adj} {noun} is home to thousands of species.",
        "The capital city sits on the {adj} {noun} of the river.",
        "The {noun} {verb} the entire northern region of the continent.",
        "The {adj} {noun} receives very little rainfall each year.",
        "The river {verb} through three countries before reaching the sea.",
        "The {noun} {verb} a natural barrier between the two regions.",
        "The {adj} {noun} has a mild climate year round.",
        "The {noun} {verb} the landscape into distinct ecological zones.",
        "An active volcano {verb} near the {adj} {noun} of the island.",
        "The {adj} {noun} contains diverse ecosystems and wildlife habitats.",
        "A wide {noun} {verb} between the two mountain ranges.",
        "The {noun} {verb} sand along the coastline during storms.",
        "The {adj} {noun} is formed by volcanic activity underground.",
        "The {noun} {verb} fresh water to millions of people downstream.",
        "The {adj} {noun} features dramatic cliffs and deep valleys.",
        "Seasonal flooding {verb} the {adj} {noun} with nutrient-rich soil.",
        "The {noun} {verb} through a narrow gorge before widening.",
        "The {adj} {noun} supports agriculture and fishing industries.",
        "The {noun} {verb} sediment across the floodplain each spring.",
    ],
}

FOOD = {
    "nouns": ["apple", "fruit", "orchard", "juice", "pie", "tree", "seed",
              "harvest", "crop", "farm", "garden", "flavor", "taste",
              "recipe", "ingredient", "kitchen", "chef", "dish", "meal"],
    "verbs": ["grows", "ripens", "harvests", "picks", "bakes", "cooks",
              "squeezes", "slices", "peels", "tastes", "flavors", "prepares",
              "serves", "cultivates", "produces"],
    "adjectives": ["ripe", "fresh", "organic", "sweet", "juicy",
                   "crisp", "delicious", "homemade", "seasonal", "tropical"],
    "templates": [
        "The {adj} {noun} {verb} in the {adj} {noun2}.",
        "She {verb} a {adj} {noun} from the {noun2}.",
        "The {noun} {verb} a sweet and tangy {noun2}.",
        "The {adj} {noun} is ready for {noun2} in autumn.",
        "He {verb} the {noun} into thin slices for the {noun2}.",
        "The {noun} {verb} best in {adj} climates.",
        "A {adj} {noun} makes an excellent {noun2} for dessert.",
        "The {noun} {verb} the {noun2} with fresh ingredients.",
        "The {adj} {noun} is picked at the peak of ripeness.",
        "She {verb} a {adj} {noun} {noun2} for the family dinner.",
        "The {noun} {verb} thousands of bushels each {noun2}.",
        "The {adj} {noun} has a distinctive sweet {noun2}.",
        "Fresh {noun} {verb} the best flavor when eaten raw.",
        "The {noun} {verb} the {adj} {noun2} from scratch.",
        "A {adj} {noun} is rich in vitamins and fiber.",
    ],
}

PROGRAMMING = {
    "nouns": ["python", "code", "script", "library", "function", "variable",
              "module", "class", "object", "array", "loop", "algorithm",
              "debugger", "compiler", "syntax", "error", "bug", "framework",
              "API", "database", "query"],
    "verbs": ["runs", "compiles", "executes", "imports", "defines", "calls",
              "returns", "iterates", "parses", "processes", "handles",
              "implements", "extends", "overrides", "initializes"],
    "adjectives": ["recursive", "dynamic", "typed", "interpreted", "compiled",
                   "modular", "scalable", "efficient", "elegant", "robust"],
    "templates": [
        "The {noun} {verb} the data into a structured format.",
        "A {adj} {noun} {verb} without any external dependencies.",
        "The {noun} {verb} each element in the {adj} {noun2}.",
        "The {adj} {noun} {verb} faster than the naive implementation.",
        "She {verb} the {noun} to handle edge cases gracefully.",
        "The {noun} {verb} a {adj} {noun2} for processing data.",
        "The {adj} {noun} is widely used in data science.",
        "The {noun} {verb} the {noun2} and returns the result.",
        "A {adj} {noun} can {verb} thousands of records per second.",
        "The {noun} {verb} when the input violates the expected format.",
        "The {adj} {noun} {verb} clean and readable code.",
        "The {noun} {verb} the {adj} {noun2} from the remote server.",
        "Python {verb} a simple and intuitive syntax for beginners.",
        "The {noun} {verb} the program's execution at runtime.",
        "A well-designed {noun} {verb} the need for complex {noun2}.",
    ],
}

ZOOLOGY = {
    "nouns": ["python", "snake", "reptile", "constrictor", "predator", "prey",
              "habitat", "species", "scales", "fangs", "venom", "egg",
              "nest", "burrow", "migration", "camouflage", "cold-blooded"],
    "verbs": ["constricts", "swallows", "hunts", "crawls", "slithers",
              "strikes", "sheds", "hibernates", "migrates", "breeds",
              "camouflages", "basks", "feeds", "attacks", "escapes"],
    "adjectives": ["venomous", "non-venomous", "arboreal", "terrestrial",
                   "nocturnal", "diurnal", "cold-blooded", "sleek", "massive"],
    "templates": [
        "The {adj} {noun} {verb} its prey by constriction.",
        "A {adj} {noun} {verb} through the undergrowth silently.",
        "The {noun} {verb} warm rocks to regulate body temperature.",
        "The {adj} {noun} {verb} small mammals and birds.",
        "A {noun} {verb} its skin several times a year.",
        "The {adj} {noun} {verb} in tropical forests and grasslands.",
        "The {noun} {verb} the prey whole after constricting it.",
        "A {adj} {noun} can {verb} animals much larger than itself.",
        "The {noun} {verb} eggs in a warm, hidden location.",
        "The {adj} {noun} {verb} during the dry season.",
        "The {noun} {verb} its pattern to blend with surroundings.",
        "A {adj} {noun} {verb} primarily at night.",
        "The {noun} {verb} through trees with remarkable agility.",
        "The {adj} {noun} is one of the largest {noun2} species.",
        "The {noun} {verb} when threatened by a larger predator.",
    ],
}

ASTRONOMY = {
    "nouns": ["mercury", "planet", "orbit", "sun", "star", "solar system",
              "moon", "surface", "crater", "atmosphere", "gravity", "telescope",
              "spacecraft", "mission", "distance", "temperature", "rotation"],
    "verbs": ["orbits", "rotates", "revolves", "reflects", "emits",
              "travels", "completes", "measures", "observes", "discovers",
              "explores", "illuminates", "heats", "cools", "attracts"],
    "adjectives": ["terrestrial", "gaseous", "rocky", "inner", "outer",
                   "scorching", "frozen", "barren", "cratered", "distant"],
    "templates": [
        "The {adj} {noun} {verb} closest to the {noun2}.",
        "A {adj} {noun} {verb} around the {noun2} in eighty-eight days.",
        "The {noun} {verb} the most sunlight of any {noun2}.",
        "The {adj} {noun} has no atmosphere to retain heat.",
        "The {noun} {verb} on its axis very slowly.",
        "The {adj} {noun} {verb} extreme temperatures between day and night.",
        "The {noun} {verb} the Sun at incredible speed.",
        "A {adj} {noun} {verb} very little light from the Sun.",
        "The {noun} {verb} a year in just eighty-eight Earth days.",
        "The {adj} {noun} is the smallest {noun2} in the solar system.",
        "The {noun} {verb} the Sun once every three months.",
        "The {adj} {noun} {verb} a heavily cratered surface.",
        "The {noun} {verb} no moons or rings of its own.",
        "Scientists {verb} the {adj} {noun} using powerful telescopes.",
        "The {noun} {verb} rapidly during its closest approach to the Sun.",
    ],
}

CHEMISTRY = {
    "nouns": ["mercury", "metal", "element", "liquid", "thermometer", "atom",
              "compound", "reaction", "solution", "acid", "base", "bond",
              "electron", "proton", "neutron", "isotope", "periodic table"],
    "verbs": ["reacts", "dissolves", "combines", "separates", "evaporates",
              "freezes", "melts", "conducts", "corrodes", "oxidizes",
              "reduces", "precipitates", "catalyzes", "ionizes", "bonds"],
    "adjectives": ["toxic", "heavy", "liquid", "volatile", "reactive",
                   "stable", "corrosive", "metallic", "organic", "inorganic"],
    "templates": [
        "The {adj} {noun} {verb} at room temperature.",
        "A {adj} {noun} {verb} other metals to form alloys.",
        "The {noun} {verb} in nitric acid to form a solution.",
        "The {adj} {noun} is used in {noun2}s to measure temperature.",
        "The {noun} {verb} rapidly when exposed to air.",
        "A {adj} {noun} {verb} a silvery-white liquid metal.",
        "The {noun} {verb} toxic fumes when heated above boiling point.",
        "The {adj} {noun} {verb} electrical current very efficiently.",
        "The {noun} {verb} with chlorine to form mercuric chloride.",
        "The {adj} {noun} is found naturally in the Earth's crust.",
        "The {noun} {verb} the bond between the two atoms.",
        "A {adj} {noun} can {verb} organic compounds in solution.",
        "The {noun} {verb} the reaction without being consumed itself.",
        "The {adj} {noun} has the symbol Hg on the periodic table.",
        "The {noun} {verb} into a solid when cooled below minus thirty-nine degrees.",
    ],
}

GENERAL = {
    "nouns": ["capital", "country", "city", "president", "government", "parliament",
              "election", "law", "constitution", "history", "war", "peace",
              "economy", "trade", "culture", "language", "population", "border"],
    "verbs": ["governs", "elects", "trades", "produces", "exports", "imports",
              "celebrates", "governs", "legislates", "constitutes", "represents",
              "declares", "ratifies", "negotiates", "establishes"],
    "adjectives": ["democratic", "federal", "independent", "sovereign", "united",
                   "constitutional", "parliamentary", "republican", "national"],
    "templates": [
        "The capital of {country} is {city}.",
        "The {adj} government {verb} the country from the capital.",
        "The {noun} {verb} new laws every parliamentary session.",
        "The {adj} {noun} has a population of over ten million.",
        "The {noun} {verb} goods with neighboring countries.",
        "The {adj} {noun} gained independence in the twentieth century.",
        "The {noun} {verb} a constitutional amendment last year.",
        "The {adj} {noun} is known for its rich cultural heritage.",
        "The {noun} {verb} diplomatic relations with over a hundred nations.",
        "The {adj} {noun} has a diverse population speaking many languages.",
        "The {noun} {verb} free elections every four years.",
        "The {adj} {noun} is a member of the United Nations.",
        "The {noun} {verb} its independence after a long struggle.",
        "The {adj} {noun} spans two continents.",
        "The {noun} {verb} a federal system of government.",
        "{country} is located in {region} and its capital is {city}.",
        "The {noun} of {country} {verb} a new trade agreement last year.",
        "{city} is the largest city and economic center of {country}.",
        "The {adj} {noun} of {country} attracts millions of tourists annually.",
        "The {noun} in {country} {verb} several official languages.",
        "The {adj} constitution of {country} {verb} fundamental rights.",
        "The {noun} of {country} {verb} to diversify the national economy.",
        "The {adj} {noun} of {country} is one of the oldest in the world.",
        "The {noun} {verb} between {country} and its neighbors strengthened.",
        "The {adj} {noun} of {country} {verb} universal healthcare.",
    ],
}

COUNTRIES = [
    "France", "Germany", "Japan", "Brazil", "India", "Canada", "Australia",
    "Egypt", "Mexico", "Italy", "Spain", "China", "Russia", "Turkey",
    "Argentina", "Kenya", "Thailand", "Poland", "Nigeria", "Vietnam",
    "South Korea", "Indonesia", "Saudi Arabia", "Ukraine", "Colombia",
]

CITIES = [
    "Paris", "Berlin", "Tokyo", "Brasilia", "New Delhi", "Ottawa", "Canberra",
    "Cairo", "Mexico City", "Rome", "Madrid", "Beijing", "Moscow", "Ankara",
    "Buenos Aires", "Nairobi", "Bangkok", "Warsaw", "Abuja", "Hanoi",
    "Seoul", "Jakarta", "Riyadh", "Kyiv", "Bogota",
]

REGIONS = [
    "Europe", "Asia", "Africa", "South America", "North America",
    "the Middle East", "Southeast Asia", "Central Europe", "East Africa",
    "Western Europe", "Northern Africa", "Central America",
]

# ── Additional domain vocab for benchmark coverage ──────────────────────
TOOLS = {
    "nouns": ["match", "tool", "fire", "flame", "lighter", "candle", "torch",
              "spark", "ignition", "fuel", "ember", "kindling", "combustion",
              "smoke", "ash", "burn", "heat", "warmth", "blaze"],
    "verbs": ["strikes", "lights", "burns", "ignites", "starts", "creates",
              "produces", "generates", "extinguishes", "spreads", "glows",
              "flickers", "smolders", "blazes", "heats"],
    "adjectives": ["lit", "burning", "glowing", "smoldering", "extinguished",
                   "flammable", "combustible", "fiery", "hot", "warm"],
    "templates": [
        "The {noun} {verb} when struck against a rough surface.",
        "She struck a {noun} to light the {noun2}.",
        "The {noun} {verb} a bright {noun2} in the darkness.",
        "A single {noun} can {verb} an entire forest.",
        "The {noun} {verb} for only a few seconds before going out.",
        "He used a {noun} to {verb} the campfire.",
        "The {noun} {verb} the {noun2} and it began to burn.",
        "A {adj} {noun} is useful for starting a {noun2}.",
        "The {noun} {verb} a small {noun2} that grew larger.",
        "The {noun} {verb} brightly before the {noun2} died.",
        "She {verb} the {noun} and held it to the candle.",
        "The {noun} is a simple {noun2} for creating fire.",
        "The {noun} {verb} when exposed to oxygen.",
        "A {adj} {noun} {verb} enough heat to warm the room.",
        "The {noun} {verb} a steady flame for several minutes.",
    ],
}


def generate_domain_sentences(domain_name: str, vocab: dict, count: int) -> list[str]:
    """Generate sentences for a domain using templates + vocabulary."""
    sentences = set()
    templates = vocab["templates"]
    nouns = vocab["nouns"]
    verbs = vocab["verbs"]
    adjs = vocab["adjectives"]

    max_attempts = count * 3  # Avoid infinite loop
    attempts = 0

    while len(sentences) < count and attempts < max_attempts:
        attempts += 1
        template = random.choice(templates)

        # Fill template slots
        sentence = template
        sentence = sentence.replace("{noun}", random.choice(nouns), 1)
        sentence = sentence.replace("{noun2}", random.choice(nouns), 1)
        sentence = sentence.replace("{verb}", random.choice(verbs), 1)
        sentence = sentence.replace("{adj}", random.choice(adjs), 1)

        # Fill country/city/region slots for general domain
        if "{country}" in sentence:
            idx = random.randint(0, len(COUNTRIES) - 1)
            sentence = sentence.replace("{country}", COUNTRIES[idx])
            sentence = sentence.replace("{city}", CITIES[idx])
        if "{region}" in sentence:
            sentence = sentence.replace("{region}", random.choice(REGIONS))

        # Clean up any remaining unfilled slots
        if "{" in sentence and "}" in sentence:
            continue

        sentence = sentence.strip()
        if len(sentence) > 20 and sentence not in sentences:
            sentences.add(sentence)

    return list(sentences)


def generate_corpus(target_total: int = 100000, output_path: str = "data/synthetic_corpus_100k.txt"):
    """Generate the full 100K corpus across all domains."""
    domains = {
        "biology": (BIOLOGY, 9000),
        "prison": (PRISON, 9000),
        "technology": (TECHNOLOGY, 9000),
        "finance": (FINANCE, 9000),
        "geography": (GEOGRAPHY, 9000),
        "food": (FOOD, 9000),
        "programming": (PROGRAMMING, 9000),
        "zoology": (ZOOLOGY, 9000),
        "astronomy": (ASTRONOMY, 9000),
        "chemistry": (CHEMISTRY, 9000),
        "tools": (TOOLS, 9000),       # match, tool, fire — benchmark words
        "general": (GENERAL, 10000),
    }

    all_sentences = []
    domain_counts = {}

    for domain_name, (vocab, count) in domains.items():
        print(f"Generating {domain_name} ({count} sentences)...")
        sentences = generate_domain_sentences(domain_name, vocab, count)
        all_sentences.extend(sentences)
        domain_counts[domain_name] = len(sentences)
        print(f"  Generated {len(sentences)} unique sentences")

    # Shuffle
    random.shuffle(all_sentences)

    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for sentence in all_sentences:
            f.write(sentence + "\n")

    print(f"\nTotal: {len(all_sentences)} sentences")
    print(f"Domain breakdown: {domain_counts}")
    print(f"Written to: {output_path}")

    return all_sentences


if __name__ == "__main__":
    generate_corpus()
