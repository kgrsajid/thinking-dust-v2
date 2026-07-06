"""German language configuration for Thinking Dust.

SKELETON — fill in the German word sets to enable German support.

To complete:
1. Fill in all word sets below with German equivalents
2. Test with: python -c "from td.languages import de; print(de.LANG_CONFIG)"
3. See DEVELOPMENT.md "Adding a New Language" for detailed guide

Reference: Universal Dependencies (Nivre et al., 2016)
Reference: Jauhar et al. (2015), *SEM, pp. 299-308
"""

from . import LanguageConfig, register_language

# ── German Stop Words ─────────────────────────────────────────────
# Source: spaCy German stop word list (de_core_news_sm)
# Fill in with: python -c "import spacy; nlp = spacy.load('de_core_news_sm'); print([t.text for t in nlp('Der die das ein eine') if t.is_stop])"
STOP_WORDS = frozenset({
    "der", "die", "das", "ein", "eine", "einer", "eines", "einem", "einen",
    "und", "oder", "ist", "sind", "war", "waren", "sein", "gewesen",
    "haben", "hat", "hatte", "werden", "wird", "wurde",
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mir", "dir", "ihn", "ihnen", "mich", "dich", "uns",
    "mein", "dein", "sein", "ihr", "unser", "euer",
    "in", "von", "zu", "mit", "auf", "für", "an", "bei", "nach", "aus",
    "über", "unter", "vor", "zwischen", "durch", "gegen", "ohne",
    # TODO: Add more German stop words
})

# ── German Prepositions ───────────────────────────────────────────
# Source: German grammar — common prepositions
PREPOSITIONS = frozenset({
    "auf", "aus", "bei", "mit", "nach", "seit", "von", "zu",
    "in", "an", "für", "über", "unter", "vor", "zwischen", "durch",
    "gegen", "ohne", "um", "bis", "ab", "entlang",
    # TODO: Add more German prepositions
})

# ── German Pronouns ───────────────────────────────────────────────
POSSESSIVE_PRONOUNS = frozenset({
    "sein", "seine", "seiner", "seines", "seinem", "seinen",
    "ihr", "ihre", "ihrer", "ihres", "ihrem", "ihren",
    "mein", "meine", "meiner", "meines", "meinem", "meinen",
    "dein", "deine", "deiner", "deines", "deinem", "deinen",
    "unser", "unsere", "unserer", "unseres", "unserem", "unseren",
    "euer", "eure", "eurer", "eures", "eurem", "euren",
    # TODO: Add accusative/dative forms
})

ENTITY_PRONOUNS = frozenset({
    "er", "sie", "es", "sie",  # he/she/it/they
    "ihn", "ihm",  # him/dative
    "ihnen",  # them/dative
    # TODO: Add more German pronouns
})

# ── German Discourse Deixis Verbs ─────────────────────────────────
# Abstract/cognitive verbs in German.
# "Das zeigt, dass..." → "das" is discourse deixis.
# "Das überrascht mich" → "das" has a real referent (NOT deixis).
#
# Reference: Jauhar et al. (2015), *SEM — two-stage approach
DISCOURSE_DEIXIS_VERBS = frozenset({
    # Verbs of demonstration/implication
    "zeigen", "beweisen", "bedeuten", "andeuten", "enthüllen",
    "bestätigen", "implizieren", "veranschaulichen", "widerspiegeln",
    # Verbs of causation/result
    "ergeben", "führen", "verursachen", "ermöglichen", "erlauben",
    "verhindern", "erfordern", "betreffen", "beeinflussen",
    # Verbs of emotional response
    "überraschen", "schockieren", "erfreuen", "ärgern", "aufregen",
    # TODO: Verify with native speaker
})

# Subset for "it" (German: "es")
DISCOURSE_DEIXIS_IT_VERBS = frozenset({
    "zeigen", "beweisen", "bedeuten", "andeuten", "enthüllen",
    "bestätigen", "implizieren", "veranschaulichen", "widerspiegeln",
    # TODO: Verify with native speaker
})

# ── Register German ───────────────────────────────────────────────
register_language(LanguageConfig(
    code="de",
    name="German",
    stop_words=STOP_WORDS,
    prepositions=PREPOSITIONS,
    possessive_pronouns=POSSESSIVE_PRONOUNS,
    entity_pronouns=ENTITY_PRONOUNS,
    discourse_deixis_verbs=DISCOURSE_DEIXIS_VERBS,
    discourse_deixis_it_verbs=DISCOURSE_DEIXIS_IT_VERBS,
))
