"""Language-agnostic text utilities: tokenisation, sentence splitting, language detection.

Language detection is a dependency-free heuristic: Unicode script first, then
stop-word voting for Latin-script languages. It is good enough to tell the LLM
which language to answer in; it is not meant as a general-purpose classifier.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")

LANGUAGE_NAMES = {
    "en": "English", "sw": "Swahili", "ru": "Russian", "uk": "Ukrainian",
    "fr": "French", "es": "Spanish", "de": "German", "pt": "Portuguese",
    "it": "Italian", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "hi": "Hindi", "el": "Greek", "he": "Hebrew",
    "am": "Amharic", "th": "Thai", "und": "the same language as the question",
}

_STOPWORDS: dict[str, set[str]] = {
    "en": set(
        "the and is are of to in for on with what how why "
        "which who does do that this it be by an as from can".split()
    ),
    "sw": set(
        "na ya wa kwa ni za la katika hii kuwa cha vya je nini "
        "gani jinsi sana pia hiyo wao yake kama huo hizi hutumia mfumo kwamba lakini au".split()
    ),
    "fr": set(
        "le la les et est des du un une pour dans "
        "que qui quoi comment pourquoi avec sur pas au aux ce cette".split()
    ),
    "es": set("el la los las y es de del un una para en que qué cómo por con se su al lo como".split()),
    "de": set("der die das und ist von zu den mit für auf wie was warum nicht ein eine sich im dem des".split()),
    "pt": set("o a os as e é de do da um uma para em que como por com não se ao dos das".split()),
    "it": set("il lo la gli le e è di del della un una per in che come perché con non si al".split()),
}

# Words that would otherwise tie between languages; give English a little boost.
_STOP_ALL = set().union(*_STOPWORDS.values())


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; works for any script that separates words with spaces.

    CJK text (no spaces) is broken into character bigrams so BM25 still gets signal.
    """
    text = normalize(text).lower()
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if _is_cjk(tok[0]) and len(tok) > 1:
            tokens.extend(tok[i:i + 2] for i in range(len(tok) - 1))
        else:
            tokens.append(tok)
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text)]
    return [p for p in parts if p]


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
        or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF
    )


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:
            key = "cyrillic"
        elif 0x0600 <= cp <= 0x06FF:
            key = "arabic"
        elif 0x0590 <= cp <= 0x05FF:
            key = "hebrew"
        elif 0x0370 <= cp <= 0x03FF:
            key = "greek"
        elif 0x0900 <= cp <= 0x097F:
            key = "devanagari"
        elif 0x1200 <= cp <= 0x137F:
            key = "ethiopic"
        elif 0x0E00 <= cp <= 0x0E7F:
            key = "thai"
        elif 0x3040 <= cp <= 0x30FF:
            key = "kana"
        elif 0xAC00 <= cp <= 0xD7AF:
            key = "hangul"
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            key = "han"
        elif cp < 0x0250:
            key = "latin"
        else:
            key = "other"
        counts[key] = counts.get(key, 0) + 1
    return counts


def detect_language(text: str) -> str:
    """Return an ISO-639-1 code, or ``"und"`` when undetermined."""
    counts = _script_counts(text)
    if not counts:
        return "und"
    script = max(counts, key=counts.get)
    if script == "cyrillic":
        return "uk" if re.search(r"[іїєґІЇЄҐ]", text) else "ru"
    if script == "kana" or (script == "han" and counts.get("kana")):
        return "ja"
    simple = {"arabic": "ar", "hebrew": "he", "greek": "el", "devanagari": "hi",
              "ethiopic": "am", "thai": "th", "hangul": "ko", "han": "zh"}
    if script in simple:
        return simple[script]
    if script != "latin":
        return "und"

    words = [w for w in tokenize(text)]
    if not words:
        return "und"
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    # Swahili morphology hints: noun-class / verb prefixes
    # (Bantu words carry class/verb prefixes and almost always end in a vowel.)
    morph = sum(1 for w in words if re.fullmatch(r"(ku|wa|ki|vi|mi|ma|hu|ana|wana|ni|ali|ata)[a-z]{2,}[aeiou]", w))
    if scores["sw"] > 0 or (morph >= 2 and morph / len(words) >= 0.5):
        scores["sw"] += morph * 0.5
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "en"  # Latin script with no stop-words (e.g. a keyword query): assume English
    return best


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)
