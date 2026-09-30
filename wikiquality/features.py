"""Language-agnostic structural features and label-leak scrubbing for wikitext."""
import re

import numpy as np
import pandas as pd

# Templates and categories that announce the assessment itself. They sit in the
# wikitext of featured/good articles and stubs and let a classifier read the
# label instead of judging quality, so they are removed before any modelling.
_LEAK_NAMES = (
    r"featured article|featured list|featured topic|good article|good topic|fa|ga"
    r"|[\w -]*stub[\w -]*"
    r"|article de qualité|article potentiellement de qualité|bon article|ébauche[\w ]*|bon début"
    r"|избранный список[\w ]*|избранная статья|избранн\w*|изб[\w -]*|хорошая статья|добротная статья|заготовка[\w ]*|[\w ]*-заготовка"
)
LEAK_TEMPLATE = re.compile(r"\{\{\s*(?:%s)\s*(?:\|[^{}]*)?\}\}" % _LEAK_NAMES, re.I)
LEAK_CATEGORY = re.compile(
    r"\[\[\s*(?:category|catégorie|категория)\s*:[^\]]*(?:stub|ébauche|заготовк|featured|good article|"
    r"article de qualité|bon article|избранн|хорош)[^\]]*\]\]",
    re.I,
)


def scrub(text):
    return LEAK_CATEGORY.sub("", LEAK_TEMPLATE.sub("", text))


_IMG = r"\.(?:jpe?g|png|svg|gif|tiff?|webp)"
_COUNTS = {
    "n_refs": re.compile(r"<ref[\s>/]", re.I),
    "n_templates": re.compile(r"\{\{"),
    "n_links": re.compile(r"\[\["),
    "n_ext_links": re.compile(r"\[https?://", re.I),
    "n_h2": re.compile(r"^==[^=]", re.M),
    "n_h3": re.compile(r"^===[^=]", re.M),
    "n_h4plus": re.compile(r"^====", re.M),
    "n_images": re.compile(_IMG, re.I),
    "n_categories": re.compile(r"\[\[\s*(?:category|catégorie|категория)\s*:", re.I),
    "n_list_items": re.compile(r"^[*#]", re.M),
    "n_tables": re.compile(r"^\{\|", re.M),
    "n_numbers": re.compile(r"\d+"),
    "n_quotes": re.compile(r"&quot;|«|»|“|”"),
}


def structural(text):
    feats = {k: len(rx.findall(text)) for k, rx in _COUNTS.items()}
    feats["n_chars"] = len(text)
    words = text.split()
    feats["n_words"] = len(words)
    feats["n_lines"] = text.count("\n") + 1
    feats["avg_word_len"] = float(np.mean([len(w) for w in words])) if words else 0.0
    feats["refs_per_kchar"] = 1000 * feats["n_refs"] / max(len(text), 1)
    feats["links_per_kchar"] = 1000 * feats["n_links"] / max(len(text), 1)
    return feats


def structural_frame(texts):
    return pd.DataFrame([structural(t) for t in texts])


_TEMPLATE_INNER = re.compile(r"\{\{[^{}]*\}\}")
_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_FILE_LINK = re.compile(r"\[\[\s*[^\[\]|:]{2,15}:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]")
_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]")
_EXT = re.compile(r"\[https?://\S+(?:\s+([^\]]*))?\]")
_TAG = re.compile(r"<[^>]+>")
_MARKUP = re.compile(r"'{2,}|^[=*#:;]+\s*|\s*=+\s*$", re.M)
_WS = re.compile(r"[ \t]+")


def plain_text(wikitext):
    """Crude wikitext to prose conversion, good enough for a text encoder."""
    t = _COMMENT.sub("", wikitext)
    t = _REF.sub("", t)
    t = _TABLE.sub("", t)
    for _ in range(4):
        t = _TEMPLATE_INNER.sub("", t)
    t = _FILE_LINK.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _EXT.sub(lambda m: m.group(1) or "", t)
    t = _TAG.sub("", t)
    t = _MARKUP.sub("", t)
    return _WS.sub(" ", t).strip()
