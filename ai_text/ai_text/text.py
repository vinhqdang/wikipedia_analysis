"""Wikitext to prose conversion and simple text statistics."""
import re

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


_SENT = re.compile(r"(?<=[.!?])\s+")


def words(text):
    return re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", text.lower())


def n_words(text):
    return len(words(text))
