"""Lexical markers of LLM prose.

The list is taken from the English Wikipedia essay "Signs of AI writing"
(https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing, read 2026-10-03), section on
overused words and phrases. We fixed it before looking at any data.
"""
import re

WORD_STEMS = [
    r"delv(?:e|es|ed|ing)", r"tapestr(?:y|ies)", r"testament", r"pivotal", r"intricac(?:y|ies)|intricate",
    r"meticulous(?:ly)?", r"underscor(?:e|es|ed|ing)", r"showcas(?:e|es|ed|ing)", r"vibrant", r"boasts?|boasting",
    r"nestled", r"enduring", r"bolster(?:s|ed|ing)?", r"garner(?:s|ed|ing)?", r"interplay", r"fostering",
    r"emphasi[sz]ing", r"exemplif(?:y|ies|ied)",
]
PHRASES = [
    r"stands as", r"serves as (?:a|an|the)", r"is a testament", r"rich (?:cultural )?(?:heritage|history|tapestry)",
    r"in the heart of", r"plays? an? (?:crucial|pivotal|vital|significant|key) role", r"commitment to",
    r"diverse array", r"valuable insights?", r"indelible mark", r"evolving landscape", r"natural beauty",
]
_WORD_RX = re.compile(r"\b(?:%s)\b" % "|".join(WORD_STEMS), re.I)
_PHRASE_RX = re.compile(r"\b(?:%s)\b" % "|".join(PHRASES), re.I)


def marker_counts(text):
    return len(_WORD_RX.findall(text)), len(_PHRASE_RX.findall(text))
