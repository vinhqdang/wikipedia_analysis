"""Assemble essay_full_draft.md from the section drafts and the reference list in literature_review.md.

    python assemble.py         # v1, nine sections
    python assemble.py --v2    # trimmed, five sections (essay_v2_body.md)
    python assemble.py --v6    # revision after round-3 review (essay_v6_body.md)
    python assemble.py --v5    # norms-and-impermanence version (essay_v5_body.md)
    python assemble.py --v4    # revision after round-1 review (essay_v4_body.md)
    python assemble.py --v3    # refocused on Wikipedia in the LLM era (essay_v3_body.md)
Only references that are cited in the body are kept.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = ["01_introduction", "02_three_platforms", "03_what_the_bastion_is", "04_what_licence_compliant_reuse_leaves_out",
         "05_what_the_unease_tracks", "06_forms_of_stewardship", "07_what_wikipedia_should_change", "08_objections", "09_conclusion"]
TITLE = "# The last bastion is a community: Wikipedia, language models, and what licence-compliant reuse leaves out"
ABSTRACT = (HERE / "drafts" / "00_abstract.md").read_text().strip()


def main(v2=False, v3=False, v4=False, v5=False, v6=False):
    global TITLE, ABSTRACT
    if v6:
        TITLE = "# Conditioned norms: impermanence and the fate of Wikipedia's authority in the era of language models"
        ABSTRACT = (HERE / "drafts" / "00_abstract_v6.md").read_text().strip()
        body = (HERE / "essay_v6_body.md").read_text().strip()
    elif v5:
        TITLE = "# Conditioned authority: impermanence and the fate of Wikipedia's norms in the era of language models"
        ABSTRACT = (HERE / "drafts" / "00_abstract_v5.md").read_text().strip()
        body = (HERE / "essay_v5_body.md").read_text().strip()
    elif v4:
        TITLE = "# What is Wikipedia for in the era of language models? An auditable practice, not a source of truth"
        ABSTRACT = (HERE / "drafts" / "00_abstract_v4.md").read_text().strip()
        body = (HERE / "essay_v4_body.md").read_text().strip()
    elif v3:
        TITLE = "# The last bastion is a community: Wikipedia in the era of language models"
        ABSTRACT = (HERE / "drafts" / "00_abstract_v3.md").read_text().strip()
        body = (HERE / "essay_v3_body.md").read_text().strip()
    elif v2:
        body = (HERE / "essay_v2_body.md").read_text().strip()
    else:
        body = "\n\n".join((HERE / "drafts" / f"{p}.md").read_text().strip() for p in PARTS)
    review = (HERE / "literature_review.md").read_text()
    refs = [r.strip() for r in review.split("## References", 1)[1].strip().split("\n\n") if r.strip()]
    kept = []
    for r in refs:
        ym = re.search(r"\((\d{4}[a-d]?|n\.d\.)", r)
        year = ym.group(1) if ym else None
        if re.match(r"^[^\W\d_][\w'’\- ]*,", r) and not r.startswith("*"):
            key = r.split(",")[0]
        else:
            key = re.split(r"\.\s*\(|\.\*|\. ", r.lstrip("*"))[0]
        key = key.replace("*", "").strip()
        pat = r"Doe v\. GitHub" if key.startswith("Doe v") else (re.escape(key) + r".{0,90}?" + re.escape(year) if year else None)
        if pat and re.search(pat, body, re.S):
            kept.append(r)
    out = TITLE + "\n\n" + ABSTRACT + "\n\n" + body + "\n\n## References\n\n" + "\n\n".join(kept) + "\n"
    (HERE / ("essay_full_draft_v6.md" if v6 else "essay_full_draft_v5.md" if v5 else "essay_full_draft_v4.md" if v4 else "essay_full_draft_v3.md" if v3 else "essay_full_draft_v2.md" if v2 else "essay_full_draft.md")).write_text(out)
    print(len(kept), "references;", len(body.split()), "words in the body")


if __name__ == "__main__":
    import sys
    main("--v2" in sys.argv, "--v3" in sys.argv, "--v4" in sys.argv, "--v5" in sys.argv, "--v6" in sys.argv)
