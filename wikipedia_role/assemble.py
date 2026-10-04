"""Assemble essay_full_draft.md from the section drafts and the reference list in literature_review.md.

    python assemble.py         # v1, nine sections
    python assemble.py --v2    # trimmed, five sections (essay_v2_body.md)
Only references that are cited in the body are kept.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = ["01_introduction", "02_three_platforms", "03_what_the_bastion_is", "04_what_licence_compliant_reuse_leaves_out",
         "05_what_the_unease_tracks", "06_forms_of_stewardship", "07_what_wikipedia_should_change", "08_objections", "09_conclusion"]
TITLE = "# The last bastion is a community: Wikipedia, language models, and what licence-compliant reuse leaves out"
ABSTRACT = (HERE / "drafts" / "00_abstract.md").read_text().strip()


def main(v2=False):
    if v2:
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
    (HERE / ("essay_full_draft_v2.md" if v2 else "essay_full_draft.md")).write_text(out)
    print(len(kept), "references;", len(body.split()), "words in the body")


if __name__ == "__main__":
    import sys
    main("--v2" in sys.argv)
