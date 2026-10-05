"""Build manuscript/current.md from title.txt, abstract.md, body.md and the reference list in
sources/literature_review.md. Only references that are cited in the body are kept.

    python manuscript/assemble.py

Prints the number of references, the word count of the abstract and of the body (appendix included).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def reference_key(ref):
    ym = re.search(r"\((\d{4}[a-z]?|n\.d\.)", ref)
    year = ym.group(1) if ym else None
    if re.match(r"^[^\W\d_][\w'’\- ]*,", ref) and not ref.startswith("*"):
        key = ref.split(",")[0]
    else:
        key = re.split(r"\.\s*\(|\.\*|\. ", ref.lstrip("*"))[0]
    return key.replace("*", "").strip(), year


def main():
    title = "# " + (HERE / "title.txt").read_text().strip()
    abstract = (HERE / "abstract.md").read_text().strip()
    body = (HERE / "body.md").read_text().strip()
    review = (ROOT / "sources" / "literature_review.md").read_text()
    refs = [r.strip() for r in review.split("## References", 1)[1].strip().split("\n\n") if r.strip()]
    kept = []
    for r in refs:
        key, year = reference_key(r)
        if key.startswith("Doe v"):
            pat = r"Doe v\. GitHub"
        elif year:
            pat = re.escape(key) + r".{0,90}?" + re.escape(year)
        else:
            continue
        if re.search(pat, body, re.S):
            kept.append(r)
    out = title + "\n\n" + abstract + "\n\n" + body + "\n\n## References\n\n" + "\n\n".join(kept) + "\n"
    (HERE / "current.md").write_text(out)
    abs_words = len(abstract.split("**Keywords")[0].replace("## Abstract", "").split())
    print(f"{len(kept)} references; abstract {abs_words} words; body {len(body.split())} words")


if __name__ == "__main__":
    main()
