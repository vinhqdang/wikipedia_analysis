# Wikipedia and language models: an essay on changing norms

A philosophy-and-technology essay on what the fate of Wikipedia shows about how social norms depend on conditions, what the people who reuse Wikipedia owe it, and what is worth keeping. Conceptual; no new empirical study except two small data checks kept under `evidence/`.

Current manuscript: [`manuscript/current.md`](manuscript/current.md) (title, abstract, body, references). Rebuild it with:

```
python manuscript/assemble.py
```

## Layout

| Folder | Contents |
|---|---|
| `manuscript/` | The current essay: `title.txt`, `abstract.md`, `body.md` (sections 1-8 and Appendix A), `assemble.py`, and the built `current.md` |
| `sources/` | `literature_review.md` (the reference list the build reads, plus the narrative review from the first stage), `annotated_bibliography/` (what was read, how, and the exact passages used), `primary_texts/` (local copies only; PDFs are git-ignored) |
| `reviews/` | Simulated peer-review reports (`round1`-`round4`), the editorial decision of round 1, and `REVISION_LOG.md` listing what each round changed and what is still open |
| `evidence/` | Data pulls and scripts used for the checks in the essay (Wikimedia statistics, the Wikidata cultural-context lower bound) |
| `archive/` | Earlier versions (`versions/`: v1 to v5 bodies and assembled drafts, the v1 section drafts), and planning notes (`planning/`) |

## Status (5 October 2026)

Sixth draft. The simulated panels moved from Major Revision on all seats (round 1) to Minor Revision on journal fit and a narrowed Major Revision on the philosophical argument (round 4). Open items are in `reviews/REVISION_LOG.md`. Still to read in the original: Simmons (1979), Klosko (1992), Hart (1955), Craig (1990), and the Pāli suttas in `sources/annotated_bibliography/14_suttas_to_read.md`. Rawls (1999), Nozick (1974) and Kant (Semple translation) have been read directly (`13_primary_texts_read.md`).

## Paths changed on 5 October 2026

Older notes refer to files by their previous locations. Mapping: `annotated_bibliography/` is now `sources/annotated_bibliography/`; `literature_review.md` is `sources/literature_review.md`; `review_round1`-`review_round4` are `reviews/round1`-`round4`; `essay_v*_body.md` and `essay_full_draft*.md` of versions 1 to 5 are in `archive/versions/`; `essay_v6_body.md` is `manuscript/body.md`; `drafts/00_abstract_v6.md` is `manuscript/abstract.md`; outlines and review notes are in `archive/planning/`.

## Books and scans

The repository is public. Copyrighted books and scans must not be committed: the essay quotes short passages with locators (see `sources/annotated_bibliography/13_primary_texts_read.md`), and PDFs placed in `sources/primary_texts/` are ignored by git.
