# v2: what changed from v1

v1 (`essay_full_draft.md`): 9 sections, about 9,200 words. v2 (`essay_full_draft_v2.md`, body in `essay_v2_body.md`): 5 sections, about 4,700 words.

| v2 | Built from v1 |
|---|---|
| 1. Introduction (includes the three-platform premise check) | 1, 2 |
| 2. What the bastion is | 3 (the "whose bastion" gaps are folded into one paragraph) |
| 3. What the licence leaves out, and what is owed (3.1 to 3.4) | 4, 5, 6 |
| 4. What Wikipedia should change | 7 |
| 5. Objections and conclusion | 8, 9 |

Cuts: evidence detail on traffic and contributions trimmed; the four forms of stewardship compressed to one paragraph; the separate criteria list folded into the discussion; the self-reference to the repository's detection study removed.

v1 files in `drafts/` are kept as the fuller source. Regenerate with `python assemble.py --v2`.

# v3: refocus on Wikipedia in the LLM era

`essay_full_draft_v3.md` (body `essay_v3_body.md`, abstract `drafts/00_abstract_v3.md`), about 4,100 words, 5 sections: Introduction; What the bastion is; Wikipedia and the language models (3.1 licence silent, 3.2 what models take and what it costs, 3.3 a short note on betrayal); Sustaining the practice (4.1 what reusers can do, 4.2 what Wikipedia should change); Objections and conclusion.

Betrayal shrinks from a subsection of its own, with fair play and exploitation, to one short note. Research question is now "what role can and should Wikipedia play in the LLM era". Regenerate with `python assemble.py --v3`.

## v3 update: philosophical grounding

Section 3.3 is rewritten as "What is owed, and why": fair play (Hart, Rawls, Nozick's objection, Simmons, Klosko), natural duty (Rawls), beneficence (Singer, Kant), and Cicero's lamp with a distinction between non-rival text and a rival practice. The statement that the principle is the essay's own is replaced by the claim that only the extension to an epistemic commons is. Reading status is in `annotated_bibliography/7_philosophical_foundations.md`.

## v3 update: Wikipedia's own change as a section

Former 4.2 is now Section 5, "What Wikipedia should change", about 1,200 words: a framing argument that a claim on others' upkeep is only as strong as the practice is open and fair (Rawls's fairness principle requires a just institution), then recommendations on auditability and provenance, newcomer survival (with the risk that defences against machine-generated text repeat the pattern Halfaker et al. describe), notability and oral knowledge by documented procedure, and publishing the terms and use of reuse income. Section 4 is now "What reusers can do"; objections and conclusion are Section 6. Body about 5,800 words.
