# Editorial decision and revision roadmap, round 1

Manuscript: `essay_full_draft_v3.md` (about 6,600 words of body, 72 references). Panel: journal-fit (R0), methodology (R1), domain (R2), perspective (R3), devil's advocate (R4), each reviewing blind from the manuscript alone. Panel calibration status: not calibrated; the reports simulate review and are not predictions of a real decision.

## Decision: Major Revision (all five seats agree)

Fit with the target venues (*Philosophy & Technology*, *Ethics and Information Technology*, *AI & Society*) is good and the topic is timely. The reframing of Wikipedia's value as auditability rather than truth, the relational account of credibility, the non-rival text versus rival practice distinction, the candour about limits, and the move of turning the fairness condition back on Wikipedia (Section 5) are the real strengths. The weaknesses concern the core normative argument and the shape of the paper, not polish.

## Consensus issues (raised by three or more seats)

1. **The duty argument does not carry its conclusion (R0 W1, R1 W1-W4, R2 W1-W4, R4 C2).** Fair play, natural duty and beneficence are each conceded in the text to fail at the premise that would bind developers, they ground different duties so their weakness does not sum, and the extension to an epistemic commons is declared "the essay's own" and not argued. The "strain" and "reliance" premises rest on the Foundation's self-report, a 0.14% token share and model-collapse papers that are contested (R1 W1-W2, R4 C3). The duty has no stated right-holder or directedness (R1 W4, R2 W2).
2. **Section 5 and the duty are tied in an unresolved way (R4 C1, R2 W10, R3 W6).** The reform of Wikipedia is presented both as a precondition of its claim on reusers and as independent of whether reusers help. Either reading leaves a gap. Section 5 is also not yet actionable: no actors, venues, costs or sequencing.
3. **Auditability is stipulated, not grounded (R0 W5, R1 W9, R2 W5, R4).** It is not tested against retrieval systems that cite sources, against the essay's own evidence that contestability is penalised, or against Hardwig's point that lay readers cannot audit. The contrast "auditability, not truth" is partly false: auditability is valued for error-correction and calibrated trust.
4. **Second-hand and thin sourcing (R0 W4, R1 W7 and W10, R2 W3).** Rawls, Kant, Klosko and others are cited through encyclopedia entries; one citation points to "sources noted in the bibliography", which the reader does not have; key empirical premises are Foundation self-reports flagged only in the conclusion.
5. **Scope and genre (R0 W2, R1 W6, R3 W1 and W3).** Three genres in one paper (moral argument, empirical sketch, policy brief); the Vietnamese remeasurement breaches the stated English-edition focus and over-reads a non-comparable lower bound; search engines and other reusers are sidelined.
6. **Repetition and residue (R0 W6, R1 W11).** Passages repeated across Sections 2, 5.2 and 5.3; working-process language ("own pull", "read 4 October 2026", "timed out"); abstract says firms "owe nothing" where the body says the licence is "silent".

## Seat-specific issues worth acting on

- **Factual error (R1 W5), now fixed:** Vietnamese was third lowest in the 2018 comparison, after Cebuano and Waray, not the lowest. Corrected in the essay and notes (commit 501b0653).
- **Missing literature (R2 W8-W9, R3 W7):** commons and infrastructure (Ostrom, Frischmann, Benkler and Nissenbaum), epistemic injustice and trust (Fricker, Goldman, Kitcher, Coady, Lackey, Longino), Wikipedia testimony (Tollefsen, Magnus), collective responsibility (Young, Kutz, Murphy), data governance (Viljoen, Delacroix, Jernite), exploitation (Wertheimer), the epistemic backstop (Rini). Titles were supplied as unverified search leads; each must be checked before use.
- **Klosko's "indispensable" condition and Arneson's free-rider threshold (R2 W2)**, Cicero supports the giver's duty and not the recipient's (R2 W11), Singer is not reliance-based (R2 W3).
- **Perspective gaps (R3 W2-W5, W8):** "the community" is not disaggregated (Foundation, affiliates, editions); the proposed route through published research risks re-inscribing the rule that creates the gap; licensing alternatives and Enterprise politics are not weighed; structural and regulatory routes are absent.
- **Devil's advocate cases (R4):** capture and weakness of small editions were illustrated by Croatian and Scots Wikipedia from the reviewer's memory, flagged for verification; they must not enter the essay unverified.

## Adjudication of Devil's Advocate CRITICAL issues

- **C1 (precondition or independent), validated:** a real structural gap. Must be resolved by choice, not left open.
- **C2 (cumulative case), validated** as stated: the essay gives no argument that the three grounds converge on one duty. Partly answerable if the claim is weakened to a pro tanto reason and the grounds are shown to share a premise (non-trivial benefit from a shared practice at small cost to the benefited).
- **C3 (reliance rests on self-report and 0.14%), validated, partly answerable:** the text does state the 0.14% and the contest over collapse, but concludes more than the evidence supports. Needs either evidence of substitutability or a weaker, conditional claim.

## Revision roadmap (ordered; the author decides each item)

**A. Decision on the paper's centre (author, before any rewriting).**
- *Option 1, recommended:* make the thesis Wikipedia's role (auditable community practice) and treat the reuse duty as a tentative pro tanto reason in one section; make Section 5 independent of it and say so. This fits the author's stated wish to focus on Wikipedia in the AI era and answers R4 C1 by choosing the second horn.
- *Option 2:* keep the duty as the centre. Then the paper needs a proper argument for the extension to an epistemic commons, a stated right-holder and directedness, an assessment of whether current Wikipedia meets the fairness precondition, and attention to Klosko's conditions, Arneson, collective responsibility and exploitation (R2 W1-W4, W7-W8).

**B. Must do under either option.**
1. Resolve C1 explicitly (A above).
2. Ground auditability: connect to Longino, Goldman, Fricker and Hardwig; test it against retrieval with citation and against the evidence that contestability is penalised; say who audits.
3. Weaken premises to what the evidence supports; mark Foundation self-reports where they carry premises, not only in the conclusion.
4. Remove or reposition the Vietnamese remeasurement (appendix or separate note) and restore the English-edition focus, or state the scope change and defend it.
5. Remove working-process language and duplicated passages; align abstract and body ("silent" not "owe nothing"); give Rawls a real locator.
6. Read primary or full texts for Rawls (sections 18, 19, 51), Nozick, Simmons, Klosko, Vetter et al. and the other items in `annotated_bibliography/8` and `9`.

**C. Should do.**
1. Add search engines and other reusers; disaggregate "the community"; discuss licensing alternatives and Enterprise as a political object; add a short acknowledgement of structural and regulatory routes.
2. Make Section 5 concrete (actors, venues, costs, sequence) or present it explicitly as a research agenda; address privacy risks in categorising deletions and republishing talk-page history.
3. Verify and add the missing literature; verify the capture cases before use.

**D. Could do.** Trim to the target journal's length; check authors' guidelines including any rule on disclosing writing tools.

## Questions for the author

1. Option 1 or Option 2 (A above)?
2. Is the Vietnamese edition a case you want in the paper (then scope it) or background (then drop it)?
3. Is the target a philosophy venue (then Section 5 shrinks) or a policy-facing one (then Section 5 grows)?
