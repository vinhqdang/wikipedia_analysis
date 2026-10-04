# 8. Evidence added 4 October 2026 for Section 5

Added to close three gaps flagged in the draft. Reading status per source.

| Source | Used for | Status |
|---|---|---|
| Morgan & Halfaker (2018), OpenSym '18 | Teahouse invitation trial, English Wikipedia, 2014-15 | Read in full (7 pages). Control group 3,092, invited 11,674; invitees significantly more likely to survive at 3-4 weeks (p = .0479) and at 5+ edits 2-6 months (p = .0311); other windows not significant but same direction; analysis is intention-to-treat, only 331 invitees took part |
| Warncke-Wang, Ho, Miller & Johnson (2023), PACM HCI 7 (CSCW) | Newcomer Homepage trial, 27 non-English wikis, 244,060 accounts, Feb-May 2021 | Read via a page summary of the arXiv HTML; volume and issue number to confirm |
| Zhou, Cho & Terveen (2026), CSCW '26 | Interviews with 16 editors on LLM-assisted edits and newcomers | Abstract and sections 4.2-4.3 read from the PDF |
| Wikimedia Enterprise data dictionary | What provenance and credibility fields reusers receive | Page fetched and parsed on 4 October 2026; fields: version.scores (revertrisk, referencerisk, referenceneed), editor name, comment, previous_version identifier and number_of_characters; no talk-page field found |
| Own pull, Wikimedia statistics API and siteinfo, 4 October 2026 | Vietnamese and English edition scale | `evidence/vi_en_wikimedia_2026-10-04.json`, script `evidence/pull_vi_en.py`. `activeusers` is the siteinfo count of users active in the past 30 days; the 5-99 edit editor series failed to return |

Leads not used: the Wikipedia Diversity Observatory dataset (Miquel-Ribé & Laniado, 2019, ICWSM) covers 300 editions and could be used to remeasure the Vietnamese cultural-context share; a 2026 AI & Society article on Wikipedia's governance of AI-generated content (paywalled, not read); Machines in the Margins, a systematic review of automated content generation for Wikipedia (arXiv 2509.22443, not read).

## Remeasuring the Vietnamese cultural-context share (4 October 2026)

The Diversity Observatory's own datasets (wdo.wmcloud.org) and code repository could not be reached from this environment, so its classifier was not run. A uniform lower bound was computed instead from the Wikidata query service: for each edition, the number of articles whose item carries a country, citizenship, country-of-origin or country-for-sport property pointing to the language's territory, or a located-in (P131) chain ending in it, excluding category and template items, divided by the edition's current article count from the site statistics. Territories are the country of the language with the main exceptions simplified (Korean: South and North Korea; Waray and Cebuano: the Philippines). Scripts and results: `evidence/ccc_lower_bound.py`, `evidence/wdq.py`, `evidence/ccc_wikidata_lower_bound_2026-10.json`.

Results (share of articles): Waray 0.17%, Cebuano 0.36%, **Vietnamese 1.79% (23,410 of 1,304,848)**, Swedish 9.7%, Malay 10.7%, Ukrainian 10.8%, Romanian 11.2%, Persian 12.7%, Bengali 12.8%, Hungarian 13.5%, Hebrew 13.5%, Thai 13.9%, Turkish 14.7%, Greek 17.0%, Indonesian 19.2%, Finnish 19.7%. Korean, Polish, Czech, Japanese and Hindi timed out at the service's 60-second limit and are omitted.

Caveats. This is not the 2018 measure, so the 2.5% and 1.8% are not directly comparable; it counts only one family of features and so understates cultural-context content in every edition. Articles without a Wikidata item, or with sparse properties, count as zero, which may penalise editions with weaker Wikidata linkage; the comparison assumes linkage is not much worse for Vietnamese than for the others, which was not tested. Denominators are current article counts and include some pages that are not items with sitelinks. The two lowest editions, Waray and Cebuano, are known for bot-created content, which fits the 2018 explanation but was not tested here.
