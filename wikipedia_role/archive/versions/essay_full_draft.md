# The last bastion is a community: Wikipedia, language models, and what licence-compliant reuse leaves out

## Abstract

Wikipedia, once barred from classroom citation, is now invoked as a safeguard against machine-generated falsehood, even as its human readership falls and the firms that train language models on its text owe nothing under its licence by way of return. This essay asks what AI developers owe the community behind Wikipedia when their reuse complies with its licence, and what the answer shows about why Wikipedia matters. It argues that Wikipedia's standing as a source of truth is relational, and that what deserves defence is a community-held practice of verification whose distinctive property is auditability (provenance, contestability, answerability) and not truth. The licence is silent on training, as its stewards concede. The common charge of betrayal is apt only where a commitment was made and broken; for reuse in general the unease tracks reliance without upkeep, which grounds a proportionate duty of stewardship that runs to the practice and not to donors. Forms of stewardship are assessed against criteria drawn from the argument, and implications for Wikipedia's own practice are set out. The argument is conceptual, draws on public documents and the scholarly literature, and notes where the evidence is thin.

**Keywords:** Wikipedia; large language models; knowledge commons; social epistemology; reciprocity; stewardship

## 1. Introduction

In 2007 the history department at Middlebury College voted to bar its students from citing Wikipedia, after a historian found near-identical errors in a set of examinations and traced them to Wikipedia entries (Jaschik, 2007; Waters, 2007). Two years earlier a *Nature* exercise had found that the encyclopedia's science entries were not much less accurate than Britannica's (Giles, 2005). Students were already using it, and so were some scientists (Head & Eisenberg, 2010; Giles, 2005). Wikipedia was at once a convenience, a suspect source, and a thing teachers told students not to cite.

Today the Wikimedia Foundation can publish a statement titled "In the AI era, Wikipedia has never been more valuable" (Wikimedia Foundation, 2025d). It says that almost all large language models train on Wikipedia datasets, and it asks the companies that build them to credit the human contributors and to pay for access (Miller, 2025; Wikimedia Foundation, 2025d). In the same period human readership fell by roughly 8%, a fall the Foundation attributes in part to search engines and chatbots that answer questions directly, often from Wikipedia's own content (Miller, 2025). A source once scorned has become something others rely on, and the people who maintain it are seeing fewer readers. Meanwhile other places where people shared knowledge have gone different ways: Stack Overflow's question volume has collapsed, and Reddit has grown.

This essay asks what, if anything, AI developers owe to the community behind Wikipedia when their reuse of it complies with its licence, and what the answer shows about why Wikipedia matters in the era of language models. It argues in four steps.

1. Wikipedia's standing as a source of truth is relational. It never claimed truth, only verifiability, and its reputation has moved with the set of alternatives against which it is compared. What is worth defending is not a body of content but a practice of verification carried by a community (Section 3).
2. The licence under which AI developers reuse the content says nothing about training and nothing about returning anything to the source, and the stewards of the licence say as much (Section 4).
3. The common reaction that reuse is a betrayal of the community is better read as an everyday label for an unease than as a technical charge. The unease tracks reliance without upkeep, and it grounds a duty of stewardship that runs to the practice and not to donors (Sections 5 and 6).
4. If the bastion is a community, its defence lies as much inside as outside: in how it treats newcomers, how it handles what counts as knowledge, and how openly it deals with those who reuse it (Section 7).

The essay does not claim that Wikipedia is failing, that AI reuse of it is unlawful, or that the harm to it is large. The evidence for decline is modest and partly contested, and no authority found establishes a breach of its licence. The argument is conceptual and rests on public documents and the scholarly literature. Its empirical premises are stated with their dates and sources, and where the evidence is thin the text says so. The focus is the English-language edition, with a few notes on others.

## 2. Three platforms, three fates

It is common to say that places where people shared knowledge are dying and to ask whether Wikipedia will follow. The claim needs a measure. A knowledge community can lose its contributors, its readers, or its business model, and these do not move together. Three cases show it.

*Stack Overflow* lost its contributors. Counting questions that are still visible through the Stack Exchange interface (deleted questions are excluded), monthly new questions were 207,171 in March 2014, 105,745 in October 2022, 52,337 in October 2023, 5,262 in October 2025 and 1,179 in July 2026. The fall began long before ChatGPT and it accelerated after it. Comparing Stack Overflow with platforms where ChatGPT access was limited, del Rio-Chanona et al. (2024) estimate about a 16% fall in weekly posts after its release, concentrated in widely used languages, and Burtch et al. (2024) find steep falls on topics where ChatGPT performs well and among newer users.

*Reddit* did not. Burtch et al. (2024) find no decline in Reddit communities in their window, which they tentatively link to the social ties that keep people engaged. Reddit's own filings for 2025 report revenue of $2.2 billion, up 69%, and 121.4 million daily active uniques in the fourth quarter, up 19% (Reddit, Inc., 2026a, 2026b). Its "other revenue" line was $140 million for the year, and the annual report says the company continues to explore content-licensing agreements. A community of conversation is growing and looking for ways to monetise its archive.

*Wikipedia* has lost some attention and kept its scale. The Foundation reports nearly 15 billion views a month (Wikimedia Foundation, 2026a) and, for human pageviews, a fall of roughly 8% against 2024 (Miller, 2025); a pull from the public pageviews interface for English Wikipedia gives falls of 6.7% for 2025 and 7.9% for 2026 over the same nine months of the previous year. The academic evidence on engagement is mixed (Reeves et al., 2025; Lyu et al., 2025), as Section 4 reports. Its revenue in fiscal 2024-25 was $208.6 million, above target (Wikimedia Foundation, 2025c).

Three things follow. The premise that Stack Overflow and Reddit are both dead holds for the first and not for the second. The claim that Wikipedia shares their fate is not supported by these data: the studies disagree on whether it has lost contributors, its readers have declined modestly, and its finances in fiscal 2024-25 were above target. And the cases mark out what is at stake in each. Stack Overflow shows a community of answers displaced by a tool that supplies answers. Reddit shows a community held together by social ties, and a business that profits from its archive. Wikipedia is a practice of verification whose attention is shifting to intermediaries that use its output. The essay's question is not whether Wikipedia is dying. It is what the position of a widely relied-upon practice, with fewer readers and an unchanged load of work, implies for those who rely on it.

One caution applies to all three. These are associations in time. Stack Overflow's decline began before the tools that are said to explain it, and causal estimates exist only for some platforms and windows.

## 3. What the bastion is

### 3.1 Verifiability, not truth

Calling Wikipedia a source of truth credits it with a claim it does not make. Its verifiability policy asks editors to "base articles on reliable, independent, published sources with a reputation for fact-checking and accuracy", and says that its content "is determined by published information rather than editors' beliefs, experiences, or previously unpublished ideas" (Wikipedia contributors, n.d.). For years the principle was summed up as "verifiability, not truth", a slogan that the policy page now describes as historical wording. Garfinkel (2008) read it as an appeal to the authority of other publications. What Wikipedia promises, on its own terms, is provenance: a reader can check where a statement came from. It does not promise that the statement is true. A representative of the Wikimedia Foundation said as much when Middlebury College's history department barred students from citing it: according to the historian who proposed the rule, he agreed with the position because Wikipedia is, like a print encyclopedia, a tertiary source (Waters, 2007).

"Source of truth" can mean three different things: that the content is accurate, that its origin can be traced, or that people defer to it. The policy asserts the second. The evidence on the first is comparative and narrow. In the *Nature* exercise, expert reviewers compared matched science entries without knowing which encyclopaedia each came from; the average Wikipedia entry had around four inaccuracies against about three for Britannica, and the difference was "not particularly great" (Giles, 2005). A comparison of 100 drugs against pharmacology textbooks found 99.7% factual accuracy in Wikipedia, with lower completeness (Kräenbring et al., 2014). Both studies are domain-specific, and neither shows that Wikipedia is true in general. The third meaning is conferred from outside. Graham et al. (2014) report, citing earlier work, that Wikipedia appeared on the first page of 99% of 1,000 Google searches for nouns, and Ford and Wajcman (2017) observe that search engines and fact boxes put its statements in front of readers until the facts fade into the background and are black-boxed. So the phrase "source of truth" mostly names authority that others have granted, and its common use runs ahead of what the policy claims and what the accuracy studies show.

The provenance claim also makes Wikipedia's standing derivative. Because the policy sends readers to published sources, a defect in those sources passes through. Hardwig (1985) argued that a chain of appeals to authority "must end somewhere", and that if it is to be sound it must end with "someone who possesses the necessary evidence" (p. 337). If the outlets and papers that Wikipedia cites come to contain errors, promotion or machine-generated text, the foot of the chain weakens whatever the encyclopedia's own care. The literature gives estimates of machine-generated text inside Wikipedia: detectors flagged over 5% of newly created English articles in one study, offered as a lower bound (Brooks et al., 2024), and another estimates LLM-related change of about 1% in some categories (Huang et al., 2025). The review behind this essay found no measurement of contamination in the external sources it relies on. That is an open empirical question, and the argument of this essay does not depend on its answer; it only shows where the bastion's exposure lies.

### 3.2 Legitimacy by comparison

The early philosophical defence of Wikipedia was comparative from the start. Fallis (2008) argued that its epistemic effects were probably positive because its accuracy was comparable to that of traditional encyclopedias and better than that of other free sources. The *Nature* result was parity, not superiority, and an information scientist quoted in the same report put the point as a comparison with the standard being used: print encyclopaedias "are often set up as the gold standards of information quality against which the failings of faster or cheaper resources can be compared", and the findings "remind us that we have an 18-carat standard, not a 24-carat one" (Giles, 2005, p. 901, quoting Michael Twidale). The question was already what a reader would otherwise use.

Its reputation then moved with the comparison class. In 2007 the history department at Middlebury College voted to bar students from citing Wikipedia (Jaschik, 2007). The historian who proposed the rule wrote that its open-source method risks conflating facts with popular opinion, that scholarship requires accountability, and that readers of an entry never know who edited it last, since most editors leave only a handle; he granted that Wikipedia compared favourably with other tertiary sources in the sciences and was spottier on history (Waters, 2007). His objection was aimed at anonymity and at popularity as a form of validation, a worry about answerability more than about accuracy. The comparison class was print reference and expert review. Practice ran ahead of reputation: students were using Wikipedia as a starting point for background reading (Head & Eisenberg, 2010), and in the survey that accompanied the *Nature* exercise, more than 70% of over 1,000 authors had heard of it and 17% of those consulted it weekly, while fewer than 10% helped update it (Giles, 2005). By the generative era the alternative is a system that produces fluent text without a visible chain of sources. The Wikimedia Foundation states that almost all large language models train on Wikipedia datasets (Miller, 2025), and a vendor analysis of about 730,000 ChatGPT conversations found Wikipedia in 5% of citations (Punturo, 2026; vendor data, to be treated with caution). The same artefact now plays opposite roles: unreliable when set against the expert-reviewed print reference, reliable when set against a model that may fabricate.

This suggests a structure for the claim. Credibility is a three-place relation: a source is credible for some purpose, relative to the alternatives open to the reader. Goldman's (2001) problem, how laypeople should choose among rival experts, already has this shape. What changed between 2007 and now was less the encyclopedia than the set it is compared with.

The story of a tool that is scorned and then rehabilitated needs care. It fits a pattern in which concern about each new technology resurges, mostly with regard to children, while research on it restarts without a theoretical baseline built on earlier technologies (Orben, 2020). Orben's own example shows the comparison effect: radio dramas were once feared as an addictive intruder, yet many parents today "would enthusiastically welcome" them if they displaced children's use of phones (p. 1144). Her analysis concerns young people and psychological research, so it supports the pattern for knowledge tools only by extension. The best-known instance of the alarm over search engines, the "Google effects on memory" experiment (Sparrow et al., 2011), did not replicate in a large replication project (Camerer et al., 2018), although its authors have pointed to differences in design. So the rehabilitation of an older tool may correct an overstated alarm and not reflect an improvement in the older tool. For generative AI the evidence on harm is early and mostly preprints (Kosmyna et al., 2025; Gerlich, 2025). Two limits on this section should be stated. No study tests how a source's standing changes with its comparison class, so the relational claim is a conceptual argument that the cases above illustrate and do not prove. And the broad accuracy comparisons are from 2005 to 2015, so the recent revival of Wikipedia's standing does not rest on a new measurement against generative systems.

### 3.3 The community as knower

If Wikipedia's claim is provenance and its standing is relational, what is worth defending? Hardwig supplies a way to answer. His argument about expert dependence ends with a conclusion he calls uncomfortable: where no individual possesses all the evidence for a proposition, one may have to accept "that there is knowledge that is known by the community, not by any individual knower", and he cites Peirce's view that the community of inquirers is the primary knower (Hardwig, 1985, p. 349). A Wikipedia article is the output of that kind of distributed checking. Editors verify sources, contest one another's claims and apply rules, and the page records the outcome. Tollefsen (2009) asks whether such entries should be treated as individual or collective testimony and how to assess the trustworthiness of a collective source, and the answer depends on the process behind the entry, not on any one author. Magnus (2009) points out that the usual signs of reliability, such as credentials and track record, are frustrated by anonymity, and Sanger (2009) argues that quality depends on voluntary deference to expertise that anonymity and aggressive editors drive away. On this view readers who trust Wikipedia are trusting a practice by which claims survive scrutiny.

Menking and Rosenberg (2021) make a related point from feminist epistemology. They argue that the first of Wikipedia's five pillars gives priority to the product over the process of knowledge production, and they propose attending to the "process of aggregation", the back-end disputes between editors about sources and reliability. Wikipedia, they write, is "more than just an encyclopedia" (p. 468). They also note that accuracy and reliability can be used to assert authority "precisely because authority can always be questioned" (p. 469). This property is what an answer from a generative system lacks by default: a visible route by which a reader can challenge a claim and have the challenge recorded.

The practice has a carrying capacity, and it is people. Halfaker et al. (2013) report that the number of active English editors peaked in March 2007 at 56,400 and has declined since. They show that the share of promising newcomers has held steady while fewer of them survive their first contributions, and they implicate the restrictiveness of the main quality-control mechanism and of tools that reject contributions; they also find that formal mechanisms for revising norms have calcified against changes proposed by newer editors. The machinery that makes the practice reliable has thinned the community that carries it.

Three properties describe what the community provides. They are a working vocabulary for the rest of the essay, not terms drawn from the literature: *provenance* (a claim points to a published source), *contestability* (any claim can be challenged through a recorded process), and *answerability* (a community with rules answers for what stands). Together they can be called auditability. The claim of this section is that auditability, not truth, is what makes Wikipedia worth defending, and that it lives in a practice and the people who carry it. Traffic and content volume are therefore poor measures of its health. The better measures are whether newcomers can still enter and whether claims can still be contested.

### 3.4 Whose bastion

A bastion has walls, and the literature documents where these have gaps. Graham et al. (2014) found that, despite millions of hours of volunteer labour, the encyclopedia remains uneven and clustered, with little content about much of the world. A model with a few conditions explained 71% of the variance, but some regions stayed well below their expected values, so better connectivity is necessary but not sufficient. On gender, Ford and Wajcman (2017) argue that being a Wikipedian requires sociotechnical expertise that is coded as male, so that those who master its technocratic system of representation become its power brokers. Tripodi (2023) combined ethnography with 38 months of deletion data on 22,174 biographies. The share of biographies about women rose from 16.83% to 18.25% over the period, yet women were consistently over 25% of those nominated for deletion. Of the nominated biographies that were kept, about 25% of the women's and 17% of the men's were retained, which she reads as women being more often wrongly classed as non-notable; that reading is an inference from "keep" decisions, and she cites a study that found women academics were not more likely to be deleted. Hill and Shaw (2013) estimate that the share of women among editors is higher than first reported, at 16.1% globally, but the gap remains.

The rules that make the encyclopedia verifiable also decide what it can contain. Gallert and van der Velden (2014) describe a catch-22 for indigenous knowledge: the reliable-sources rule excludes oral transmission, and knowledge keepers are judged too close to their subjects to be neutral (their text is an unedited preprint, and only its opening section was read for this essay). Menking and Rosenberg (2021) argue that the pillars entrench the gender gap and report that editors who challenged them have been punished, with topic bans in 2014. Language editions differ too: in a comparison of 40 editions, Vietnamese Wikipedia had the lowest share of culturally specific content, 2.5%, which the authors link to heavy bot-generated translated content (Miquel-Ribé & Laniado, 2018). These absences follow from the same rules that give the bastion its legitimacy. If knowledge must be published and its subjects notable by the standards of a particular community, then gaps in the published record become gaps in the encyclopedia. "The last bastion of human knowledge" overstates the case. What is being defended is published, verifiable knowledge as judged by a particular community.

This does not make the bastion worthless. Shi et al. (2019) find that politically diverse editorial teams write higher-quality articles, which implies that these gaps weaken the practice itself and are not merely a fairness concern. And the property that distinguishes the practice, contestability, is also what makes its gaps visible and open to repair. Tripodi's data exist because deletion debates are public, and edit-a-thons exist because the record shows which biographies are missing. A closed system's gaps cannot be audited in the same way. The bastion that remains worth defending is a verification practice carried by people, with known and contestable holes.

If so, what reuse by AI developers does to the people matters more than what it does to the pages. Section 4 turns to what the licence says about that and what it leaves out.

## 4. What licence-compliant reuse leaves out

### 4.1 What the licence promises

Contributors to Wikipedia license their text under the Creative Commons Attribution-ShareAlike 4.0 licence (CC BY-SA 4.0). The Wikimedia Foundation's terms of use say that these licences "do allow commercial uses of your contributions, as long as such uses are compliant with the terms of the respective licenses", and set out how reusers attribute: by a link or URL to the article, by a link to a stable copy, or by a list of authors (Wikimedia Foundation, 2023).

The conditions of the licence attach to particular acts. It grants the right to reproduce and share the material, and to produce and share adapted material (section 2(a)(1)). The attribution duty applies when the material is shared (section 3(a)(1)), and share-alike applies "if You Share Adapted Material You produce" (section 3(b)) (Creative Commons, 2013). The text says nothing about training a model, and nothing about returning anything to the source.

The steward of the licences reads their reach narrowly. In its legal primer on AI training, Creative Commons writes that "AI training is often permitted by copyright", which "means that the CC license conditions have limited application to machine reuse", and that "using a more restrictive CC license in an effort to prevent AI training is not an effective approach" (Pearson, 2025). Its September 2026 guidance adds that the guidance on licences "was developed for a world of reuse by people, a premise that doesn't fit as neatly in a world of widespread machine use", that "licensing alone cannot address all of the challenges to sharing that AI presents", and that its CC Signals project is "a flexible, commons-friendly framework for communicating expectations around AI use of content and data" (Creative Commons, 2026). The expectations in question are those the licence does not carry.

Courts have not filled the gap. The nearest case concerns code under open-source licences, not Wikipedia text. In *Doe v. GitHub*, programmers alleged that Copilot reproduced their code without attribution, in violation of section 1202(b) of the Digital Millennium Copyright Act. The Ninth Circuit affirmed the dismissal of that claim on the "output" theory, holding that the tools do not "remove or alter" copyright management information from a copy of an existing work but create new works that never contained it. It declined to consider the "input" theory, that stripping the information at the training stage was itself the violation, because the plaintiffs had forfeited it. The court records that, asked whether copying training data into Copilot violated the attribution requirement of open-source licences, the plaintiffs' counsel answered "Perhaps it doesn't." Two breach-of-contract claims remain pending in the district court (*Doe v. GitHub, Inc.*, 2026). So the attribution route to enforcement failed for outputs, and the case does not decide whether training breaches an open licence's conditions. Lemley and Casey (2021) argue that training on copyrighted works should generally count as fair use, in which case the licence conditions are not reached at all. Nothing found in this review establishes that reuse of Wikipedia text by AI developers breaches its licence, and the essay does not assume that it does.

### 4.2 What the licence does not regulate

If the licence is silent on training, what might a commons reasonably ask of those who rely on it? Four things stand out: credit in outputs that draw on it, a route back for readers, a share of the cost of keeping it running, and openness about how it is used. The evidence that each matters is uneven.

*Attention and the route back.* The Wikimedia Foundation reports that human pageviews were down roughly 8% against the same months of 2024, and attributes the fall to "the impact of generative AI and social media on how people seek information, especially with search engines providing answers directly to searchers, often based on Wikipedia content" (Miller, 2025). It asks LLMs, chatbots, search engines and social platforms to "make it clear where the information is sourced from and elevate opportunities to visit and participate in those sources". A pull from the public pageviews interface for English Wikipedia (user traffic, January to September) shows 69.2 billion views in 2024, 64.6 billion in 2025 and 59.5 billion in 2026, falls of 6.7% and 7.9%. The academic evidence on engagement is mixed. Reeves et al. (2025) found no overall decline in views, visitors, edits or editors across twelve language editions through early 2024, though growth was lower where ChatGPT was available; Lyu et al. (2025) found larger declines in editing and viewership for new, popular articles whose content overlapped with ChatGPT's output. The aggregate decline is modest and its cause is partly contested.

*Cost.* Mueller et al. (2025) report that at least 65% of the most resource-consuming traffic to the Foundation's core data centres comes from bots, which account for about 35% of pageviews, and that bandwidth used to download multimedia has grown by 50% since January 2024, "largely from automated programs that scrape the Wikimedia Commons image catalog of openly licensed images to feed images to AI models". Not all bot traffic is linked to AI developers, and the post does not attribute the whole of it to them.

*Money.* Wikimedia Enterprise, the Foundation's paid-access service, had revenue of $8.3 million in fiscal 2024-25, up 148%, which was 4.0% of the Foundation's revenue, from 13 commercial customers (Wikimedia Foundation, 2025a). In January 2026 it named Amazon, Meta, Microsoft, Mistral AI and Perplexity among its partners, alongside Google and others, without giving terms (Wikimedia Enterprise, 2026). Paid access to structured data is a form of contribution, but its scale and terms are not public, and it is not a condition of the licence.

None of this shows a large harm, and the argument of the following sections does not rest on one. It shows where the licence is silent and where the community has begun to ask for something.

### 4.3 What the Foundation says it expects

The Foundation's own statement of what it wants from AI developers is explicit. It asks for attribution, which "means that generative AI gives credit to the human contributions that it uses to create its outputs"; for access through its paid service, since "most AI developers should properly access Wikipedia's content through the Wikimedia Enterprise platform"; and for the upkeep of "a virtuous cycle that continues those human contributions that create the training data that these new technologies rely on" (Wikimedia Foundation, 2025d). The statement is framed as a request and as a shared interest. It does not say that current reuse breaches the licence.

Both stewards, then, place these expectations outside the licence's terms: Creative Commons because the licence was written for reuse by people, and the Foundation because it asks for things the licence does not require. The gap this essay is concerned with is not a violation. It is a set of expectations that no instrument currently binds anyone to meet. Whether anything is owed in respect of them is a question about obligations and not about licences, and the next section takes it up.

## 5. What the unease tracks

The common reaction to AI developers' reuse of Wikipedia is that something has gone wrong with the people who built it, and the everyday word for that is betrayal. This section treats the word as a loose label and tests whether it fits. Where it does not, it asks what the unease is tracking.

### 5.1 Why "betrayal" fits few cases

Two accounts of trust give a test for betrayal, and they set the same bar from different sides. For Baier (1986), trust is "accepted vulnerability to another's possible but not expected ill will (or lack of good will) toward one" (p. 235). She separates it from mere dependence: neighbours who set their clocks by Kant's daily walk may be disappointed if he sleeps in, but they are not let down, still less betrayed. She adds that proper trust is the kind that survives the awareness of both parties and where the trusted has had an opportunity to signify acceptance or rejection and to warn the trusting if their trust is unacceptable. Holton (1994) locates the difference in a participant stance: to trust is to rely on someone while being ready to feel betrayal if the reliance fails, where a machine's failure produces annoyance but not resentment. Hawley (2014) builds on both and puts commitment at the centre. To trust someone to do something is to believe that they have a commitment to do it and to rely on them to meet it. Commitments may be implicit or explicit, and she takes it that "mutual expectation and convention give rise to commitment unless we take steps to disown these"; they can also be acquired by allowing others to continue to rely on us. On her account the wrong of betrayal lies in a failure to fulfil a commitment, whether or not anyone trusted. And expecting what nobody has committed to produces a feeling and not a fact: "we may feel betrayed, but we have not been betrayed". She adds that when a commitment is made to a third party, the person betrayed is the one to whom it was made.

Applied to reuse of Wikipedia, the test asks whether AI developers hold a commitment, and to whom. Four candidates can be separated.

- *The licence.* Its attribution and share-alike duties attach when material is shared (Creative Commons, 2013). A reuser who shares without attributing breaches a commitment, but that is an ordinary breach and not what the unease is about.
- *Explicit public assurances.* Stack Overflow's announcement of its partnership with OpenAI says that OpenAI will "provide attribution to the Stack Overflow community within ChatGPT" (Stack Overflow, 2024). That is a commitment, made within a deal with the platform. If unmet, the party betrayed is the one to whom it was made.
- *The Foundation's requests.* The Foundation asks AI developers for attribution, paid access and a "virtuous cycle" (Wikimedia Foundation, 2025d). A request creates an expectation on one side. It does not create a commitment on the other.
- *The convention of open knowledge.* Mutual expectation can give rise to commitment unless disowned, in Hawley's terms, and one might say that sharing under an open licence invites a convention of acknowledgement. Whether editors hold such an expectation, and whether developers allowed them to rely on it, is not studied in the literature reviewed here. Baier's condition, an opportunity to signify acceptance or rejection, cannot be assessed either.

Where commitments exist they run to platforms or to the Foundation and not to individual editors or readers. So the word "betrayal" fits particular cases, those with an explicit commitment that was not kept, and fits reuse in general poorly.

### 5.2 The hard case: a gift to anyone

The strongest objection is that nothing is owed because the contributors gave freely and to everyone. The licence they accepted is "worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable" and the Foundation's terms say that it allows commercial uses (Creative Commons, 2013; Wikimedia Foundation, 2023). The Open Source Definition for software, which refuses restrictions on fields of endeavour and requires no royalty or fee (Open Source Initiative, 2007), expresses the same culture, though CC BY-SA is not a software licence and the parallel is an analogy. Mauss's (2024) work on the gift, in its standard reading, treats gifts as creating obligations to give, receive and reciprocate, but a deliberate gift to anyone, with no named recipient and no named return, is the case such an account handles worst. If contributors waived any claim to a return, non-reciprocation does no wrong to them.

The objection is correct about one thing. It rules out a claim by individual editors to payment or credit as a debt for past gifts. It does not show that nothing is owed to the thing that was given. It shifts the question from what the reusers owe the donors to what they owe the practice.

### 5.3 Three better descriptions, and how far each goes

*Free riding.* Hart (1955) and Rawls (1964), in their standard reading, ground a duty of fair play: those who accept the benefits of a mutually advantageous scheme in which others bear restrictions owe their share. Wikipedia's contributors do accept restrictions, such as sourcing rules, notability standards and deletion discussions. AI developers do not accept them and did not join the scheme, and fair play usually presupposes participation. The description gives partial support at best.

*Exploitation.* "To exploit someone is to take unfair advantage of them", and exploitation can be "mutually beneficial, where both parties walk away better off", yet unfair (Zwolinski et al., 2022). Contributors consented and, in the sense that the encyclopedia exists, gained. Whether the division of benefits is unfair depends on a theory of fair division that the literature reviewed here does not supply, and exploitation is usually tied to the use of a vulnerability that volunteer contributors do not obviously have. The literature on data as labour recasts contributions as work that deserves pay (Arrieta-Ibarra et al., 2018), and work on data leverage treats withholding contributions as collective bargaining power (Vincent et al., 2021). Both fit commercial platforms better than a volunteer commons whose point is that the work is unpaid.

*Dependence.* The strongest description starts from reliance. Four premises carry it.

1. The practice is a shared epistemic infrastructure on which others rely. This is the claim of Section 3: knowledge held by a community that checks, contests and answers for it (Hardwig, 1985; Menking & Rosenberg, 2021).
2. Reusers rely on it. The Foundation says that almost all large language models train on its datasets (Miller, 2025). Its share of training text is small in volume, about 0.14% of the tokens in one open corpus and 4.5% of the sampling weight in another (Soldaini et al., 2024; Touvron et al., 2023), so the reliance is on curation and quality and not on bulk. Shumailov et al. (2024) show that training recursively on model output degrades models, and note that human-generated data gain value as synthetic text spreads; the dependence is contested, since collapse can be avoided when real data are kept and synthetic data accumulate (Gerstgrasser et al., 2024), and forecasts of data scarcity allow escape routes (Villalobos et al., 2024).
3. Reuse imposes costs or erodes the conditions of upkeep. The costs of automated traffic and the fall in human readership are documented, though modest and partly contested (Section 4).
4. Reusers can help at small cost relative to their means. Crediting a source in an output, sending readers back, and paying for structured access are inexpensive for the largest firms, and some already pay (Wikimedia Foundation, 2025a).

If these hold, a duty follows that is proportioned to reliance and to capacity: those who depend on a shared epistemic infrastructure, and whose reuse strains it, owe something toward keeping it going. This is the essay's principle, not one found in this form in the literature reviewed. Its nearest relatives are the proposal that developers should "contribute back" to the commons they draw on (Huang & Siddarth, 2023) and the commons tradition's point that shared resources survive through community rules and not through licences alone (Hess & Ostrom, 2007). Because the duty runs to the practice and to those who steward it, and not to donors as repayment for past gifts, it is compatible with the gift objection.

### 5.4 What the unease tracks

The charge of betrayal is apt for particular broken commitments, and it is an unreliable guide for reuse in general. What it tracks across the cases is reliance without upkeep. Developers draw on a practice that others sustain, strain some of the conditions of its upkeep, and are able to help at modest cost, without being bound by the licence, or by a commitment, or by any other instrument to do so.

Three difficulties remain and are stated here rather than resolved. The size of the harm is modest and partly contested, so the duty is correspondingly light. Reliance varies among thousands of reusers, so proportioning the duty raises a collective-action problem about who should do what. And an obligation of this kind needs forms through which it can be met, which is the question of the next section.

## 6. Forms of stewardship

Section 5 argued that those who rely on a shared epistemic infrastructure, and whose reuse strains it, owe something toward keeping it going, in proportion to their reliance and their means. A duty needs forms through which it can be met. This section asks what the literature and the stewards' own statements offer, and judges each form against criteria that follow from the argument.

### 6.1 What a form of stewardship must do

Four criteria follow from Section 5.

1. *It runs to the practice, not to donors.* The duty is owed to the verification practice and to those who carry it, and not as repayment to individual contributors for past gifts.
2. *It is proportionate.* It scales with reliance and capacity, so a form that costs the same for a large firm and a small one fails.
3. *It can be checked.* If the value of the practice lies in auditability, a form of support that cannot itself be audited, whose terms are hidden or whose delivery cannot be observed, sits badly with what it supports.
4. *It does not damage what it protects.* A form that crowds out volunteers, closes the commons, or hands payers influence over content would defeat its purpose.

The second and fourth criteria matter because of two facts about the commons. The practice is carried by people, and its binding constraint is who can enter it: Halfaker et al. (2013) trace the decline in active editors to the way verification machinery treats newcomers. And the money and the labour go to different places. The Wikimedia Foundation owns and runs the servers and software and controls the budget for community programmes, while editors are volunteers (Menking & Rosenberg, 2021). A form of support that funds the Foundation does not by itself reach the people who carry the practice.

### 6.2 Candidate forms

*Credit and a route back.* The Foundation asks that generative AI "gives credit to the human contributions that it uses to create its outputs" (Wikimedia Foundation, 2025d), and that platforms "make it clear where the information is sourced from and elevate opportunities to visit and participate in those sources" (Miller, 2025). Its draft plan for 2026-27 proposes "a sustainable reuse model that reaches consumers where they are, and sends value and future contributors back to Wikipedia — not just traffic away" (Wikimedia Foundation, 2026b). This form is cheap for a reuser, runs to the practice, and, if readers do follow the link, addresses the recruitment problem as well as the traffic problem. Its weaknesses are practical. Attribution cannot currently be enforced by law against model outputs on the route tested in *Doe v. GitHub* (Section 4), and whether a firm's link actually leads readers into the practice is an empirical matter that this review did not find studied.

*Paid structured access.* Wikimedia Enterprise sells access to Wikipedia's content at volumes and speeds that public interfaces do not serve. The Foundation presents it as the way for commercial reusers "to help carry the burden of sustaining those commons", states that income from it "is capped at 30% of the total Wikimedia Foundation annual revenue" so that most funding keeps coming from small donations, and has begun rate-limiting large-scale reusers of its public interfaces, for whom Enterprise is "a must-do rather than a nice-to-have" (Becker, 2026). Revenue was $8.3 million in fiscal 2024-25, from 13 commercial customers (Wikimedia Foundation, 2025a), and the terms of the partnerships announced in 2026 are not public (Wikimedia Enterprise, 2026). The form is proportionate in principle, since heavy users pay for heavy use, and the cap speaks to the fourth criterion. It does less well on the third and on reaching volunteers: undisclosed terms cannot be audited from outside, and the revenue goes to the Foundation.

*Preference signals and consent.* Creative Commons describes CC Signals as "a flexible, commons-friendly framework for communicating expectations around AI use of content or data", meant to build "new norms and governance so shared knowledge is used in ways that embeds reciprocity", and the page does not say they are legally binding (Creative Commons, n.d.). Longpre et al. (2024) show what happens without such a channel: across about 14,000 web domains they found rapidly growing restrictions on crawling and in terms of service in 2023-24, with about 5% of tokens in one common corpus fully restricted and about 45% restricted by terms of service. The commons is being closed by data holders who have no other way of stating their terms. Signals offer a way to state conditions without closing, but a statement of expectations binds nobody, and the criteria of proportion and checkability depend on what reusers do with them.

*Contribution back.* Huang and Siddarth (2023) argue that foundation-model developers should be expected to contribute quality data back to the commons they draw on, with transparency, funded auditing and shared ownership of fine-tuning data. For a verification practice the relevant contribution is less data than help with the work that keeps the practice reliable. The community has had to organise against machine-generated text, through a project for cleaning it up from December 2023, a narrow speedy-deletion criterion, and in March 2026 a prohibition on LLM-generated or rewritten article content adopted by 44 votes to 2 (Bansal, 2026; Froneman, 2026). That work falls on volunteers. The machine-generated text they police is introduced by people using language-model tools, not by developers' reuse of Wikipedia content, so this cost is adjacent to the argument and not part of it; whether the makers of the tools owe anything toward it is a further question that this essay does not take up.

### 6.3 Assessing the forms

No single form meets all four criteria. Credit and a route back scores well on the first, second and fourth and is hard to enforce. Paid access scores well on proportion and independence, and weakly on checkability and on reaching volunteers. Signals preserve openness and say what the commons expects, without force. Contribution back could reach the people who carry the practice, but it has no established form.

Two points follow from the pattern. First, the forms work better together than apart: a route back recruits editors, paid access funds infrastructure, and signals say what is expected of those who do neither. Second, the weakest point in most of them is the same one, a gap between those who receive support and those who carry the practice. If the bastion is a community, support should be judged partly by whether it reaches the community: whether it widens participation, shortens the path for newcomers, or reduces the load of policing, and not only by whether it keeps servers running.

The collective-action problem of Section 5.4 points the same way. Reliance varies among thousands of reusers, and a bilateral arrangement with each is costly and opaque. A pooled arrangement run by the commons' own governance, with published terms and a published account of where money and effort go, would sit best with the criteria. The Foundation's Enterprise service moves in that direction without yet meeting the standard of publication.

### 6.4 What can reasonably be asked of developers

Four requests follow, offered as proposals of this essay and not as established norms.

1. Credit and a route back in outputs that draw on Wikipedia, at a cost the firm can bear.
2. Acceptance of the terms the commons states for heavy use, including paid access instead of scraping.
3. Disclosure of how the content is used, so that the arrangement can be audited.
4. Support that reaches participation, not only infrastructure.

Each is proportioned to reliance and capacity, none is a legal duty on present law, and each could be met by a firm acting alone or through a shared mechanism. The measure of success is whether the practice that makes Wikipedia worth relying on is kept up by those who rely on it. The next section turns from what developers should do to what Wikipedia itself should change.

## 7. What Wikipedia should change

If the bastion is a community that practises verification, then what Wikipedia should change concerns the practice and its people more than its pages. The changes below are questions the community can examine, grounded in the literature, and they are offered as implications of the argument and not as prescriptions. Only the community and its stewards can decide them.

### 7.1 Say what it offers

Wikipedia's policy already promises provenance and not truth (Wikipedia contributors, n.d.), but the label "source of truth" is used of it widely and is not corrected. The Foundation can describe its offer as auditability: a claim points to a published source, anyone can contest it through a recorded process, and a community with rules answers for what stands (Section 3). That description is accurate, and it is the property that generative answers lack by default.

Describing it is not enough if the audit trail stays at the back end. Menking and Rosenberg (2021) argue that the focus should be on the process of aggregation, and that "there needs to be a deeper connection and more transparency between the design of the back end and the design of the front end" (p. 468). Ford and Wajcman (2017) note that when facts are shown in search engines' fact boxes they fade into the background and are black-boxed. A request that reusers carry credit and a route back (Section 6) is partly a request that the audit trail travel with the content.

### 7.2 Treat newcomers and inclusion as the central defence

The practice has about 273,000 editors, 0.02% of its monthly readers, on the Foundation's figure (Wikimedia Foundation, 2026b). Halfaker et al. (2013) show that the number of active editors fell from its 2007 peak because the machinery that protects quality, including tools that reject contributions, treats good-faith newcomers badly, and that the mechanisms for revising norms calcified. Menking and Rosenberg (2021) report that the five pillars have been little changed since the project began and that editors who challenged them have been punished. If the bastion is the people who carry the practice, three questions follow for the community.

- Are the tools and rules for rejecting contributions calibrated so that good-faith newcomers can stay (Halfaker et al., 2013)?
- Do notability and deletion processes treat comparable subjects alike? Tripodi (2023) finds women's biographies over-represented among those nominated for deletion, and reads the higher rate at which nominated biographies about women are kept as evidence of mistaken nomination, an inference that remains open to challenge.
- Do the rules on reliable sources leave room for knowledge that is held orally (Gallert & van der Velden, 2014; Menking & Rosenberg, 2021)?

Diversity is not an ornament here. Shi et al. (2019) find that politically diverse editorial teams write higher-quality articles, which suggests that the narrowness of the editor base weakens the practice itself. For smaller language editions the practice is thinner and the stakes are higher: the Vietnamese edition had the lowest share of culturally specific content among 40 editions compared in 2018 (Miquel-Ribé & Laniado, 2018). The evidence on that edition is old and should be refreshed before it is relied on.

### 7.3 Be open about how it is reused

The stewardship argument asks reusers for transparency. It asks the same of the steward. Under present arrangements the Foundation controls the budget for community programmes (Menking & Rosenberg, 2021) and has announced partners for its paid service without giving terms (Wikimedia Enterprise, 2026). Publishing the terms of reuse agreements, and an account of where the money and effort go, would let the community and outside observers audit the arrangement that is supposed to sustain them. The Foundation's statement that Enterprise income is capped at 30% of its annual revenue and that its content will "forever be free and open to anyone" (Becker, 2026) names the commitments that such an account should let people check.

### 7.4 Two cautions

First, any attempt to make Wikipedia a "source of truth" in a strong sense would conflict with its own policy and would invite the failure the policy avoids. The stronger and more defensible claim is auditability.

Second, the changes proposed here are internal and slow, and the pressures from outside are not. The community has already acted against machine-generated text, with a project for cleaning it up, a narrow deletion criterion, and in March 2026 a vote of 44 to 2 to prohibit LLM-generated or rewritten article content (Bansal, 2026; Froneman, 2026). That response spends volunteer effort on a cost that language-model tools create. The argument of Section 5 does not make developers answerable for it, but it is a reason to ask what the makers of the tools should contribute to the people who bear it.

## 8. Objections and replies

### 8.1 A gift to anyone cannot be wronged

Contributors accepted a licence that is "worldwide, royalty-free ... irrevocable" and permits commercial use, so nothing is owed (Creative Commons, 2013; Wikimedia Foundation, 2023). *Reply.* The objection defeats a claim by individual editors to payment or credit as a debt for past gifts, and the essay makes no such claim (Section 5.2). It leaves open what is owed to the practice that the gift sustains, on which the duty rests (Section 5.3).

### 8.2 AI answers may serve readers better, so who is wronged?

If readers get what they need from a chatbot, the fall in visits may be a gain. *Reply.* The essay does not claim that readers are harmed. Its claim concerns the upkeep of a practice that others rely on. It is also conditional: if developers cease to depend on the practice, the dependence premise weakens, and the evidence on that dependence is contested (Shumailov et al., 2024; Gerstgrasser et al., 2024; Villalobos et al., 2024). What readers lose, if anything, is a visible route to contest a claim, which an answer generated by default does not provide. Even this should be stated cautiously, since some systems now cite their sources.

### 8.3 Wikipedia is financially healthy

Its fiscal 2024-25 revenue was $208.6 million, 11% above target (Wikimedia Foundation, 2025c). *Reply.* The argument does not rest on need. It rests on reliance and strain. Financial health does not measure the health of the practice, whose binding constraint is its people (Halfaker et al., 2013). The point does cut against any claim that the Foundation is at risk, and the essay makes none; it makes the duty correspondingly proportionate and light.

### 8.4 "Last bastion" is romantic

The phrase overstates what Wikipedia holds and for whom. *Reply.* Section 3.4 grants this. What is defended is published, verifiable knowledge as judged by a particular community, and the defence includes the property that makes its gaps visible and repairable.

### 8.5 The principle proves too much

If reliance and strain ground a duty, then search engines, teachers and ordinary readers owe something too. *Reply.* They do, in proportion. The principle is not special to AI developers. McMahon et al. (2017) describe the interdependence of Wikipedia and a search engine that predates generative AI, in which the engine's direct display of content reduced visits to Wikipedia while Wikipedia improved the engine's results. What is distinctive about the present case is scale and mode of reuse. That the principle generalises is a feature of it.

### 8.6 Machine-generated text is entering Wikipedia

If the practice is being flooded, the picture of a robust bastion fails. *Reply.* The evidence on how much is entering is uneven: detectors flagged over 5% of newly created English articles in one study, given as a lower bound (Brooks et al., 2024), and another estimates LLM-related change of about 1% in some categories (Huang et al., 2025). A working study of new English articles in the repository that holds this essay finds a flagged share only slightly above the rate expected from human text, and a rise in LLM-style wording in 2024 and after; that study is not peer reviewed and depends on its detector. If contamination grows, auditability matters more, since provenance and contestability are the tools for dealing with it, and the cost to volunteers rises with it.

### 8.7 Who is the bearer of the duty?

Reliance varies among thousands of reusers, and a duty with no assignable bearer is easily ignored. *Reply.* This is a real difficulty, stated in Section 5.4. It favours pooled arrangements with published terms over bilateral deals (Section 6.3), and it is a reason to proportion the duty and not to drop it.

## 9. Conclusion

The question was what AI developers owe the community behind Wikipedia when their reuse complies with its licence, and what the answer shows about why Wikipedia matters. The answer given here has three parts.

Wikipedia matters as an auditable practice of verification carried by a community, not as a source of truth. Its standing has risen and fallen with the alternatives against which it is compared, and what is worth defending is the practice and the people who keep it going, with its known gaps in view.

The licence is silent on training and on return, and its stewards say so. The charge of betrayal is apt only for particular broken commitments. For reuse in general the unease tracks reliance without upkeep. That grounds a duty of stewardship that is proportioned to reliance and capacity, and runs to the practice and not to donors. The duty is light, because the evidence of harm is modest and partly contested, and conditional, because the dependence on which it rests is contested too.

The duty can be met in several forms, none complete: credit and a route back, paid access, preference signals and contributions that reach the people. Wikipedia has its own part to play, in how it treats newcomers, what it counts as knowledge, and how openly it deals with those who reuse it.

Three limits should be kept in view. The essay is conceptual and relies on public documents and a literature that is thin on its central questions. No study asks whether editors hold an expectation of return from AI developers, how readers fare when they are sent to a model and not to the source, or how far the external sources that Wikipedia cites are being contaminated by machine-generated text. Each would sharpen or weaken the argument. The principle at its centre, that reliance and strain ground a proportionate duty of upkeep, is this essay's own and is open to the objections of Section 8. What the argument gives, if it holds, is a way of saying why the unease is justified without claiming that a promise was broken.

## References

Arrieta-Ibarra, I., Goff, L., Jiménez-Hernández, D., Lanier, J., & Weyl, E. G. (2018). Should we treat data as labor? Moving beyond "free". *AEA Papers and Proceedings, 108*, 38-42. https://doi.org/10.1257/pandp.20181003

Baier, A. (1986). Trust and antitrust. *Ethics, 96*(2), 231-260. https://doi.org/10.1086/292745

Bansal, A. (2026, March 26). Wikipedia bans AI-generated article content. *MediaNama*. https://www.medianama.com/2026/03/223-english-wikipedia-bans-ai-generated-text-allows-limited-use-copyediting-translation/

Becker, L. (2026, July 16). The cost of "free": How Wikimedia Enterprise protects Wikipedia in the AI era. Wikimedia Foundation. https://wikimediafoundation.org/news/2026/07/16/wikimedia-enterprise-protecting-wikipedia-ai/

Brooks, C., Eggert, S., & Peskoff, D. (2024). The rise of AI-generated content in Wikipedia. In *Proceedings of the First Workshop on Advancing NLP for Wikipedia* (pp. 67-79). https://aclanthology.org/2024.wikinlp-1.12/

Burtch, G., Lee, D., & Chen, Z. (2024). The consequences of generative AI for online knowledge communities. *Scientific Reports, 14*, Article 10413. https://doi.org/10.1038/s41598-024-61221-0

Camerer, C. F., Dreber, A., Holzmeister, F., Ho, T.-H., Huber, J., Johannesson, M., ... Wu, H. (2018). Evaluating the replicability of social science experiments in Nature and Science between 2010 and 2015. *Nature Human Behaviour, 2*, 637-644. https://doi.org/10.1038/s41562-018-0399-z

Creative Commons. (n.d.). *CC Signals*. https://creativecommons.org/ai-and-the-commons/cc-signals/

Creative Commons. (2013). *Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) legal code*. https://creativecommons.org/licenses/by-sa/4.0/legalcode.en

Creative Commons. (2026, September 3). *Guidance on using CC licenses in an AI ecosystem*. https://creativecommons.org/2026/09/03/guidance-on-using-cc-licenses-in-an-ai-ecosystem/

del Rio-Chanona, M., Laurentsyeva, N., & Wachs, J. (2024). Large language models reduce public knowledge sharing on online Q&A platforms. *PNAS Nexus, 3*(9). https://doi.org/10.1093/pnasnexus/pgae400

*Doe v. GitHub, Inc.*, No. 24-7700 (9th Cir. Sept. 16, 2026). https://www.eff.org/files/2026/09/16/doe_v_github.pdf

Fallis, D. (2008). Toward an epistemology of Wikipedia. *Journal of the American Society for Information Science and Technology, 59*(10), 1662-1674. https://doi.org/10.1002/asi.20870

Ford, H., & Wajcman, J. (2017). 'Anyone can edit', not everyone does: Wikipedia's infrastructure and the gender gap. *Social Studies of Science, 47*(4), 511-527. https://doi.org/10.1177/0306312717692172

Froneman, W. (2026). Failed comprehensiveness, successful minimalism: Wikipedia's 3-year struggle to govern AI-generated content (2022-2025). *AI & Society*. https://doi.org/10.1007/s00146-026-03046-1

Gallert, P., & van der Velden, M. (2014). Reliable sources for indigenous knowledge: Dissecting Wikipedia's catch-22. In N. J. Bidwell & H. Winschiers-Theophilus (Eds.), *Indigenous Knowledge Technology Conference (IKTC) 2011 post-conference book*. (Unedited preliminary version; final publication details unconfirmed.) https://upload.wikimedia.org/wikipedia/commons/5/51/Indigenous_Knowledge_for_Wikipedia.pdf

Garfinkel, S. L. (2008, October 20). Wikipedia and the meaning of truth. *MIT Technology Review*. https://www.technologyreview.com/2008/10/20/218162/wikipedia-and-the-meaning-of-truth/

Gerlich, M. (2025). *AI tools in society: Impacts on cognitive offloading and the future of critical thinking* [Preprint]. SSRN. https://doi.org/10.2139/ssrn.5082524

Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., Sleight, H., Hughes, J., ... Koyejo, S. (2024). *Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data* [Preprint]. arXiv:2404.01413. https://arxiv.org/abs/2404.01413

Giles, J. (2005). Internet encyclopaedias go head to head. *Nature, 438*(7070), 900-901. https://doi.org/10.1038/438900a

Goldman, A. I. (2001). Experts: Which ones should you trust? *Philosophy and Phenomenological Research, 63*(1), 85-110. https://doi.org/10.1111/j.1933-1592.2001.tb00093.x

Graham, M., Hogan, B., Straumann, R. K., & Medhat, A. (2014). Uneven geographies of user-generated information: Patterns of increasing informational poverty. *Annals of the Association of American Geographers, 104*(4), 746-764. https://doi.org/10.1080/00045608.2014.910087

Halfaker, A., Geiger, R. S., Morgan, J. T., & Riedl, J. (2013). The rise and decline of an open collaboration system: How Wikipedia's reaction to popularity is causing its decline. *American Behavioral Scientist, 57*(5), 664-688. https://doi.org/10.1177/0002764212469365

Hardwig, J. (1985). Epistemic dependence. *The Journal of Philosophy, 82*(7), 335-349. https://doi.org/10.2307/2026523

Hart, H. L. A. (1955). Are there any natural rights? *The Philosophical Review, 64*(2), 175-191. https://doi.org/10.2307/2182586

Hawley, K. (2014). Trust, distrust and commitment. *Noûs, 48*(1), 1-20. https://doi.org/10.1111/nous.12000

Head, A. J., & Eisenberg, M. B. (2010). How today's college students use Wikipedia for course-related research. *First Monday, 15*(3). https://doi.org/10.5210/fm.v15i3.2830

Hess, C., & Ostrom, E. (Eds.). (2007). *Understanding knowledge as a commons: From theory to practice*. MIT Press.

Hill, B. M., & Shaw, A. (2013). The Wikipedia gender gap revisited: Characterizing survey response bias with propensity score estimation. *PLOS ONE, 8*(6), e65782. https://doi.org/10.1371/journal.pone.0065782

Holton, R. (1994). Deciding to trust, coming to believe. *Australasian Journal of Philosophy, 72*(1), 63-76. https://doi.org/10.1080/00048409412345881

Huang, S., & Siddarth, D. (2023). *Generative AI and the digital commons* [Preprint]. arXiv:2303.11074. https://arxiv.org/abs/2303.11074

Huang, S., Xu, Y., Geng, M., Wan, Y., & Chen, D. (2025). *Wikipedia in the era of LLMs: Evolution and risks* [Preprint]. arXiv:2503.02879. https://arxiv.org/abs/2503.02879

Jaschik, S. (2007, January 26). A stand against Wikipedia. *Inside Higher Ed*. https://www.insidehighered.com/news/2007/01/26/stand-against-wikipedia

Kosmyna, N., Hauptmann, E., Yuan, Y. T., Situ, J., Liao, X.-H., Beresnitzky, A. V., Braunstein, I., & Maes, P. (2025). *Your brain on ChatGPT: Accumulation of cognitive debt when using an AI assistant for essay writing task* [Preprint]. arXiv:2506.08872. https://arxiv.org/abs/2506.08872

Kräenbring, J., Monzon Penza, T., Gutmann, J., Muehlich, S., Zolk, O., Wojnowski, L., Maas, R., Engelhardt, S., & Sarikas, A. (2014). Accuracy and completeness of drug information in Wikipedia: A comparison with standard textbooks of pharmacology. *PLoS ONE, 9*(9), e106930. https://doi.org/10.1371/journal.pone.0106930

Lemley, M. A., & Casey, B. (2021). Fair learning. *Texas Law Review, 99*, 743. https://doi.org/10.2139/ssrn.3528447

Longpre, S., Mahari, R., Lee, A., Lund, C., et al. (2024). *Consent in crisis: The rapid decline of the AI data commons* [Preprint]. arXiv:2407.14933. https://arxiv.org/abs/2407.14933

Lyu, L., Siderius, J., Li, H., Acemoglu, D., Huttenlocher, D., & Ozdaglar, A. (2025). Wikipedia contributions in the wake of ChatGPT. In *Companion Proceedings of the ACM Web Conference 2025*. https://doi.org/10.1145/3701716.3715543

Magnus, P. D. (2009). On trusting Wikipedia. *Episteme, 6*(1), 74-90. https://doi.org/10.3366/E1742360008000555

Mauss, M. (2024). *The gift: The form and reason for exchange in archaic societies*. Routledge. https://doi.org/10.4324/9781003572350 (Original work published 1925.)

McMahon, C., Johnson, I., & Hecht, B. (2017). The substantial interdependence of Wikipedia and Google: A case study on the relationship between peer production communities and information technologies. *Proceedings of the International AAAI Conference on Web and Social Media, 11*(1), 142-151. https://doi.org/10.1609/icwsm.v11i1.14883

Menking, A., & Rosenberg, J. (2021). WP:NOT, WP:NPOV, and other stories Wikipedia tells us: A feminist critique of Wikipedia's epistemology. *Science, Technology, & Human Values, 46*(3), 455-479. https://doi.org/10.1177/0162243920924783

Miller, M. (2025, October 17). New user trends on Wikipedia. *Diff*, Wikimedia Foundation. https://diff.wikimedia.org/2025/10/17/new-user-trends-on-wikipedia/

Miquel-Ribé, M., & Laniado, D. (2018). Wikipedia culture gap: Quantifying content imbalances across 40 language editions. *Frontiers in Physics, 6*, Article 54. https://doi.org/10.3389/fphy.2018.00054

Mueller, B., Danis, C., & Lavagetto, G. (2025, April 1). How crawlers impact the operations of the Wikimedia projects. *Diff*, Wikimedia Foundation. https://diff.wikimedia.org/2025/04/01/how-crawlers-impact-the-operations-of-the-wikimedia-projects/

Open Source Initiative. (2007). *The open source definition* (v1.9). https://opensource.org/osd

Orben, A. (2020). The Sisyphean cycle of technology panics. *Perspectives on Psychological Science, 15*(5), 1143-1157. https://doi.org/10.1177/1745691620919372

Pearson, S. H. (2025, May 15). *Understanding CC licenses and AI training: A legal primer*. Creative Commons. https://creativecommons.org/2025/05/15/understanding-cc-licenses-and-ai-training-a-legal-primer/

Punturo, B. (2026, February 3). How ChatGPT sources the web. Profound. https://www.tryprofound.com/blog/chatgpt-citation-sources

Rawls, J. (1964). Legal obligation and the duty of fair play. In S. Hook (Ed.), *Law and philosophy* (pp. 3-18). New York University Press.

Reddit, Inc. (2026a, February 5). *Reddit reports fourth quarter and full year 2025 results* (Form 8-K, Exhibit 99.1). U.S. Securities and Exchange Commission. https://www.sec.gov/Archives/edgar/data/1713445/000171344526000020/earningspressreleaseq425.htm

Reddit, Inc. (2026b). *Annual report (Form 10-K) for the fiscal year ended December 31, 2025*. U.S. Securities and Exchange Commission. https://www.sec.gov/Archives/edgar/data/1713445/000171344526000022/rddt-20251231.htm

Reeves, N., Yin, W., & Simperl, E. (2024). *Exploring the impact of ChatGPT on Wikipedia engagement* [Preprint]. arXiv:2405.10205. https://arxiv.org/abs/2405.10205

Reeves, N., Yin, W., & Simperl, E. (2025). Exploring the impact of ChatGPT on Wikipedia engagement. *Collective Intelligence, 4*(3). https://doi.org/10.1177/26339137251372599

Sanger, L. M. (2009). The fate of expertise after Wikipedia. *Episteme, 6*(1), 52-73. https://doi.org/10.3366/E1742360008000543

Shi, F., Teplitskiy, M., Duede, E., & Evans, J. A. (2019). The wisdom of polarized crowds. *Nature Human Behaviour, 3*(4), 329-336. https://doi.org/10.1038/s41562-019-0541-6

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature, 631*(8022), 755-759. https://doi.org/10.1038/s41586-024-07566-y

Soldaini, L., et al. (2024). *Dolma: An open corpus of three trillion tokens for language model pretraining research* [Preprint]. arXiv:2402.00159. https://arxiv.org/abs/2402.00159

Sparrow, B., Liu, J., & Wegner, D. M. (2011). Google effects on memory: Cognitive consequences of having information at our fingertips. *Science, 333*(6043), 776-778. https://doi.org/10.1126/science.1207745

Stack Overflow. (2024, May 6). *Stack Overflow and OpenAI partnership*. https://stackoverflow.co/company/press/archive/openai-partnership

Tollefsen, D. P. (2009). Wikipedia and the epistemology of testimony. *Episteme, 6*(1), 8-24. https://doi.org/10.3366/E1742360008000518

Touvron, H., et al. (2023). *LLaMA: Open and efficient foundation language models* [Preprint]. arXiv:2302.13971. https://arxiv.org/abs/2302.13971

Tripodi, F. (2023). Ms. Categorized: Gender, notability, and inequality on Wikipedia. *New Media & Society, 25*(7), 1687-1707. https://doi.org/10.1177/14614448211023772

Villalobos, P., Ho, A., Sevilla, J., Besiroglu, T., Heim, L., & Hobbhahn, M. (2024). *Will we run out of data? Limits of LLM scaling based on human-generated data* [Preprint]. arXiv:2211.04325. https://arxiv.org/abs/2211.04325

Vincent, N., Li, H.-L., Tilly, N., Chancellor, S., & Hecht, B. (2021). Data leverage: A framework for empowering the public in its relationship with technology companies. In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency* (pp. 215-227). https://doi.org/10.1145/3442188.3445885

Waters, N. L. (2007). Why you can't cite Wikipedia in my class. *Communications of the ACM, 50*(9), 15-17. https://doi.org/10.1145/1284621.1284635

Wikimedia Enterprise. (2026, January 15). Announcing new Wikimedia Enterprise partners for Wikipedia's 25th birthday. https://enterprise.wikimedia.com/blog/wikipedia-25-enterprise-partners/

Wikimedia Foundation. (2023). *Terms of use* (effective 7 June 2023). https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use

Wikimedia Foundation. (2025a, November 24). Wikimedia Enterprise financial report: Fiscal year 2024-2025. *Diff*. https://diff.wikimedia.org/2025/11/24/wikimedia-enterprise-financial-report-fiscal-year-2024-2025/

Wikimedia Foundation. (2025c, November 24). Highlights from the Wikimedia Foundation's fiscal year 2024-2025 audit report. *Diff*. https://diff.wikimedia.org/2025/11/24/highlights-from-the-wikimedia-foundations-fiscal-year-2024-2025-audit-report/

Wikimedia Foundation. (2025d, November 10). *In the AI era, Wikipedia has never been more valuable*. https://wikimediafoundation.org/news/2025/11/10/in-the-ai-era-wikipedia-has-never-been-more-valuable/

Wikimedia Foundation. (2026a, January 15). *Wikipedia celebrates 25 years of knowledge at its best*. https://wikimediafoundation.org/news/2026/01/15/wikipedia-celebrates-25years/

Wikimedia Foundation. (2026b, April 27). Introducing the Wikimedia Foundation's draft FY 2026-2027 annual plan. *Diff*. https://diff.wikimedia.org/2026/04/27/introducing-the-wikimedia-foundations-draft-fy-2026-2027-annual-plan/

Wikipedia contributors. (n.d.). *Wikipedia:Verifiability*. https://en.wikipedia.org/wiki/Wikipedia:Verifiability (retrieved 3 October 2026)

Zwolinski, M., Ferguson, B., & Wertheimer, A. (2022). Exploitation. In E. N. Zalta & U. Nodelman (Eds.), *The Stanford encyclopedia of philosophy* (Winter 2022 ed.). https://plato.stanford.edu/entries/exploitation/
