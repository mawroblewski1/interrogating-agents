# ECS 172 Final Project — Full Briefing for Paper Editors and GitHub Contributors

**Course:** ECS 172 Recommender Systems, Spring 2026, UC Davis  
**Team:** Bill Koumba, Sanjay Manivasagam, Marcin Wróblewski, Haoyu Yan  
**GitHub:** https://github.com/mawroblewski1/interrogating-agents  
**Working branch:** `william-dev`  
**Paper file:** `paper.docx` (shared separately — gitignored, do not commit)

---

## 1. What This Project Is

We built a **multi-agent conversational system** in which a RAG-powered Interrogator tries to shift the stated position of a simulated LLM Suspect on real ballot-measure topics. A Judge scores every Suspect utterance on a five-point stance scale. The central research question: does recommending real-world interrogation and debate techniques at each conversational turn produce measurably greater stance change than an unguided baseline?

**Short answer from results:** Yes, on directional accuracy. Treatment condition achieves **75% directional accuracy vs. 35% for control** (pooled across 80 legs, 4 topics). The strongest individual result is the Transit topic (control 10% → treatment 90%).

**Framing note:** The paper deliberately uses non-partisan civic ballot topics (housing, arts, transit, Davis Measure V) to keep the work politically neutral. The long-term research vision — which may be alluded to but not dwelt on — is AI-facilitated constructive dialogue and healthy online discourse, not coercive interrogation.

---

## 2. System Architecture (5 Milestones, All Complete)

### Roles (three LLM agents, all `llama3.1:8b` via local Ollama)

| Role | File | Purpose |
|---|---|---|
| Suspect | `src/roles/suspect.py` | Stateful agent. Initialized with random starting stance (±), latent persona, and aligned argument bank. Only surfaces arguments consistent with current stance (`aligned_only` mode). |
| Interrogator | `src/roles/interrogator.py` | **Control:** receives only topic + suspect utterance. **Treatment:** runs 2-stage RAG pipeline (Stage 1: technique selector; Stage 2: argument-aware response generator using Scharff-style indirect deployment). |
| Judge | `src/roles/judge.py` | Scores every suspect utterance on scale −2 to +2 vs. the topic proposition. Batched quad scoring (14 utterances per call) to enforce a consistent within-quad scale. Clamps out-of-range scores; per-utterance fallback on batch failure. |

### RAG (two channels, `src/rag/`)

- **Channel 1 — Technique Recommender:** 18 technique cards (Reid ×5, Scharff ×7, debate ×6) in `data/techniques/*.md`, embedded with `all-MiniLM-L6-v2` into ChromaDB. Phase-filtered (opening/middle/closing) before retrieval. `k=3` candidates passed to Stage-1 selector.
- **Channel 2 — Argument Recommender:** Topic argument strings (from `data/topics/*.yaml`) embedded individually. `k=3` same-side arguments retrieved per turn for Stage-2 generator.

### Experimental Design

- **Quad:** 4 legs sharing the same seed — 2 conditions (control/treatment) × 2 starting directions (pro/anti). Direction-matching cancels stance-initial effects.
- **Scale:** 5 quads × 4 topics × 4 legs × 6 turns = **480 dialogue turns, 80 experimental legs** (40 per condition).
- **Topics:** Housing (Rezoning), Arts (Funding), Transit (Fare-free), Measure V (Davis Village Farms — verbatim real arguments from the June 2, 2026 special election)

### Metrics (computed in `src/metrics.py`, plotted by `analysis.py`)

| Metric | Definition | Notes |
|---|---|---|
| Magnitude | |score_final − score_initial| | Noisy due to judge scale drift on complex propositions. Secondary metric. |
| **Directional Accuracy** | Fraction of legs where final shift direction matches interrogator target | **Primary metric.** Depends only on sign of shift; much less sensitive to scale drift. |
| Consistency | Within-trial variance of the score trajectory | Lower = smoother stance evolution. |

---

## 3. Authoritative Experiment Results

From `results/summary.txt` (run 2026-05-28, 80 legs):

```
Housing (Rezoning)     control  dir_acc=40%   mag=2.30  consistency=1.83
Housing (Rezoning)     treatment dir_acc=80%  mag=1.60  consistency=1.09
                       effect = −0.70 mag

Arts (Funding)         control  dir_acc=60%   mag=1.50  consistency=1.30
Arts (Funding)         treatment dir_acc=60%  mag=1.80  consistency=1.08
                       effect = +0.30 mag

Transit (Fare-free)    control  dir_acc=10%   mag=1.20  consistency=1.30
Transit (Fare-free)    treatment dir_acc=90%  mag=2.10  consistency=0.88
                       effect = +0.90 mag  ← STRONGEST RESULT

Measure V (Village Farms)  control  dir_acc=30%  mag=2.00  consistency=1.65
Measure V (Village Farms)  treatment dir_acc=70%  mag=1.60  consistency=1.16
                       effect = −0.40 mag

POOLED:  control dir_acc=35%  |  treatment dir_acc=75%  |  Δ = +40 pp
```

**Key interpretive notes:**
- Directional accuracy is the reliable signal. Magnitude is confounded by judge scale drift (llama3.1:8b assigns systematically inconsistent absolute scores on complex propositions like Housing and Measure V).
- Arts is the only no-improvement topic. Reason: arts-funding arguments are values-based (equity, culture); Reid/Scharff techniques are calibrated for factual-claim disputes.
- Measure V uses **verbatim** arguments from the real Davis ballot measure — it is the real-data generalization test.
- All roles use `llama3.1:8b`, introducing a known shared-bias confound (Estornell & Liu, NeurIPS 2024): the judge may reward argumentative styles correlated with the suspect's training rather than genuine persuasiveness.

---

## 4. Paper Structure and Key Claims

The paper is in **ACM sigconf format**, ≤8 pages. The draft has been revised as of 2026-05-29. Sections and their key content:

| Section | Key claim / content |
|---|---|
| Abstract | RAG-guided technique recommendation → 75% vs. 35% directional accuracy (40pp improvement) across 80 legs. Transit: 10%→90%. Measure V (real data) replicates gain. |
| Introduction | Frames the problem as belief-layer intervention, not content-layer filtering. RecSys framing: query = conversational state, catalog = technique cards. Positions within prosocial/constructive-dialogue AI research. |
| Related Work | 5 subsections (see §5 below). Each subsection ends with a sentence distinguishing our approach. |
| System Design | Three-role architecture; technique corpus (18 cards); two-channel RAG; two-stage treatment pipeline; phase-filtered retrieval; robustness fixes (clamping, per-utterance fallback). |
| Experimental Design | Quad structure; 4 topics; 3 metrics (magnitude, directional accuracy, consistency); scale (80 legs). |
| Results | Per-topic table + 3 figures. Magnitude mixed/secondary; directional accuracy primary; consistency lower in treatment. |
| Discussion | Why treatment helps; Arts exception (values vs. facts); limitations (shared-bias, scale drift, small N, technique logging gap, single real dataset). |
| Ethical Considerations | Research tool only; Reid technique criticism noted; compassionate framing: individuals with entrenched views have unmet needs, not adversaries. |
| Conclusion | Summary of findings; future work: held-out judge, technique logging, broader real-data, SNA-based network targeting extension. |
| Contribution Statement | Per-person contribution one-liner. |

---

## 5. Related Work — Cited Papers

All three venue-qualifying papers (assignment requirement: ≥3 from top venues in past 3 years) are included:

| Paper | Venue | Role in paper |
|---|---|---|
| Estornell & Liu 2024. "Multi-LLM Debate: Framework, Principals, and Interventions" | **NeurIPS 2024** | Shared-bias failure mode; multi-agent debate baseline for comparison |
| Hong et al. 2024. "Curiosity-Driven Red-Teaming for Large Language Models" | **ICLR 2024** | LLM-based content moderation; motivates robust intervention beyond passive filtering |
| Yu et al. 2023. "Towards Better Dynamic Graph Learning: New Architecture and Unified Library" (DygFormer) | **NeurIPS 2023** | Temporal dynamics modeling; cited in RAG section + Future Work |
| Costello et al. 2024. "Durably reducing conspiracy beliefs through dialogues with AI" | Science | Baseline for conversational belief change without a technique catalog |
| Kumar et al. 2023. "Watch Your Language: Investigating Content Moderation with LLMs" | arXiv | LLM content moderation context |
| Izacard et al. 2022. "Few-shot Learning with Retrieval Augmented Language Models" | arXiv | RAG foundation |
| Ribeiro et al. 2018. "Like Sheep Among Wolves": Characterizing Hateful Users on Twitter | arXiv:1801.00317 | Network centrality of hateful communities; cited in new Network Science subsection + Future Work |
| Sun et al. 2023. "Explicit Time Embedding Based Cascade Attention Network for Information Popularity Prediction" | **[Venue TBD — editor must complete this citation before submission]** | Cascade/influence propagation; cited alongside DygFormer in Future Work |
| Leo 2008. *Police Interrogation and American Justice* | Harvard U Press | Reid technique criticism; ethical considerations |
| Oleszkiewicz et al. 2014. "On the Technique of Eliciting Information: The Scharff Technique" | Applied Cognitive Psychology | Scharff technique source |
| Gorwa et al. 2020. "Algorithmic content moderation" | Big Data & Society | Content moderation background |

**Important:** The draft was recently corrected — the Estornell reference previously had a completely wrong title ("Scaling LLM Test-Time Compute...") and wrong co-authors. The correct citation is now in the document.

---

## 6. Changes Made to the Paper on 2026-05-29

The following revisions were applied programmatically. All are saved in the current `paper.docx`:

1. **Introduction** — appended framing sentence situating the work within "constructive dialogue and healthy discourse" AI research (not just "moderation")
2. **Related Work / Costello** — added: our approach makes strategy selection an explicit recommendation problem vs. their implicit pretraining-driven strategy
3. **Related Work / Kumar+Hong** — added: we operate at the belief layer (proactive) vs. their detection/removal layer (reactive)
4. **Related Work / Estornell** — rewrote to accurately reflect Estornell & Liu: "debates can stall, converging on the majority opinion embedded in shared training data—including shared misconceptions." Made more layman-accessible. Added contrast: our use case is belief intervention, not answer verification.
5. **Related Work / RAG** — fixed "Tian et al." → "Yu et al." (DygFormer) with accurate description of temporal-dynamics motivation
6. **Related Work** — inserted new subsection **"Network Science and Influence Propagation"** (between RAG and System Design): cites Ribeiro et al. 2018, Yu et al. 2023, Sun et al. 2023; sets up Future Work SNA discussion
7. **Ethical Considerations** — appended compassionate framing: individuals with entrenched views should be treated as people with unmet needs; system should be capable of shifting to motivational/supportive exchange
8. **Conclusion / Future Work** — inserted new paragraph describing the SNA network-targeting extension: identify high-centrality actors via dynamic graph tools, initiate technique-guided dialogue, with compassionate framing
9. **Reference list** — fixed Estornell citation (wrong title + wrong co-authors → corrected)
10. **Reference list** — added Ribeiro et al. 2018 and Sun et al. 2023 (Sun et al. venue is marked TBD)

**Backup of the pre-revision draft** is at `Document/paper_backup.docx`.

---

## 7. Known Issues / Things Still Needed in the Paper

- **Sun et al. 2023 venue:** Must be looked up and filled in before submission. Currently marked `[Venue TBD]` in the reference list.
- **Citation numbering:** The document uses a numeric citation system (inline `[N]` superscripts). The two new references (Ribeiro, Sun) have been added as text only. Someone needs to assign them citation numbers consistent with the rest of the document and update any inline uses (currently written as author-year inline in the new paragraphs).
- **Abstract numbers:** The abstract cites "75% vs. 35%" which is correct per `results/summary.txt`. However the abstract also mentions "10%→90%" for Transit — verify this is still consistent with the table in the Results section.
- **Table 1 numbers:** The actual numbers in the Results table were not part of the text extraction used for revision. Verify the table values match `results/summary.txt` exactly.
- **Arts Discussion paragraph:** Could be strengthened with a mention of Motivational Interviewing (which is already one of the 18 technique cards in the corpus) as the technique class that would address values-based appeals.
- **Shared-bias limitation:** The paper correctly identifies this but the suggested fix (rerun with `qwen2.5:7b` as Judge) was not executed. If time allows, this would substantially strengthen the paper.

---

## 8. File Structure

```
D:\Academic\2026\ECS_172\Project\          ← project root / git repo
├── analysis.py                            ← M5: produces figures + summary.txt
├── build_slides.py                        ← slide deck helper (untracked)
├── config/
│   ├── models.yaml
│   ├── topics.yaml
│   └── trial_defaults.yaml
├── data/
│   ├── techniques/                        ← 18 .md technique cards
│   └── topics/                            ← 4 .yaml topic cards (3 synthetic + Measure V)
├── src/
│   ├── llm.py                             ← Ollama HTTP wrapper (180s timeout)
│   ├── metrics.py                         ← magnitude, directional accuracy, consistency
│   ├── trial.py                           ← run_trial + run_dialogue
│   ├── quad.py                            ← direction-matched pair batching
│   ├── experiment.py                      ← sweeps quads, writes 3 output files
│   ├── roles/
│   │   ├── judge.py                       ← batched quad scoring, worked-example prompt
│   │   ├── suspect.py                     ← stateful, aligned_only arguments
│   │   └── interrogator.py               ← control + treatment 2-stage RAG pipeline
│   └── rag/
│       ├── index.py                       ← builds ChromaDB index
│       └── retrieve.py                    ← HF_HUB_OFFLINE=1, module-level cache
├── tests/                                 ← smoke tests for all major components
├── results/
│   ├── figures/                           ← fig1.png, fig2.png, fig3.png (committed)
│   ├── summary.txt                        ← authoritative results (committed)
│   ├── quads_*.csv                        ← per-leg quad scores (committed)
│   ├── trials_*.csv                       ← per-turn trial data (committed)
│   └── transcripts_*.jsonl               ← full dialogue transcripts (committed)
└── Document/
    ├── paper.docx                         ← GITIGNORED (share separately)
    ├── paper_backup.docx                  ← GITIGNORED (pre-2026-05-29 backup)
    └── PROJECT_BRIEFING.md               ← this file
```

---

## 9. GitHub Workflow

**Branch:** `william-dev`  
**Remote:** `origin/william-dev` (up to date as of 2026-05-29)

```bash
# Clone / set up
git clone https://github.com/mawroblewski1/interrogating-agents.git
cd interrogating-agents
git checkout william-dev

# Install dependencies
pip install -r requirements.txt   # or: pip install chromadb sentence-transformers docx

# Run the full experiment (requires local Ollama with llama3.1:8b pulled)
python -m src.experiment

# Regenerate figures and summary.txt from existing CSVs
python analysis.py

# Run tests
pytest tests/
```

**To push code changes:**
```bash
git add <specific files>          # never git add -A (avoid committing .env, chroma_db/)
git commit -m "Your message"
git push origin william-dev
```

**The paper (paper.docx) is gitignored and should NOT be committed.** Share the draft directly with teammates.

---

## 10. Tone and Framing Guidance for Paper Editors

- **Primary framing:** AI as a facilitator of *constructive dialogue* and *healthy discourse* — not coercive interrogation. The interrogation techniques are the tool; the goal is prosocial.
- **Avoid:** Framing extremists or people with entrenched views as adversaries. The compassionate framing (people who may need help, conversations that can shift to support) should be present in Ethical Considerations and Future Work.
- **Topics are deliberately non-partisan:** Housing, arts, transit, Davis Measure V. These are civic ballot measures, not hot-button political topics. This was a deliberate choice to keep the paper politically neutral.
- **The long-term vision** (SNA targeting of superspreaders, network-science upstream of dialogue) may be *alluded to* in Future Work but should not overshadow the main findings.
- **Citation style:** ACM format, numeric references. The inline citation markers in the .docx are non-breaking spaces (Unicode U+00A0) following author names — they render as superscript numbers in Word when the bibliography tool is active.
- **Page budget:** ≤8 pages ACM. New content (Network Science subsection, SNA future work paragraph, expanded Ethical Considerations) adds ~0.5 pages. Monitor total length.
