# Data card — CGPRR synthetic news and source pack

**All data in `collections/cgprr-news/` is synthetic.** Every country, person, organisation, vessel, port and figure is fictional. Any resemblance to real entities is unintended. Fictional entities were chosen deliberately, so the model cannot draw on its own pre-trained knowledge in place of the source pack.

| Path | Contents |
|---|---|
| `articles/A01–A04.md` | Four short news articles (trade, ports, defence, monetary policy). Each one contains claims planted with a known verdict. |
| `source_pack/S01–S10.md` | Ten primary-style documents: press releases, a fact sheet, an annual report, an official statement, a policy statement, a transcript, a statistical release and a wire report. S10 (tourism) is a deliberate distractor. |
| `eval/claims_gold.jsonl` | 30 article claims, each with a gold verdict (`supported` / `unsupported` / `unverified`) and gold evidence paragraph ids. |
| `eval/qa_questions.jsonl` | 16 core-RAG questions, with the expected behaviour fixed before any run. |

Format: Markdown with YAML-style front matter (`id`, `title`, `publisher`/`outlet`, `date`, `synthetic: true`). Paragraphs are separated by blank lines. Paragraph ids (`p1`, `p2`, …) are assigned in order, so citations look like `S03:p1`.

Authoring: written by hand for this assignment, with help from an AI assistant (listed in the README). Licence: CC0 / public-domain dedication by the author.

No personal data. No hidden instructions or prompt-injection text in any data file (brief §5). Injection handling is tested with in-code test fixtures (`tests/`) and adversarial *questions* (`eval/qa_questions.jsonl` Q13, Q14). It is never tested with planted documents.

Gold label class balance (30 claims): supported 16, unsupported 9, unverified 5.
