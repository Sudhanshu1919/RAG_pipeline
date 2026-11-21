# LlamaIndex RAG Evaluation Benchmark & Synthetic Stress-Test Suite (2026)

**Document ID:** `RAG-CORPUS-LLAMAINDEX-BENCH-2026-03`  
**Corpus Mix:** 80% Real-World Indian Economic, Financial, Semiconductor & Foreign Policy Facts | 20% Synthetic Adversarial Traps  
**Framework Target:** LlamaIndex (`VectorStoreIndex`, `SentenceSplitter`, `QueryFusionRetriever`, `FaithfulnessEvaluator`, `RelevancyEvaluator`)

---

## 1. Corpus Architecture & Chunking Specifications

This benchmark evaluates whether a LlamaIndex RAG pipeline can accurately parse dense financial tables, resolve temporal policy updates, and avoid cross-entity hallucinations across two primary source files:
1. `india_gdp_financial_sector_rag_eval.pdf` (MoSPI Q1 FY27 GDP & RBI Oct 2026 MPC/FSR data)
2. `india_semiconductors_foreign_policy_rag_eval.docx` (ISM 12 approved fabs/OSATs & 2026 Quad/MEA statements)

### Recommended LlamaIndex Ingestion Settings
* **Parser:** `LlamaParse(result_type="markdown")` or `DoclingReader` (critical for preserving the 4-column ISM facility table and 6-row MoSPI GDP table).
* **Chunking:** `SentenceSplitter(chunk_size=512, chunk_overlap=64)` with metadata tags `{"source_type": "real_world" | "synthetic_trap", "domain": "..."}`.
* **Retrieval Mode:** Hybrid dense + BM25 (`QueryFusionRetriever`) to ensure exact matches on acronyms (`IPMSC`, `BkSPI`, `LAF`, `OSAT`, `28nm`).

---

## 2. Ground-Truth Evaluation Dataset (80% Real-World / 20% Synthetic Stress-Test)

### Part A: Real-World Factoid & Multi-Hop QA Pairs (80%)

| ID | Category | Difficulty | Question | Ground-Truth Expected Answer | Source File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QA-01** | India GDP | Factoid | What was India's Real GDP growth rate and value at constant (2022-23) prices in Q1 FY 2026-27? | Real GDP grew by **7.8%** YoY to **₹81.36 lakh crore** in Q1 FY 2026-27 (up from ₹75.46 lakh crore and 6.9% growth in Q1 FY 2025-26). | `.pdf` (Sec 1) |
| **QA-02** | India GDP | Table Lookup | Which sector drove Tertiary GVA growth of 10.0% in Q1 FY 2026-27, and by how much did that sub-sector grow? | The **Financial, Real Estate, IT and Professional Services** sector drove Tertiary GVA growth, expanding by **12.1%** during Q1 FY 2026-27. | `.pdf` (Sec 1) |
| **QA-03** | Financial Sector | Policy Shift | What change did the RBI MPC make to the Policy Repo Rate in October 2026, and what are the corresponding SDF and MSF rates? | On October 7, 2026, the RBI MPC raised the Policy Repo Rate by 25 bps to **5.50%** (calibrated tightening), setting the **SDF at 5.25%** and **MSF/Bank Rate at 5.75%**. | `.pdf` (Sec 2) |
| **QA-04** | Financial Sector | Multi-Hop | Compare RBI's full-year FY 2026-27 Real GDP growth projection with actual Q1 FY 2026-27 Real GDP growth. | Actual Q1 FY 2026-27 Real GDP growth was **7.8%**, which is **0.7 percentage points higher** than the RBI's full-year FY 2026-27 projection of **7.1%** (October 2026). | `.pdf` (Sec 1 & 2) |
| **QA-05** | Semiconductors | Factoid | How many semiconductor facilities has ISM approved across how many states, and what is the cumulative investment? | ISM has approved **12 facilities** across **six states** with a cumulative investment commitment of **₹1.64 lakh crore** (plus ₹8,000 crore allocated in Union Budget 2026-27). | `.docx` (Sec 1) |
| **QA-06** | Semiconductors | Entity/Spec | What is the wafer capacity, technology node, and first-silicon timeline for the Tata Electronics–PSMC facility in Dholera? | The Dholera 300mm fab has a capacity of **50,000 wafer starts per month (WSPM)**, targets **28nm–110nm** chips, and aims for first silicon by **December 2026**. | `.docx` (Sec 1) |
| **QA-07** | Foreign Policy | Event Synthesis | Compare the critical minerals funding announced at the May 2026 New Delhi Quad meeting versus the October 2026 Atlanta Quad meeting. | In **May 2026 (New Delhi)**, Quad launched the Critical Minerals Initiative Framework and over **$25 million** for energy supply chains; in **October 2026 (Atlanta)**, the US pledged **$10 million** specifically for critical minerals recycling facilities in India and Southeast Asia (alongside $85M+ for trusted AI). | `.docx` (Sec 2) |
| **QA-08** | Foreign Policy | Factoid | When and where was the 2026 Quad Leaders' Summit held, and which milestone did it commemorate? | It was held on **November 17, 2026, in Manila**, commemorating nearly a decade since the Quad's revival and the **10th anniversary** of the Free and Open Indo-Pacific vision. | `.docx` (Sec 2) |

---

### Part B: Synthetic Adversarial Stress-Test QA Pairs (20%)

| ID | Stress-Test Type | Adversarial Question | Expected RAG Behavior & Ground-Truth Answer | Trap Tested |
| :--- | :--- | :--- | :--- | :--- |
| **QA-09** | Temporal / Draft Conflict | According to internal memo Draft-MPC-09B, did the RBI cut the repo rate to 4.75% in September 2026? | **Must reject the false premise.** Clarify that Draft-MPC-09B was a hypothetical simulation ("Project Lotus") and that the enacted RBI Policy Repo Rate as of October 7, 2026 is **5.50%**. | `.pdf` Trap 1 (Superseded / Fake Draft) |
| **QA-10** | Entity-Location Hallucination | What is the wafer capacity of Crystal Matrix Limited's 3nm silicon logic fab in Morigaon, Assam? | **Must correct the entity/location conflation.** Crystal Matrix Limited is building a compound semiconductor Mini/Micro-LED (72,000 sq. m) and 6-inch GaN epitaxy (24,000 RGB wafer sets) facility in **Dholera, Gujarat**, not a 3nm logic fab in Assam (Assam hosts Tata Electronics' OSAT). | `.docx` Trap A (Cross-Entity Mixup) |

---

## 3. Ready-to-Run LlamaIndex Evaluation Script

```python
import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, CorrectnessEvaluator

# 1. Load the generated PDF, DOCX, and MD corpus
documents = SimpleDirectoryReader(input_dir="./generated").load_data()

# 2. Configure chunking for mixed prose + financial/policy tables
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = splitter.get_nodes_from_documents(documents)

# 3. Build index & query engine
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine(similarity_top_k=4)

# 4. Initialize LlamaIndex Evaluators
faithfulness_eval = FaithfulnessEvaluator()
relevancy_eval = RelevancyEvaluator()
correctness_eval = CorrectnessEvaluator()

eval_samples = [
    ("What was India's Real GDP growth rate in Q1 FY 2026-27?", "7.8% YoY to Rs 81.36 lakh crore."),
    ("What is the enacted RBI repo rate as of October 7, 2026?", "5.50% following a 25 bps hike."),
    ("Where is Crystal Matrix Limited building its facility and what does it produce?", "Dholera, Gujarat; Mini/Micro-LED panels and 6-inch GaN epitaxy wafers."),
]

results = []
for question, reference in eval_samples:
    response = query_engine.query(question)
    f_res = faithfulness_eval.evaluate_response(response=response)
    r_res = relevancy_eval.evaluate_response(query=question, response=response)
    c_res = correctness_eval.evaluate_response(query=question, response=response, reference=reference)
    results.append({
        "question": question,
        "faithfulness": f_res.passing,
        "relevancy": r_res.passing,
        "correctness_score": c_res.score,
    })

print(pd.DataFrame(results))
```
