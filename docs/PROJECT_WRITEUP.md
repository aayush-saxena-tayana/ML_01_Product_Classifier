# Product Auto-Classifier — Technical Write-Up

**Project:** ML-01 · UNSPSC L1 Product Classification  
**Organisation:** Tayana Solutions  
**Status:** MVP complete · v2.0 in production

---

## 1. What We Built

A machine-learning system that reads a product's name and/or description (free text) and assigns it to one of 15 **UNSPSC Level 1 Segments** — the global procurement taxonomy used inside SAP, Oracle, and Ariba. The output is a segment label plus a confidence percentage.

The system has two interfaces:
- **Flask web app** (`localhost:5000`) — single-product text input and bulk CSV/Excel upload
- **REST API** (`POST /predict`, `POST /classify_batch`) — for ERP or catalog system integration

Two models are trained and shipped:
- **FastText** (production) — trained on 5.5M rows in 169 seconds; Macro F1 = 0.86
- **TF-IDF + LinearSVC** (backup) — trained on 1M-row subsample; Macro F1 = 0.85

---

## 2. Why We Built It This Way

### 2.1 Taxonomy choice: UNSPSC over custom labels

We evaluated three options: a custom Amazon category list, a simplified 12-class taxonomy, and UNSPSC.

UNSPSC won because:
- It is the **international standard** for procurement categorisation (used in SAP ARIBA, Oracle iProcurement, Coupa, Jaggaer)
- When a client sees "UNSPSC Segment" as the output, it **maps directly to their existing ERP fields** — no translation layer needed
- It is a credible demo differentiator: a custom label like "Electronics & Electricals" is a toy; "Electronic Components and Supplies (UNSPSC 43)" is enterprise-ready
- L1 (Segment) is the correct starting level for an MVP — L2 (Family) requires much more data per class

### 2.2 Training data: Amazon Reviews 2023, not client data

Client catalog data was not yet available at MVP stage. We needed:
- A large, diverse, English-language product corpus
- Categories that could be mapped to UNSPSC segments with reasonable confidence
- Freely available without licensing issues

Amazon Reviews 2023 (McAuley Lab, UC San Diego) was the best public source. 31 Amazon categories were manually mapped to 15 UNSPSC segments. This mapping is the single most important intellectual asset in this pipeline — it is what makes the training data trustworthy.

### 2.3 Model choice: FastText as primary

We trained three models and selected FastText. Here is why each decision was made:

**Why not a deep learning / transformer model?**
- BERT/SentenceTransformers require 16+ GB GPU VRAM for fine-tuning on 5.5M rows
- Inference latency is 50-200ms per text; FastText is < 1ms
- For UNSPSC L1 (only 15 classes, short product titles), transformers add complexity with no meaningful accuracy gain
- FastText achieves 0.86 Macro F1 — already production-grade for L1 classification
- Transformers remain reserved for L2/L3 when class boundary ambiguity increases

**Why not TF-IDF + LinearSVC as primary?**
- On a 14 GB RAM machine, LinearSVC runs out of memory on 5.5M rows
- Subsampled to 1M rows, it achieves 0.847 Macro F1 — slightly below FastText (0.861)
- FastText's subword n-gram representations handle product code variants (e.g. "SS304", "S.S.Pipe", "ss-pipe") that TF-IDF bags of words miss
- FastText model file (~350 MB) is more portable than a TF-IDF sparse matrix pipeline (~1 GB serialised)

**Why keep LinearSVC at all?**
- It is interpretable: you can inspect which words drive each class
- It is a strong baseline that any data scientist can explain to a client
- When LinearSVC and FastText agree, confidence is higher; when they disagree, it signals an edge case
- It demonstrates to clients that we benchmarked multiple approaches

### 2.4 Data pipeline: PyArrow row-group streaming

The 23 GB raw Amazon cache could not fit in RAM (14 GB machine). We wrote a custom `load_sampled()` function that reads one Parquet row group at a time (~200 MB peak) and samples proportionally. This was the key engineering decision that made the full-dataset training possible on constrained hardware.

### 2.5 Confidence scores

FastText reports raw softmax probabilities. We display them as-is, rounded to one decimal. These are **not** calibrated probabilities (see Section 6 — gaps). For MVP, uncalibrated confidence is sufficient to give users a sense of certainty; it should not be used for automated cutoff decisions without calibration.

---

## 3. Dataset Summary

| Property | Small (Pilot) | Large (Production) |
|---|---|---|
| Source | `milistu/AMAZON-Products-2023` | `McAuley-Lab/Amazon-Reviews-2023` |
| Raw size on disk | ~1.6 GB (4 arrow files) | ~23 GB (31 parquet files) |
| Rows after cleaning | ~117K | 5.5M train + 1.38M test |
| Amazon categories | 12 | 31 |
| UNSPSC segments (classes) | 12 | 15 |
| Imbalance ratio | 239× | 14× |
| Cleaning runtime (local) | < 30s | ~142s |
| FastText training runtime (local) | ~20s | 169s |

The large dataset is dramatically better on imbalance (14× vs 239×) because it has more categories and we cap each at 300K rows, evening out class sizes.

---

## 4. Model Performance

### 4.1 FastText — large dataset (production model)

Trained on 5.5M rows, 15 UNSPSC segments, 15 epochs, 8 threads.

**Macro F1 = 0.8613 · Weighted F1 = 0.8700**

| UNSPSC Segment | Precision | Recall | F1 | Test Rows |
|---|---|---|---|---|
| Apparel and Luggage and Personal Care | 0.969 | 0.966 | **0.967** | 7363 |
| Audio and Visual Presentation | 0.948 | 0.930 | **0.939** | 370 |
| Electronic Components and Supplies | 0.924 | 0.942 | **0.933** | 1893 |
| Animals and Birds and Fish | 0.888 | 0.931 | **0.909** | 375 |
| Domestic Appliances and Consumer Electronics | 0.871 | 0.886 | **0.878** | 2819 |
| Transportation and Storage and Mail | 0.868 | 0.902 | **0.884** | 1192 |
| Pharmaceuticals and Healthcare Products | 0.857 | 0.845 | **0.851** | 692 |
| Office Equipment and Supplies | 0.842 | 0.845 | **0.843** | 787 |
| IT Broadcasting and Telecommunications | 0.957 | 0.710 | **0.815** | 31 |
| Building and Construction Machinery | 0.800 | 0.800 | **0.800** | 1580 |
| Sports and Recreational Equipment | 0.747 | 0.705 | **0.725** | 644 |
| Industrial Machinery and Equipment | 0.669 | 0.575 | **0.618** | 489 |

**Weak classes explained:**
- **IT / Telecom (F1 0.815)**: Only 31 test rows — too few to evaluate reliably; training data was thin for "Computers" vs "Cell Phones" combined
- **Sports (F1 0.725)**: Vocabulary overlap with Industrial (barbell = sports; metal rod = industrial) and Apparel (athletic wear)
- **Industrial (F1 0.618)**: Most challenging — "SS Pipe 50MM" could be Construction or Industrial; these two segments have near-identical vocabulary

### 4.2 LinearSVC — small dataset (pilot / backup)

Trained on 117K rows, 12 UNSPSC segments, TF-IDF word + char n-grams.

**Macro F1 = 0.8469 · Weighted F1 = 0.9003 · Accuracy = 90.1%**

Stored in `models/model_svc.pkl`. Not currently serving predictions in the app (FastText is primary), but available for comparison and future A/B testing.

---

## 5. Azure ML Cost Estimate

> **Disclaimer:** Azure prices vary by region, subscription type, and change over time. All figures below are approximate using **East US, Linux, Pay-As-You-Go** rates as of 2025. Use the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) for exact quotes before budgeting.

### 5.1 Small dataset (117K rows, 1.6 GB)

**Recommended compute:** `Standard_DS3_v2` — 4 vCPU, 14 GB RAM, ~$0.27/hr  
*(Same spec as our development machine; we know it handles this workload.)*

| Phase | Estimated Time | Cost |
|---|---|---|
| Environment setup (conda/pip) | 8 min | $0.04 |
| Data upload to Azure Blob (1.6 GB) | 2 min | $0.01 |
| Data download to compute node | 2 min | $0.01 |
| Cleaning + preprocessing | 1 min | < $0.01 |
| LinearSVC training | 5 min | $0.02 |
| FastText training | 2 min | $0.01 |
| Evaluation + results upload | 3 min | $0.01 |
| **Total per training run** | **~23 min** | **~$0.11** |

Storage cost (1.6 GB in Azure Blob, hot tier):  
`1.6 GB × $0.018/GB/month = ~$0.03/month`

**Cost for 10 experiment runs (hyperparameter search):** ~$1.10  
**Monthly storage:** ~$0.03  
**Full experiment budget: < $2**

---

### 5.2 Large dataset (5.5M rows, 23 GB cache)

**Recommended compute:** `Standard_E4s_v3` — 4 vCPU, **32 GB RAM**, ~$0.25/hr  
*(32 GB allows LinearSVC to run on a 2M-row subsample without paging; FastText needs only ~2 GB regardless.)*

If you want LinearSVC on the full 5.5M rows without subsampling, use `Standard_E8s_v3` (64 GB RAM, ~$0.50/hr) — adds ~$0.50 per run.

| Phase | Estimated Time | Cost (E4s_v3) |
|---|---|---|
| Environment setup | 8 min | $0.03 |
| Data upload to Azure Blob (23 GB, one-time) | 20 min | $0.08 |
| Data mount / stream from Blob (no download needed with AML datasets) | — | — |
| Cleaning (run_cleaning.py) | 5 min | $0.02 |
| FastText training | 5 min | $0.02 |
| LinearSVC on 2M subsample (32 GB RAM) | 25 min | $0.10 |
| Evaluation + logging | 5 min | $0.02 |
| Compute idle / scale-down | 5 min | $0.02 |
| **Total per training run** | **~53 min** | **~$0.29** |

Storage cost (23 GB in Azure Blob, hot tier):  
`23 GB × $0.018/GB/month = ~$0.41/month`

**Cost for 10 experiment runs:** ~$2.90 compute + $0.41 storage = **~$3.30**  
**Full month of active experimentation (50 runs):** ~$14.50 compute + $0.41 storage = **~$15**

---

### 5.3 GPU compute (future — SentenceTransformers fine-tuning)

If/when we move to transformer-based models:

| VM | GPU | RAM | Rate | Fine-tune 1 epoch (5.5M rows) | Est. cost |
|---|---|---|---|---|---|
| `Standard_NC4as_T4_v3` | 1× T4 (16 GB) | 28 GB | ~$0.53/hr | ~45 min | ~$0.40 |
| `Standard_NC6s_v3` | 1× V100 (16 GB) | 112 GB | ~$3.06/hr | ~20 min | ~$1.00 |

For fine-tuning SentenceTransformers on our task, a T4 is sufficient and cheapest. 10 fine-tuning runs with hyperparameter search: **~$4–8**.

---

### 5.4 Total budget summary

| Scenario | One-time | Per experiment run | 50-run experiment |
|---|---|---|---|
| Small dataset (117K) | $0.03/month storage | $0.11 | ~$5.50 |
| Large dataset (5.5M) | $0.41/month storage | $0.29 | ~$14.90 |
| GPU fine-tuning (future) | same storage | $0.40–1.00 | ~$20–50 |

**Azure ML Spot Instances** reduce compute cost by 60–90% but can be evicted mid-run. Acceptable for long hyperparameter sweeps; risky for production retraining. With spot pricing, a 50-run experiment on the large dataset drops to ~$3–6.

---

## 6. Industry-Grade Assessment — Honest Gaps

This section is intentionally honest. The MVP is strong for a demo; moving to production requires addressing these gaps.

### What we are doing right

| Practice | Status |
|---|---|
| Industry taxonomy (UNSPSC) as output | ✅ Done |
| Multiple models with comparison | ✅ Done |
| Macro F1 as primary metric (correct for imbalanced classes) | ✅ Done |
| Memory-safe streaming pipeline (PyArrow row-group) | ✅ Done |
| Train / test split (no leakage) | ✅ Done |
| Confidence scores with every prediction | ✅ Done |
| REST API for system integration | ✅ Done |
| Batch processing (CSV / Excel) | ✅ Done |
| Gitignore for large files; clean repo structure | ✅ Done |
| CRISP-DM methodology documented | ✅ Done |

### Gaps — what a production-grade system would add

#### G1. No experiment tracking (High Priority)
We cannot compare training runs. If we retrain with different hyperparameters, we lose the previous results unless we manually note them. Industry standard: **MLflow** (open source) or **Azure ML Experiments** (built into Azure ML SDK). Every run should log: hyperparameters, metrics, dataset version, model artefact, runtime.

**Fix:** Add `mlflow.start_run()` and `mlflow.log_metric()` to `run_modeling.py`. Zero-cost on local; ~$0/run on Azure ML.

#### G2. No model registry or versioning (High Priority)
`model_fasttext.bin` is overwritten every time we retrain. There is no record of which model version is in production, when it was trained, or what data it saw.

**Fix:** Azure ML Model Registry (built-in) or MLflow Model Registry. Register each trained model with version number, dataset commit hash, and evaluation metrics. One line: `mlflow.fasttext.log_model()`.

#### G3. Confidence scores are uncalibrated (Medium Priority)
FastText's softmax probabilities do not represent true probabilities. A confidence of "85%" does not mean the model is correct 85% of the time for that class — in practice FastText tends to be overconfident. This matters if confidence is used to automate decisions (e.g., auto-approve if confidence > 90%).

**Fix:** Post-hoc calibration using **Platt scaling** or **isotonic regression** on a held-out validation set. 20 lines of sklearn code; adds no latency.

```python
from sklearn.calibration import CalibratedClassifierCV
# For LinearSVC this is already done via CalibratedClassifierCV.
# For FastText: collect (raw_prob, true_label) pairs on validation set,
# fit an IsotonicRegression, wrap the predictor.
```

#### G4. No input validation or sanitisation (Medium Priority)
The `/predict` and `/classify_batch` routes accept any text. A 10 MB string or a file with 500K rows would block the server for minutes.

**Fix:**
- Text length cap (e.g., max 2000 characters per row)
- Row count cap on batch uploads (e.g., max 50K rows per file, then chunk)
- File type validation beyond extension check (check MIME type / magic bytes)
- Rate limiting: `flask-limiter` (5 requests/minute for single, 2 uploads/minute for batch)

#### G5. No authentication on the API (Medium Priority)
Any device on the same network can call `/predict`. For client demos this is fine; for SaaS deployment it is a security hole.

**Fix:** Add API key header check (`X-API-Key`) or OAuth2. For internal demo, HTTP Basic Auth via nginx proxy is sufficient.

#### G6. No monitoring or observability (Medium Priority)
We have no visibility into: how many predictions are being made, which segments are most common, what texts are being submitted, latency distribution, or error rates.

**Fix:**
- Python `logging` module with structured JSON logs (already partially in place)
- Azure Application Insights SDK (`opencensus-ext-azure`) — logs request counts, latency, exceptions
- A simple SQLite table logging `(timestamp, input_text[:100], top_segment, confidence, latency_ms)` gives basic usage analytics for client demos without infrastructure

#### G7. No data drift detection (Medium Priority)
The model was trained on Amazon product descriptions. If a client's products use different language (e.g., procurement codes, German product names, technical abbreviations), accuracy will silently degrade without any alert.

**Fix:** Log prediction confidence distribution over time. If mean confidence drops below a threshold (e.g., 70%), trigger a retraining alert. More rigorous: use **Evidently AI** or **Azure ML Data Drift** to compare live input distributions against training data distributions.

#### G8. No automated tests (Medium Priority)
`predictor.py` and `app.py` have zero test coverage. A refactor could silently break predictions.

**Fix:** Minimum viable test suite (15 minutes to write):
```python
# tests/test_predictor.py
def test_predict_returns_top3():
    p = ProductClassifier(MODEL_PATH, ENCODER_PATH)
    result = p.predict("Nike Air Max Running Shoes Men Size 10")
    assert len(result) == 3
    assert result[0]['segment'] == 'Apparel and Luggage and Personal Care Products'
    assert 0 < result[0]['confidence'] <= 100

def test_predict_empty_text_returns_empty():
    result = predictor.predict("")
    assert result == []
```

#### G9. LinearSVC subsampled to 1M rows (Low Priority for now)
The LinearSVC model never sees 4.5M of the training rows due to RAM constraints. On a 32 GB Azure node, this could be raised to 2M rows; on 64 GB, the full 5.5M rows.

**Expected improvement:** Macro F1 likely rises from 0.847 → ~0.86–0.87, closer to FastText.

#### G10. No Docker / containerisation (Low Priority for MVP)
Deploying the app to a new machine requires: conda, uv, the right Python version, fasttext compiled from source. A `Dockerfile` reduces this to `docker run`.

**Fix:**
```dockerfile
FROM python:3.11-slim
RUN pip install flask fasttext-wheel pandas openpyxl joblib scikit-learn
COPY app/ /app/
COPY models/ /models/
CMD ["python", "/app/app.py"]
```

#### G11. No CI/CD pipeline (Low Priority for MVP)
Every deploy is manual (`git pull && python app/app.py`). No automated test run on push, no deployment pipeline.

**Fix:** GitHub Actions workflow: on push to `main`, run pytest, lint with ruff, build Docker image.

---

## 7. Scope and Roadmap

### What is proven by this MVP

1. Text-based UNSPSC L1 classification is feasible at production accuracy (0.86 Macro F1) using FastText on 5.5M diverse training examples.
2. The Amazon → UNSPSC mapping is the bottleneck, not the model — more precise mappings would directly improve F1.
3. The batch upload feature is the commercially relevant feature: clients will upload 10K–100K row catalogs, not type products one by one.
4. FastText trains in < 3 minutes on commodity hardware. Retraining on new client data costs almost nothing.

### What is not proven yet

- Performance on **actual client procurement data** (different vocabulary, abbreviations, part numbers, mixed languages)
- Performance below L1 — L2 (UNSPSC Family, 4-digit) requires ~10× more labelled examples per class
- Whether confidence scores are reliable enough to auto-route high-confidence predictions

### Recommended next steps (prioritised)

| Priority | Action | Effort | Value |
|---|---|---|---|
| 1 | Add MLflow experiment tracking to `run_modeling.py` | 2 hours | High — essential before more experiments |
| 2 | Calibrate FastText confidence scores (Platt/Isotonic) | 4 hours | High — needed for auto-routing feature |
| 3 | Add row cap + rate limiting to batch upload route | 2 hours | High — production safety |
| 4 | Write 10 pytest unit tests for predictor + app | 3 hours | High — prevent regressions |
| 5 | Write a `Dockerfile` for the app | 2 hours | Medium — simplifies client demo deployment |
| 6 | Collect first batch of actual client catalog data | — | High — validates real-world accuracy |
| 7 | Retrain LinearSVC on full 5.5M rows (32 GB Azure node) | 1 hour | Medium — closes the gap with FastText |
| 8 | Build L2 classifier (UNSPSC Family) | 2–4 weeks | High — next commercial milestone |
| 9 | SentenceTransformer semantic search as Model 3 | 1 week | Medium — demo differentiator |
| 10 | Project 2: Duplicate SKU detection (entity matching) | 4–6 weeks | High — next product |

### Where this fits in the product vision

```
MVP (done)     →    v1.0 (3 months)      →    v2.0 (6 months)
────────────────────────────────────────────────────────────────
L1 UNSPSC           L2 UNSPSC (Family)        L3 UNSPSC (Class)
FastText            FastText + SBERT           Fine-tuned LLM
Demo UI             Multi-tenant API           ERP connector
Single language     English + Hindi            Multilingual
Batch CSV           Streaming API              Real-time webhook
```

---

## 8. Technology Decisions Reference

| Decision | Chose | Rejected | Reason |
|---|---|---|---|
| Output taxonomy | UNSPSC (standard) | Custom Amazon labels | Enterprise ERP compatibility |
| Primary model | FastText | TF-IDF+SVC, BERT | Speed, RAM, F1 balance |
| Training data | Amazon Reviews 2023 (5.5M) | Hospital UNSPSC data, custom scraping | Scale, diversity, licence-free |
| Data loading | PyArrow row-group streaming | `pd.read_parquet` whole file | 14 GB RAM server constraint |
| App framework | Flask | FastAPI, Django | Simplest; sufficient for MVP |
| App env manager | uv | conda, virtualenv | Fast installs, modern lockfile |
| Training env | conda (Python 3.11) | uv | fasttext compilation reliability |
| GPU acceleration | Not used (CPU only) | cuML, PyTorch | Founder decision — defer to L2 |
| Confidence calibration | Not done | Platt scaling | MVP scope; flagged for v1.0 |

---

*Written by Tayana Solutions · June 2026*
