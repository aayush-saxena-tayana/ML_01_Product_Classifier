# CRISP-DM Phase 5 & 6 — Modeling and Evaluation
## Project 01: Product Catalog Auto-Classification
**Date:** 2026-06-23 | **Dataset:** Amazon Products 2023 (milistu/AMAZON-Products-2023) | **Taxonomy:** UNSPSC Segment (L1)

---

## Phase 5: Modeling

Three models were trained and evaluated on the same 80/20 stratified train/test split (72,940 train / 18,235 test rows, 12 UNSPSC segments).

### Model 1 — TF-IDF + LinearSVC (Baseline)

A FeatureUnion of word n-grams (1–2) and character n-grams (3–5) was vectorized using TF-IDF (sublinear scaling, max 200K features each). A LinearSVC with `class_weight='balanced'` was trained to handle the 239x class imbalance. Training time: 71 seconds.

### Model 2 — FastText

Product text was formatted as `__label__SegmentName text` and passed to FastText's supervised trainer (lr=0.5, 25 epochs, word bigrams, 100-dim embeddings, minCount=5). FastText's internal subword character n-grams handle product codes and abbreviations natively. Training time: 11 seconds.

### Model 3 — SentenceTransformers (Zero-Shot Semantic Search)

`all-MiniLM-L6-v2` encoded each of the 12 UNSPSC segment descriptions into anchor embeddings. Test products were encoded in batches and assigned to the nearest segment by cosine similarity. No training data was used — this is a retrieval approach, not a classifier.

---

## Phase 6: Evaluation

### Model Comparison

| Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| **LinearSVC (TF-IDF + char n-grams)** | **0.8469** | **0.9003** | **90.1%** |
| FastText | 0.8301 | — | 88.96% |
| SentenceTransformers (zero-shot) | 0.4474 | — | — |

**Winner: LinearSVC.** Macro F1 is the primary metric because it weights all 12 classes equally regardless of size — critical given the 239x imbalance between Apparel (40.7%) and IT/Telecom (0.2%).

### Per-Class Performance

| UNSPSC Segment | F1 | Notes |
|---|---|---|
| Apparel and Luggage | 0.97 | Largest class, cleanest vocabulary |
| Audio and Visual Equipment | 0.94 | Distinct terms (guitar, headphones) |
| Electronic Components | 0.93 | Strong signal from model/part numbers |
| Animals and Birds and Fish | 0.91 | Distinct pet vocabulary |
| Transportation and Storage | 0.88 | Automotive parts terminology |
| Domestic Appliances | 0.88 | Broad but distinctive |
| Pharmaceuticals and Healthcare | 0.85 | Drug names are strong features |
| Office Equipment and Supplies | 0.84 | Overlaps with Arts & Crafts |
| Building and Construction | 0.80 | Bleeds into Industrial |
| IT/Telecom | 0.81 | Unreliable — only 31 test samples |
| Sports and Recreation | 0.73 | Athletic apparel confuses with Apparel segment |
| **Industrial Machinery** | **0.62** | **Hardest class — see below** |

### Root Cause: Industrial Machinery (F1 = 0.62)

Industrial Machinery is the weakest class because it shares vocabulary with two adjacent segments: "Building and Construction" (drills, pumps, valves appear in both) and "Electronic Components" (industrial sensors, PLCs, controllers). This is a taxonomy overlap problem, not a model failure. Mitigation options: merge the two industrial segments, or collect more segment-specific labeled data from actual procurement records.

### Why SentenceTransformers Failed (Macro F1 = 0.45)

Zero-shot semantic retrieval depends on anchor descriptions being clearly distinct. When 12 categories share overlapping vocabulary (tools, equipment, accessories), cosine similarity cannot reliably separate them without training signal. This result demonstrates a key real-world finding: **a trained classifier on domain-relevant data outperforms a zero-shot foundation model**, even when the trained model is a simple linear one. For SentenceTransformers to be competitive here, the encoder would need to be fine-tuned on labeled product text.

---

## Decisions and Recommendations

**Production model: LinearSVC pipeline** (`models/model_svc.pkl` + `models/label_encoder.pkl`).

- Fast to retrain (71 seconds on 73K rows), scales to millions of rows
- Interpretable — top TF-IDF features per class are readable by domain experts
- 90% accuracy on consumer product data; expected to perform better on B2B procurement data where category vocabulary is more distinctive

**For Divya's procurement data (next phase):**

1. Concatenate `Product_Name + Long_Description + Material_Type + UOM` into a single text field — same pipeline, no architecture change needed
2. Re-run label mapping using actual UNSPSC codes from the customer's SAP/Oracle system rather than the Amazon-to-UNSPSC approximation used here
3. The UNSPSC dataset (hospital procurement, 94K rows) can serve as supplementary training data specifically for segments 51 (Pharma), 42 (Medical Equipment), and 44 (Paper/Stationery) if those appear in the customer's catalog
4. Watch Industrial Machinery F1 closely — B2B catalogs are heavy in this segment

**For L2 classification (Family level):** Retrain the same LinearSVC pipeline with UNSPSC Family codes (4-digit) as labels. Expect lower F1 due to finer granularity — more classes, less data per class.

---

## Deliverables

| File | Description |
|---|---|
| `models/model_svc.pkl` | Production LinearSVC pipeline |
| `models/label_encoder.pkl` | UNSPSC segment label decoder |
| `models/model_fasttext.bin` | FastText model (reference) |
| `data/train.parquet` | 72,940-row training set |
| `data/test.parquet` | 18,235-row held-out test set |
| `evaluation/classification_report.csv` | Full per-class precision/recall/F1 |
| `figures/confusion_matrix.png` | Normalized confusion matrix (best model) |
| `figures/category_distribution.png` | Class distribution chart |
