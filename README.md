# Product Auto-Classifier

**Built by [Tayana Solutions](https://tayanasolutions.com)**

> Automatically classify product names and descriptions into industry-standard UNSPSC categories — in seconds, at scale.

---

## What Is This?

Procurement and supply chain teams spend enormous time manually sorting product SKUs into spend categories. A single analyst can process a few hundred items per day. Misclassified items lead to wrong spend reports, failed compliance checks, and bad supplier negotiations.

This project is an ML-powered tool that reads a product name or description and instantly assigns it the correct **UNSPSC Level 1 Segment** — the global procurement taxonomy used in SAP, Oracle, Ariba, and every major ERP system.

You give it: `"Drawtex 4" x 4" Hydroconductive Dressing sterile"`
It returns: `Pharmaceuticals and Healthcare Products — 94.2% confidence`

It works as a web app: paste one product, or upload a full spreadsheet and download results in seconds.

---

## Results

| Model | Training Rows | Macro F1 | Notes |
|---|---|---|---|
| **FastText** ← production | 5.5M | **0.8613** | Trains in ~3 min; runs in the Flask app |
| LinearSVC + TF-IDF | 1M (RAM-capped) | 0.847 | Kept as backup |

### Per-Class Accuracy (FastText, 15 UNSPSC Segments)

| UNSPSC Segment | F1 Score |
|---|---|
| Apparel and Luggage and Personal Care Products | 0.967 |
| Audio and Visual Presentation and Composing Equipment | 0.939 |
| Electronic Components and Supplies | 0.933 |
| Animals and Birds and Fish | 0.909 |
| Transportation and Storage and Mail Services | 0.884 |
| Domestic Appliances and Supplies and Consumer Electronic Products | 0.878 |
| Pharmaceuticals and Healthcare Products | 0.851 |
| Office Equipment and Accessories and Supplies | 0.843 |
| Information Technology Broadcasting and Telecommunications | 0.815 |
| Building and Construction Machinery and Accessories | 0.800 |
| Sports and Recreational Equipment and Supplies | 0.725 |
| Industrial Machinery and Equipment | 0.618 |

The weakest classes (Industrial, Sports) share vocabulary — steel pipes and gym equipment use similar words. This improves significantly when fine-tuned on client-specific data.

---

## Repo Structure

```
project_01_product_classifier/
│
├── notebooks/                          # Run these in order ↓
│   ├── 01_data_download.ipynb          # Download raw Amazon dataset from HuggingFace
│   ├── 02_data_cleaning.ipynb          # Clean text, map to UNSPSC, split train/test
│   ├── 03_eda.ipynb                    # Exploratory analysis on the training data
│   └── 04_modeling.ipynb               # Train FastText + LinearSVC, save models
│
├── app/                                # Flask web application (production)
│   ├── app.py                          # Routes: /, /predict, /preview, /classify_batch, /health
│   ├── predictor.py                    # Model wrapper — single and batch inference
│   ├── requirements.txt                # App dependencies
│   ├── static/                         # CSS (Tayana brand styles)
│   └── templates/index.html            # Web UI — single product + batch upload tabs
│
├── data/
│   ├── UNSPSCdataset.csv               # UNSPSC reference taxonomy
│   ├── UNSPSCtestDataSet.csv           # Small labeled test set
│   ├── train.parquet                   # 5.5M training rows (output of notebook 02)
│   ├── test.parquet                    # 1.38M test rows (output of notebook 02)
│   ├── train_ft.txt / test_ft.txt      # FastText format files (output of notebook 04)
│   └── category_cache/                 # Raw per-category parquet files (output of notebook 01)
│
├── models/
│   ├── model_fasttext.bin              # Production FastText model (~350 MB)
│   ├── label_encoder.pkl               # 15-class LabelEncoder (must match the model)
│   └── model_svc.pkl                   # LinearSVC backup
│
├── evaluation/
│   └── classification_report.csv       # Per-class precision / recall / F1
│
├── figures/
│   ├── category_distribution.png
│   ├── confusion_matrix.png
│   └── text_length_distribution.png
│
├── docs/
│   ├── CLIENT_FINETUNING_GUIDE.md      # ★ How to retrain on client data
│   ├── CRISP_DM_Phase1_Phase2.md       # Business understanding & data understanding notes
│   ├── CRISP_DM_Phase5_Phase6.md       # Evaluation & deployment notes
│   └── PLAN_Phase3_onwards.md          # Architecture decisions & full build plan
│
├── pyproject.toml                      # Python dependency spec (uv)
└── .gitignore
```

> `data/category_cache/`, `data/train.parquet`, `data/test.parquet`, and `models/` are gitignored. They are reproduced by running the notebooks.

---

## Running the Web App

The app needs `models/model_fasttext.bin` and `models/label_encoder.pkl` to be present. If you already have them:

```bash
conda activate prod_classifier
cd app
python app.py
```

Open `http://localhost:5000` in your browser.

**Verify it's working:**
```bash
curl http://localhost:5000/health
# → {"status": "ok", "model_ready": true, "classes": 15}
```

**Test a prediction via API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "stainless steel centrifugal pump 5HP 415V 3-phase"}'
```

---

## App Features

### Single Product Tab
Paste any product name or description. Get back the top 3 UNSPSC segment predictions with confidence scores. Eight example products are pre-loaded to try immediately.

### Batch Upload Tab
1. Upload a **CSV or Excel file** (up to 50 MB)
2. A preview of the file appears — **select the column that contains product descriptions** (not the ID or SKU column)
3. Click **Classify All Rows**
4. Download the enriched file — two columns are added: `UNSPSC_Segment` and `UNSPSC_Confidence_%`

> **Common mistake:** the column dropdown defaults to the first column in your file. Make sure to change it to the column that has actual product text (e.g. `Description`), not a code or ID column.

### REST API
| Endpoint | Method | What it does |
|---|---|---|
| `/predict` | POST | Classify a single product text. Body: `{"text": "..."}`. Returns top-3 predictions. |
| `/preview` | POST | Upload a file, get column names + row count + first 3 rows. Used by the UI column picker. |
| `/classify_batch` | POST | Upload file + column name, returns enriched CSV. |
| `/health` | GET | Sanity check — confirms model is loaded and returns class count. |

---

## Reproducing Everything from Scratch

Requirements: Python 3.11 via conda, 16 GB+ RAM, ~30 GB free disk space.

### Step 0 — Set up the environment

```bash
conda create -n prod_classifier python=3.11 -y
conda activate prod_classifier
pip install pandas pyarrow scikit-learn fasttext-wheel joblib datasets numpy jupyter
```

### Step 1 — Download the data

Open and run **`notebooks/01_data_download.ipynb`**.

Downloads 30 Amazon product categories from `McAuley-Lab/Amazon-Reviews-2023` on HuggingFace into `data/category_cache/`. Each category is saved as a separate `.parquet` file. Already-downloaded categories are skipped so the download is resumable.

- Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Expected size: ~23 GB total
- Expected runtime: 30–90 minutes depending on connection speed

### Step 2 — Clean and prepare the data

Open and run **`notebooks/02_data_cleaning.ipynb`**.

- Reads each category file one row-group at a time (peak RAM ~200 MB per file, never loads a full file)
- Concatenates `title + description + features` into one `text` field
- Maps Amazon categories to UNSPSC segments (31 Amazon → 15 UNSPSC)
- Caps at 300K rows per category
- Removes short texts (< 3 words) and exact duplicates
- Splits 80/20 into train and test, stratified by class

Output: `data/train.parquet` (~5.5M rows), `data/test.parquet` (~1.38M rows), `models/label_encoder.pkl`

### Step 3 — Explore the data (optional)

Open **`notebooks/03_eda.ipynb`** to see class distribution, text length analysis, vocabulary statistics, and sample products per category.

### Step 4 — Train the models

Open and run **`notebooks/04_modeling.ipynb`**.

Trains two models:

**FastText** (production model)
- Trains on all 5.5M rows
- ~3 minutes on CPU (8 threads)
- Macro F1: 0.8613

**LinearSVC + TF-IDF** (backup model)
- Trains on a stratified 1M-row subsample (RAM constraint)
- Word n-grams (1–2) + character n-grams (3–5)
- ~20 minutes
- Macro F1: 0.847

Output: `models/model_fasttext.bin`, `models/model_svc.pkl`, `evaluation/final_report.json`

### Step 5 — Run the app

```bash
conda activate prod_classifier
cd app && python app.py
```

---

## Dataset

**Amazon Reviews 2023** — McAuley Lab, UC San Diego
([HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023))

| Property | Value |
|---|---|
| Amazon categories used | 30 |
| Rows after cleaning | 5.5M train + 1.38M test |
| UNSPSC Segment classes | 15 |
| Text fields combined | `title` + `description` + `features` |
| Class imbalance ratio | 14× |

### Amazon Category → UNSPSC Segment Mapping

| Amazon Categories | UNSPSC Segment |
|---|---|
| All Beauty, Baby Products, Beauty & Personal Care, Health & Household, Health & Personal Care | Pharmaceuticals and Healthcare Products |
| Amazon Fashion, Clothing Shoes & Jewelry, Handmade Products | Apparel and Luggage and Personal Care Products |
| Appliances, Home and Kitchen | Domestic Appliances and Supplies and Consumer Electronic Products |
| Arts Crafts and Sewing, Office Products | Office Equipment and Accessories and Supplies |
| Automotive | Transportation and Storage and Mail Services |
| Books, Kindle Store, Magazine Subscriptions | Printed Media |
| CDs and Vinyl, Digital Music, Movies and TV, Musical Instruments, Video Games | Audio and Visual Presentation and Composing Equipment |
| Cell Phones and Accessories, Electronics | Electronic Components and Supplies |
| Grocery and Gourmet Food | Food Beverage and Tobacco Products |
| Industrial and Scientific | Industrial Machinery and Equipment |
| Patio Lawn and Garden, Tools and Home Improvement | Building and Construction Machinery and Accessories |
| Pet Supplies | Animals and Birds and Fish |
| Software | Information Technology Broadcasting and Telecommunications |
| Sports and Outdoors, Toys and Games | Sports and Recreational Equipment and Supplies |

---

## Fine-Tuning on Client Data

The base model performs well on general consumer products. For industrial, medical, or highly specialised client inventories, retraining on a small set of labeled client data significantly improves accuracy.

See **[docs/CLIENT_FINETUNING_GUIDE.md](docs/CLIENT_FINETUNING_GUIDE.md)** for:
- What columns to collect from the client
- Minimum labeled rows needed
- Step-by-step retrain scripts for both FastText and LinearSVC
- How to add new UNSPSC labels
- Deployment instructions

---

## Why FastText?

| | FastText | LinearSVC + TF-IDF |
|---|---|---|
| RAM during training | ~2 GB (streams a text file) | OOM on 5.5M rows (14 GB server) |
| Training time | ~3 min on full 5.5M rows | ~20 min on 1M subsample |
| Macro F1 | **0.8613** | 0.847 (fewer rows) |
| Handles unknown words | Yes — subword character n-grams | No — OOV tokens ignored |
| Model file size | ~350 MB | ~1 GB |
| Inference speed | ~50K products/second | Fast |

FastText trains faster, uses less memory, scores higher, and handles abbreviations and product codes better via subword n-grams. LinearSVC is kept as a tested backup.

---

## Tech Stack

| Component | Technology |
|---|---|
| Text classification | [FastText](https://fasttext.cc/) (Meta AI Research) |
| Backup model | scikit-learn LinearSVC |
| Data pipeline | pandas, PyArrow (row-group streaming) |
| Web application | Flask 3.0 |
| Frontend | Bootstrap 5.3 |
| Training environment | conda, Python 3.11 |
| GPU (future phases) | NVIDIA RTX 5070 |

---

## Roadmap

- [ ] UNSPSC L2 classification (Family level) — when client catalog data arrives
- [ ] SentenceTransformer semantic search — embedding-based retrieval as Model 3
- [ ] Project 2: Duplicate SKU detection (entity matching)
- [ ] Multi-tenant API with authentication for SaaS deployment

---

*Built and maintained by **[Tayana Solutions](https://tayanasolutions.com)** · Procurement Intelligence · contact: aayush.s@tayanasolutions.com*
