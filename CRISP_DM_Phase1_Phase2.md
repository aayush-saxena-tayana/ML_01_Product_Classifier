# Project 01 — Product Catalog Auto-Classification
## CRISP-DM Phase 1: Business Understanding & Phase 2: Data Understanding

---

## Phase 1: Business Understanding

### 1.1 Business Objective

Master data teams in manufacturing, supply chain, and e-commerce companies manage thousands of product SKUs across item masters, purchase orders, and catalogs. These products arrive with raw names and descriptions (e.g., "SS PIPE 50MM SCH40 6M") and no standard category attached. Teams manually assign categories — a slow, inconsistent, error-prone process that degrades data quality over time.

**The goal:** Build an ML classifier that reads a product name and description and automatically predicts the correct product category. This eliminates manual tagging, ensures category consistency, and enables downstream analytics (inventory grouping, spend analysis, catalog governance).

---

### 1.2 Business Use Case

| Field | Value |
|---|---|
| Use Case | Product Catalog Auto-Classification |
| Business Area | Master Data Management |
| Input Data Family | Text / Documents |
| Data Ingested | Product names, descriptions, attributes, existing category labels |
| Expected Output | Predicted product category (L1 or L1 + L2) |
| Business Action | Master data team reviews AI-suggested categories, bulk-approves correct ones, corrects edge cases |
| Customer Pitch | Classify product descriptions into categories, raw material groups, finished goods groups, or e-commerce taxonomy |

---

### 1.3 Success Criteria

**Technical:** Category F1 score ≥ 0.80 on a held-out test set.

**Business:** A master data analyst should be able to upload a bulk list of uncategorized product descriptions and receive category suggestions in under 30 seconds, review them in a simple UI, and approve or correct them.

---

### 1.4 CRISP-DM Roadmap for This Project

| Phase | Activity | Status |
|---|---|---|
| 1 — Business Understanding | Define objective, success criteria, use case | ✅ This document |
| 2 — Data Understanding | Identify datasets, explore structure, assess quality | ✅ This document |
| 3 — Data Preparation | Clean text, encode labels, split train/test | Pending |
| 4 — Modeling | TF-IDF + Logistic Regression baseline; optional transformer upgrade | Pending |
| 5 — Evaluation | F1 score, confusion matrix, error analysis | Pending |
| 6 — Deployment | Streamlit demo — bulk CSV upload → category predictions | Pending |

---

## Phase 2: Data Understanding

### 2.1 Taxonomy Depth — Explained

This is the most important architectural decision before building the classifier. Here is what each option means:

**Option A — Flat (Single-Level, L1 only)**

Every product gets one label from a single flat list of categories.

Example labels:
- Raw Material
- Packaging Material
- Electrical & Electronics
- Mechanical Components
- Finished Goods — Apparel
- Finished Goods — Food & Beverage
- Consumables & MRO
- Office Supplies

Pros: Simple to build, easy to explain, fast to train, works well with limited data.
Cons: Not granular enough for real master data use — "Raw Material" is too broad to be actionable.

---

**Option B — Two-Level Hierarchy (L1 + L2)**

Each product gets an L1 (broad group) and an L2 (specific sub-group).

Example:

```
Raw Material
  ├── Polymers & Plastics
  ├── Metals & Alloys
  ├── Chemicals & Solvents
  └── Textiles & Fibres

Packaging Material
  ├── Corrugated Boxes
  ├── Flexible Packaging
  └── Labels & Tags

Electrical & Electronics
  ├── Cables & Wiring
  ├── Connectors & Terminals
  └── Switches & Relays
```

Pros: Actionable for real item master governance. Matches how ERP systems like SAP or Oracle structure material groups. Stronger customer pitch.
Cons: Needs more labeled data per sub-category (at least 50–100 examples per L2 class). Slightly more complex to build.

---

**Recommendation for This Project**

Start with **flat (L1) classification** for the MVP. It is faster to build, easier to demo, and good enough to show business value. Once it works and is demonstrated, extend to L2 in the next sprint. This is the standard CRISP-DM approach — build a working baseline before adding complexity.

The public datasets below use flat categories, which aligns perfectly with starting at L1.

---

### 2.2 Public Datasets — Shortlist

These are the best publicly available datasets for this project, ranked by fit.

---

**Dataset 1 (Recommended for MVP): Massive Product Text Classification — Kaggle**

- URL: https://www.kaggle.com/datasets/asaniczka/product-titles-text-classification
- What it contains: Product titles mapped to categories — text input, category label output
- Why it fits: Clean, text-only, classification-ready. Closest match to an item master use case.
- Format: CSV with `title` and `category` columns
- Scikit-learn compatible: Yes, directly

---

**Dataset 2 (Good for richer descriptions): Amazon Products 2023 — HuggingFace**

- URL: https://huggingface.co/datasets/milistu/AMAZON-Products-2023
- What it contains: Product titles, descriptions, and Amazon category hierarchy
- Why it fits: Rich text per product, multi-level categories available (can use L1 only), large volume
- Format: Parquet/HuggingFace datasets library
- Scikit-learn compatible: Yes, load with `datasets` library then convert to DataFrame

---

**Dataset 3 (Hierarchical benchmark): Shopify Product Catalogue — HuggingFace**

- URL: https://huggingface.co/datasets/Shopify/product-catalogue
- What it contains: Real Shopify product titles, descriptions, brands, and Shopify's hierarchical taxonomy
- Why it fits: Multimodal (text + image), real e-commerce taxonomy, excellent for L1+L2 phase
- Format: HuggingFace datasets
- Scikit-learn compatible: Text features yes; images need CNN

---

**Dataset 4 (B2B/industrial context): Product Classification and Categorization — Kaggle**

- URL: https://www.kaggle.com/datasets/lakritidis/product-classification-and-categorization
- What it contains: Industrial and B2B-style product descriptions with categories
- Why it fits: Closer to a manufacturing/supply chain item master than e-commerce datasets
- Format: CSV
- Scikit-learn compatible: Yes

---

**GitHub Reference Implementation**

- URL: https://github.com/ndgigliotti/amazon-product-classifier
- What it is: A complete scikit-learn pipeline for Amazon product classification achieving 97% accuracy
- Why useful: Proven feature engineering approach (TF-IDF, n-grams, feature selection) we can adapt directly

---

### 2.3 Recommended Dataset Choice

**Start with Dataset 1 (Kaggle Massive Product Titles)** for Phase 3 and 4. It is the most direct match to the task — product title → category label — and requires zero transformation to get into a scikit-learn pipeline.

**Layer in Dataset 2 (Amazon 2023)** if we want richer product descriptions and more class diversity in the evaluation.

---

### 2.4 Data Fields We Need

Regardless of public or real customer data, the classifier needs:

| Field | Required | Description |
|---|---|---|
| `product_name` | ✅ Yes | Short product title or item description (e.g., "SS Pipe 50mm Schedule 40") |
| `product_description` | Optional | Long-form description of the product |
| `product_attributes` | Optional | Unit of measure, brand, specifications (can be concatenated to description) |
| `category_label` | ✅ Yes (training only) | The correct category — this is the target variable |
| `sub_category_label` | Optional | L2 label if available — used only in Phase 2 (L1+L2 model) |

For inference (live use), only `product_name` and/or `product_description` are needed.

---

### 2.5 Data Quality Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Inconsistent category labels in customer data | High | Clean and standardize labels before training; merge near-duplicates |
| Very short product names (3–4 words) with no description | Medium | Add n-gram features; augment with attributes |
| Class imbalance (many SKUs in one category, few in others) | High | Use class-weighted training; oversample minority classes |
| Mixed languages (product names in Hindi, Marathi, etc.) | Medium if real data | Use multilingual model (paraphrase-multilingual-MiniLM) in transformer phase |

---

## Divya Data Brief — What to Request for Real Customer Data

**Context for Divya:** We are building an ML model that automatically classifies products into categories. We need a sample of the customer's item master to train and test the model. This brief specifies exactly what to request.

---

### What to Ask For

Please request the following from the customer's SAP, Oracle, or ERP system:

**Table / Report:** Item Master / Material Master export

**Fields to request (in priority order):**

1. Item / Material Code (we will anonymize to `SKU_001`, `SKU_002`, etc.)
2. Item / Material Description (short name — 10 to 60 characters)
3. Long Description or Remarks (if available)
4. Material Group or Category (the existing classification — this is our training label)
5. Unit of Measure (e.g., KG, MTR, NOS, PCS)
6. Material Type (e.g., Raw Material, Finished Good, Semi-Finished, Trading Good, Consumable)
7. HSN Code (if available — useful as an additional feature)

**Volume:** Minimum 500 rows, ideally 2,000+ rows. At least 10 products per category for the model to learn.

**Format:** Excel or CSV export is fine.

**What NOT to send:** No supplier names, no pricing, no vendor codes, no employee data, no customer identifiers.

---

### Before Using in the Model

Apply Tayana anonymization rules:
- Replace Item Codes with `SKU_001`, `SKU_002`, etc.
- Do NOT replace product descriptions — the text is the feature, not an identifier
- Remove any column that contains vendor, customer, employee, or pricing information

---

### Example of What a Good Row Looks Like

| SKU_ID | Product_Name | Long_Description | Material_Group | Material_Type | UOM |
|---|---|---|---|---|---|
| SKU_001 | SS PIPE 50MM SCH40 | Stainless Steel Pipe 50mm Schedule 40 6 Metre Length | Metals & Alloys | Raw Material | MTR |
| SKU_002 | HDPE BAG 25KG | High Density Polyethylene Bag 25kg Capacity | Packaging Material | Packaging | NOS |
| SKU_003 | SERVO MOTOR 750W | AC Servo Motor 750 Watt 3 Phase | Electrical | Capital Equipment | NOS |

---

## Next Steps (Phase 3: Data Preparation)

Once dataset is confirmed:

1. Download Dataset 1 from Kaggle
2. Load into a pandas DataFrame
3. Explore class distribution (value_counts on category)
4. Clean text: lowercase, remove special characters, strip numbers where not meaningful
5. Concatenate name + description into one `text` feature column
6. Encode labels with `LabelEncoder`
7. Split 80/20 train/test with stratification
8. Build TF-IDF vectorizer pipeline
9. Fit Logistic Regression → evaluate F1

Shall we proceed to Phase 3 (Data Preparation + Modeling)?
