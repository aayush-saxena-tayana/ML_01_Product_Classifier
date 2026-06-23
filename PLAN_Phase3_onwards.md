# Project 01 — Product Catalog Auto-Classification
## Master Build Plan — Step-by-Step (Approved)

---

## Architecture Decision: UNSPSC as the Taxonomy Standard

After reviewing three external repos, two decisions are locked in:

**Decision 1 — Taxonomy:** Use UNSPSC Segment level (L1) as the category scheme instead of a custom Amazon-mapped list. UNSPSC is the global procurement standard used inside SAP, Oracle, and Ariba. When a customer sees "UNSPSC Segment" as the output category, it maps directly to what their system already uses. This is a far more credible customer demo than a custom label like "Electronics & Electricals."

UNSPSC has 4 levels: Segment (2-digit) → Family (4-digit) → Class (6-digit) → Commodity (8-digit). We classify at Segment for L1. When Divya's data arrives and we go to L2, we classify at Family.

**Decision 2 — Model Architecture:** Add semantic search (SentenceTransformers + UNSPSC database cosine similarity) as Model 3 alongside TF-IDF and FastText. This gives a direct comparison between traditional ML classification and modern embedding-based retrieval. Both are valid production approaches; showing both is a stronger customer demo.

**Decision 3 — DeepMatcher:** Filed for Project 2 (Duplicate SKU Detection). Not relevant here.

---

## Why One Text Column — Not Multiple

A text classifier learns from words and patterns, not from which spreadsheet column those words came from. The model does not care whether "Stainless Steel" appeared in `Product_Name` or `Long_Description` — it sees words.

Multiple columns matter in structured data (numbers, dates, IDs). In text classification, field names add no signal. The signal is entirely in the words.

We concatenate all text fields into one `text` column before the model sees anything:

```
"SS PIPE" + "50MM Schedule 40 Stainless Steel 6M" + "MTR" + "Raw Material"
→ "SS PIPE 50MM Schedule 40 Stainless Steel 6M MTR Raw Material"
→ one string → model classifies → Segment: "Metal and Mineral and Ore"
```

When Divya's data arrives: concatenate `Product_Name + Long_Description + Material_Type + UOM`. Same pipeline, new training data.

---

## Why FastText and Semantic Search Beat TF-IDF

| Method | How it works | Weakness for product text |
|---|---|---|
| TF-IDF (word) | Word frequency weighting, no context | "HDPE" = unknown token; no relationship between "pipe" and "tube" |
| TF-IDF + char n-grams | Also captures character sequences | Partial fix for abbreviations, still no semantics |
| FastText | Word embeddings + internal subword char n-grams | Handles "SCH40", "HDPE", OOV tokens natively; fast on 15M rows |
| SentenceTransformers | Contextual embeddings, retrieval-based | No training data needed; "tube" ≈ "pipe" via semantics |

We run all three and compare on the same test set.

---

## What We Carry Forward from the GitHub Repos

From the original Amazon classifier repo, we keep the dataset source and improve everything else:

| Original choice | Our version |
|---|---|
| Filtered all numbers | Keep numbers — "50MM", "750W", "SCH40" carry signal |
| Dropped shortest 50% | Keep short text — industrial names are short by design |
| Brand terms as primary features | Spec terms matter more than brands in B2B |
| Word tokens only | Word n-grams + FastText + SentenceTransformers |
| ~1M rows (RAM limit) | All 15M rows (64GB RAM, no constraint) |
| Accuracy metric only | Macro F1, per-class report, confusion matrix |

From UNSPSCPrediction: the taxonomy framework (UNSPSC) and the CSV dataset of industrial product descriptions. Download and examine immediately.

From ClassiCore: the semantic search architecture — SentenceTransformers + cosine similarity over UNSPSC description database. Model 3 in Phase 4.

---

## 3-Day Timeline

Total: 20 hours — 4h today + 8h Day 2 + 8h Day 3.

| Day | Phase | Tasks | Hours |
|---|---|---|---|
| Today | Phase 3 | Data download, environment setup, first load | 4h |
| Day 2 | Phase 3 + Phase 4 | Complete data prep, full EDA | 8h |
| Day 3 | Phase 5 + Phase 6 | All 3 models, evaluation, error analysis | 8h |

---

## PHASE 3: Data Preparation — Step-by-Step

### Step 1: Set Up Environment (30 min)

Create a conda or venv environment. Install:

```
pip install pandas pyarrow fastparquet scikit-learn fasttext-wheel sentence-transformers transformers torch datasets matplotlib seaborn wordcloud tqdm jupyter
```

For the UCSD Amazon dataset download, also install:
```
pip install requests gzip
```

Confirm GPU is available:
```python
import torch
print(torch.cuda.is_available())  # must return True
```

---

### Step 2: Download the Amazon Product Metadata Dataset (1–1.5h)

**Source:** UCSD Amazon Review Data (2018), Ni/Li/McAuley. Same dataset as the GitHub reference repo.

**URL:** `https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/`

The metadata files are category-specific gzipped JSON files (one product record per line). Download the `All_Amazon_Meta.json.gz` file (full metadata, ~15M products) or individual category files.

Alternatively, use the Kaggle mirror which has the data in Parquet format — search "Amazon Product 2023" on Kaggle or HuggingFace (`milistu/AMAZON-Products-2023`).

**Load the data:**

```python
import pandas as pd
import json, gzip

def load_amazon_meta(filepath, n_rows=None):
    records = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_rows and i >= n_rows:
                break
            try:
                records.append(json.loads(line.strip()))
            except:
                continue
    return pd.DataFrame(records)

df = load_amazon_meta('All_Amazon_Meta.json.gz')
```

**Fields to keep:**
- `asin` — product ID
- `title` — product name (most important text field)
- `description` — long description (list of strings, join with space)
- `feature` — bullet point features (list of strings, join with space)
- `category` — list from root to leaf (e.g., `["Electronics", "Computers", "Laptops"]`)
- `brand` — brand name (optional, keep for reference)

**Drop all other columns immediately** to save memory.

---

### Step 3: Download and Examine the UNSPSC Dataset (30 min)

Download `UNSPSCdataset.csv` from:
`https://raw.githubusercontent.com/KathiravanNatarajan/UNSPSCPrediction/master/UNSPSCdataset.csv`

Also download the test set:
`https://raw.githubusercontent.com/KathiravanNatarajan/UNSPSCPrediction/master/UNSPSCtestDataSet.csv`

Load and inspect:

```python
unspsc_df = pd.read_csv('UNSPSCdataset.csv')
print(unspsc_df.shape)
print(unspsc_df.head(10))
print(unspsc_df.columns.tolist())
print(unspsc_df.dtypes)
```

Key questions to answer about this dataset:
- How many rows? (tells us if it's trainable on its own or just reference data)
- What are the columns? (material description + UNSPSC code?)
- What UNSPSC level does it classify to? (Segment, Family, Class, or Commodity?)
- Are there product descriptions similar to what Tayana customers would have?

Record your findings — this determines whether we use it as training data or just as a category reference.

---

### Step 4: Build the UNSPSC Segment Mapping (45 min)

The full UNSPSC Segment list has 55 segments. For our L1 classifier, we map Amazon's top-level categories to the closest UNSPSC Segment.

**UNSPSC Segments (abbreviated — the ones most relevant to our training data):**

```python
AMAZON_TO_UNSPSC_SEGMENT = {
    "Electronics": "Electronic Components and Supplies",
    "Computers": "Electronic Components and Supplies",
    "Cell Phones & Accessories": "Electronic Components and Supplies",
    "Clothing, Shoes & Jewelry": "Apparel and Luggage and Personal Care Products",
    "Home & Kitchen": "Domestic Appliances and Supplies and Consumer Electronic Products",
    "Tools & Home Improvement": "Building and Construction Machinery and Accessories",
    "Industrial & Scientific": "Industrial Machinery and Equipment",
    "Automotive": "Transportation and Storage and Mail Services",
    "Sports & Outdoors": "Sports and Recreational Equipment and Supplies",
    "Grocery & Gourmet Food": "Food Beverage and Tobacco Products",
    "Health & Personal Care": "Pharmaceuticals and Healthcare Products",
    "Beauty": "Pharmaceuticals and Healthcare Products",
    "Baby": "Pharmaceuticals and Healthcare Products",
    "Pet Supplies": "Animals and Birds and Fish",
    "Books": "Printed Media",
    "Movies & TV": "Audio and Visual Presentation and Composing Equipment",
    "Music": "Audio and Visual Presentation and Composing Equipment",
    "Video Games": "Audio and Visual Presentation and Composing Equipment",
    "Software": "Information Technology Broadcasting and Telecommunications",
    "Office Products": "Office Equipment and Accessories and Supplies",
    "Toys & Games": "Sports and Recreational Equipment and Supplies",
    "Garden & Outdoor": "Farming and Fishing and Forestry and Wildlife Machinery",
    "Arts, Crafts & Sewing": "Office Equipment and Accessories and Supplies",
}
```

Apply the mapping:

```python
def extract_top_category(category_list):
    """Extract the top-level Amazon category from the category path list."""
    if isinstance(category_list, list) and len(category_list) > 0:
        return category_list[0]
    return None

df['amazon_top_category'] = df['category'].apply(extract_top_category)
df['unspsc_segment'] = df['amazon_top_category'].map(AMAZON_TO_UNSPSC_SEGMENT)
df = df.dropna(subset=['unspsc_segment'])
```

---

### Step 5: Build the Combined Text Column (30 min)

```python
def clean_text(text):
    """Supply-chain-aware text cleaner — keeps numbers and specs."""
    if not isinstance(text, str):
        return ""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # Keep alphanumeric, hyphens, slashes, dots (spec-relevant)
    text = re.sub(r'[^a-zA-Z0-9\s\-\/\.]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def join_list_field(field):
    """Join a list of strings into one string."""
    if isinstance(field, list):
        return ' '.join([str(x) for x in field if x])
    if isinstance(field, str):
        return field
    return ''

# Build text column
df['text'] = (
    df['title'].apply(clean_text) + ' ' +
    df['description'].apply(join_list_field).apply(clean_text) + ' ' +
    df['feature'].apply(join_list_field).apply(clean_text)
).str.strip()

# Drop empty text rows
df = df[df['text'].str.len() > 10].copy()

# Drop rows with fewer than 3 words
df = df[df['text'].str.split().str.len() >= 3].copy()
```

---

### Step 6: Class Distribution Check and Deduplication (20 min)

```python
# Class distribution
print(df['unspsc_segment'].value_counts())
print(f"\nTotal rows: {len(df):,}")
print(f"Unique segments: {df['unspsc_segment'].nunique()}")

# Imbalance ratio
counts = df['unspsc_segment'].value_counts()
print(f"Imbalance ratio: {counts.max() / counts.min():.1f}x")

# Drop exact text duplicates
df = df.drop_duplicates(subset=['text']).copy()
print(f"After dedup: {len(df):,}")

# Drop segments with fewer than 500 examples (too few to train reliably)
min_count = 500
valid_segments = counts[counts >= min_count].index
df = df[df['unspsc_segment'].isin(valid_segments)].copy()
print(f"After min-class filter: {len(df):,}")
print(f"Final segments: {df['unspsc_segment'].nunique()}")
```

---

### Step 7: Train/Test Split and Save (15 min)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['unspsc_segment'])

# Stratified split
train_df, test_df = train_test_split(
    df[['text', 'unspsc_segment', 'label']],
    test_size=0.20,
    stratify=df['label'],
    random_state=42
)

print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

# Save
train_df.to_parquet('data/train.parquet', index=False)
test_df.to_parquet('data/test.parquet', index=False)

# Save label encoder
import joblib
joblib.dump(le, 'models/label_encoder.pkl')

print("Saved train.parquet, test.parquet, label_encoder.pkl")
```

**Phase 3 deliverables:** `data/train.parquet`, `data/test.parquet`, `models/label_encoder.pkl`

---

## PHASE 4: EDA — Step-by-Step (Intern Brief)

**Context:** We have just cleaned the Amazon product dataset and mapped it to UNSPSC Segment labels. Your EDA notebook (`eda.ipynb`) must explore this data and produce the outcome table at the end. Every finding maps to a specific modeling decision. Work on `train.parquet` only — never touch the test set.

**Setup:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

df = pd.read_parquet('data/train.parquet')
print(df.shape)
print(df.head())
```

---

### EDA Section 1: Dataset Overview

```python
# Shape and columns
print(f"Rows: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

# Missing values
print(df.isnull().sum())
print(df.isnull().mean().round(3))  # % missing per column

# Sample rows
df.sample(10)
```

**Write a conclusion:** How many rows are there? Are there any missing values in the `text` or `unspsc_segment` columns?

---

### EDA Section 2: Category (Label) Distribution

```python
# Counts and percentages
counts = df['unspsc_segment'].value_counts()
pct = (counts / len(df) * 100).round(1)
dist = pd.DataFrame({'count': counts, 'pct': pct})
print(dist)

# Bar chart
plt.figure(figsize=(14, 6))
counts.plot(kind='bar', color='steelblue')
plt.title('UNSPSC Segment Distribution (Training Set)')
plt.ylabel('Product Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/category_distribution.png', dpi=150)
plt.show()

# Imbalance ratio
print(f"Largest class: {counts.max():,} ({pct.max()}%)")
print(f"Smallest class: {counts.min():,} ({pct.min()}%)")
print(f"Imbalance ratio: {counts.max() / counts.min():.1f}x")
```

**Write a conclusion:** Is there severe class imbalance (> 10:1 ratio)? This determines whether we use `class_weight='balanced'` or SMOTE in the model.

---

### EDA Section 3: Text Length Distribution

```python
# Word counts
df['word_count'] = df['text'].str.split().str.len()
df['char_count'] = df['text'].str.len()

# Summary stats
print(df[['word_count', 'char_count']].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

# What % of products have very short text?
short_text = (df['word_count'] < 5).mean() * 100
print(f"\n% of products with < 5 words: {short_text:.1f}%")
print(f"% of products with < 10 words: {(df['word_count'] < 10).mean()*100:.1f}%")

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
df['word_count'].clip(upper=200).hist(bins=50, ax=axes[0], color='steelblue')
axes[0].set_title('Word Count Distribution (clipped at 200)')
axes[0].set_xlabel('Word Count')

df['word_count'].clip(upper=200).hist(bins=50, ax=axes[1], color='steelblue')
axes[1].set_xscale('log')
axes[1].set_title('Word Count Distribution (log scale)')
plt.tight_layout()
plt.savefig('figures/text_length_distribution.png', dpi=150)
plt.show()

# Per-category median word count
print(df.groupby('unspsc_segment')['word_count'].median().sort_values())
```

**Write a conclusion:** What is the median word count? What % of products have very short text (< 10 words)? If > 20% are short, char n-grams in FastText are essential.

---

### EDA Section 4: Vocabulary Analysis

```python
from collections import Counter
import re

# Build word frequency
all_words = ' '.join(df['text'].tolist()).split()
word_freq = Counter(all_words)

print(f"Total vocabulary size: {len(word_freq):,}")
print(f"Words appearing only once (noise): {sum(1 for w,c in word_freq.items() if c==1):,}")
print(f"\nTop 30 most frequent words:")
print(word_freq.most_common(30))

# Plot top 50 words
top_words = pd.Series(dict(word_freq.most_common(50)))
plt.figure(figsize=(14, 5))
top_words.plot(kind='bar', color='steelblue')
plt.title('Top 50 Most Frequent Words (Overall)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/top_words_overall.png', dpi=150)
plt.show()

# Top words per category (top 5 unique words per category)
for segment in df['unspsc_segment'].unique():
    segment_text = ' '.join(df[df['unspsc_segment'] == segment]['text'].tolist())
    segment_words = Counter(segment_text.split())
    print(f"\n{segment}: {[w for w,c in segment_words.most_common(10)]}")
```

**Write a conclusion:** Are the top words per category clearly distinct? Or do most categories share the same top words (e.g., "product", "new", "free")? If top words are generic, the model will struggle — we need TF-IDF's IDF weighting to suppress them.

---

### EDA Section 5: Product Code and Abbreviation Detection

```python
# Pattern: 2+ uppercase letters or alphanumeric codes (e.g., HDPE, SCH40, SS316, M12)
code_pattern = re.compile(r'\b[A-Z]{2,}[0-9]*\b|\b[A-Z][0-9]{2,}\b')

# How many rows have at least one product code?
df['has_product_code'] = df['text'].apply(
    lambda x: bool(code_pattern.search(x.upper()))
)
print(f"% of products with product codes: {df['has_product_code'].mean()*100:.1f}%")

# Per category
print(df.groupby('unspsc_segment')['has_product_code'].mean().sort_values(ascending=False))

# Sample codes found
sample_codes = []
for text in df['text'].sample(500).tolist():
    sample_codes.extend(code_pattern.findall(text.upper()))
print(f"\nSample product codes found:")
print(Counter(sample_codes).most_common(20))
```

**Write a conclusion:** What % of products have product codes or abbreviations? If > 30%, TF-IDF alone will miss these patterns — FastText's subword char n-grams are essential.

---

### EDA Section 6: Duplicate and Quality Check

```python
# Exact duplicates (after cleaning)
dup_count = df['text'].duplicated().sum()
print(f"Exact text duplicates: {dup_count:,} ({dup_count/len(df)*100:.1f}%)")

# Same text, different label (label noise)
label_noise = df.groupby('text')['unspsc_segment'].nunique()
noisy = label_noise[label_noise > 1]
print(f"\nTexts with conflicting labels (label noise): {len(noisy):,}")
if len(noisy) > 0:
    # Show examples
    example_text = noisy.index[0]
    print(f"Example: '{example_text[:80]}'")
    print(df[df['text'] == example_text][['text', 'unspsc_segment']])

# Very long texts (potential noise/boilerplate)
long_texts = df[df['word_count'] > 300]
print(f"\nProducts with > 300 words: {len(long_texts):,}")
print("Sample long text (first 200 chars):")
print(long_texts['text'].iloc[0][:200] if len(long_texts) > 0 else "None")

# Very short texts (< 3 words)
short_texts = df[df['word_count'] < 3]
print(f"\nProducts with < 3 words: {len(short_texts):,}")
print(short_texts['text'].head(10).tolist())
```

**Write a conclusion:** Are there significant quality issues? Label noise (same text, different category) is the most dangerous — list the count. If > 1%, flag it for cleaning.

---

### EDA Section 7: Language Check

```python
# Count non-ASCII characters
df['non_ascii_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if ord(c) > 127) / max(len(x), 1)
)

print(f"% of products with any non-ASCII characters: {(df['non_ascii_ratio'] > 0).mean()*100:.1f}%")
print(f"% of products with > 10% non-ASCII characters: {(df['non_ascii_ratio'] > 0.1).mean()*100:.1f}%")

# Sample non-ASCII rows
non_ascii_sample = df[df['non_ascii_ratio'] > 0.1]['text'].head(5).tolist()
for t in non_ascii_sample:
    print(t[:100])
```

**Write a conclusion:** Is the dataset predominantly English? If > 10% of products have significant non-ASCII content, we may need language filtering or a multilingual model.

---

### EDA Final Outcomes Summary Table

Fill this in at the end of `eda.ipynb`. This table drives every decision in Phase 5.

| Question | Finding | Modeling Decision |
|---|---|---|
| Total training rows | | |
| Number of UNSPSC segments (classes) | | |
| Imbalance ratio (largest / smallest class) | | Use `class_weight='balanced'` if > 5x |
| % of products with text < 10 words | | If > 20%, char n-grams essential |
| % of products with product codes / abbreviations | | If > 30%, FastText essential |
| Total vocabulary size | | Sets TF-IDF `max_features` |
| % exact duplicate texts removed | | |
| % label noise (same text, different label) | | If > 1%, deduplicate before training |
| % non-ASCII / non-English content | | If > 10%, add language filter |
| Top 3 categories likely to be confused (shared top words) | | Watch these in confusion matrix |

**This table is the handoff from EDA to modeling. Do not start Phase 5 until this table is complete.**

---

## PHASE 5: Modeling — Step-by-Step

Load data:

```python
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_parquet('data/train.parquet')
test_df = pd.read_parquet('data/test.parquet')

X_train = train_df['text'].tolist()
y_train = train_df['label'].tolist()
X_test = test_df['text'].tolist()
y_test = test_df['label'].tolist()

le = joblib.load('models/label_encoder.pkl')
```

---

### Model 1: TF-IDF + LinearSVC (Baseline)

This is the closest to the original GitHub repo — replicate and improve it.

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import numpy as np

# FeatureUnion: word n-grams + char n-grams
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),       # unigrams and bigrams
    min_df=5,                  # ignore very rare words
    max_features=200000,
    binary=True,               # presence not frequency (as in original repo)
    sublinear_tf=True          # log scaling
)

char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),        # character 3-grams to 5-grams
    min_df=5,
    max_features=200000,
    binary=True,
    sublinear_tf=True
)

features = FeatureUnion([
    ('word', word_vectorizer),
    ('char', char_vectorizer)
])

pipeline_svc = Pipeline([
    ('features', features),
    ('clf', LinearSVC(
        class_weight='balanced',
        max_iter=2000,
        C=1.0
    ))
])

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline_svc, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"LinearSVC CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train on full training set and evaluate on test
pipeline_svc.fit(X_train, y_train)
y_pred_svc = pipeline_svc.predict(X_test)

print(f"\nLinearSVC Test Macro F1: {f1_score(y_test, y_pred_svc, average='macro'):.4f}")
print(f"LinearSVC Test Weighted F1: {f1_score(y_test, y_pred_svc, average='weighted'):.4f}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred_svc, target_names=le.classes_))

# Save model
joblib.dump(pipeline_svc, 'models/model_svc.pkl')
```

---

### Model 2: FastText (Main Upgrade)

FastText requires data in a specific text format: `__label__ClassName text of the product`.

```python
import fasttext
import tempfile, os

def write_fasttext_file(texts, labels, filepath, label_encoder):
    with open(filepath, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            label_str = label_encoder.classes_[label].replace(' ', '_')
            f.write(f"__label__{label_str} {text}\n")

# Write training and validation files
write_fasttext_file(X_train, y_train, 'data/train_ft.txt', le)
write_fasttext_file(X_test, y_test, 'data/test_ft.txt', le)

# Train FastText model
ft_model = fasttext.train_supervised(
    input='data/train_ft.txt',
    lr=0.5,                    # learning rate
    epoch=25,                  # number of passes over training data
    wordNgrams=2,              # word bigrams
    dim=100,                   # embedding dimension
    minCount=5,                # ignore words appearing < 5 times
    loss='softmax',            # softmax for multi-class
    thread=8                   # parallel threads
)

# Evaluate
result = ft_model.test('data/test_ft.txt')
print(f"\nFastText samples: {result[0]}, Precision: {result[1]:.4f}, Recall: {result[2]:.4f}")

# Get per-class F1
ft_preds = [ft_model.predict(text)[0][0].replace('__label__', '').replace('_', ' ') for text in X_test]
ft_labels = [le.classes_[label] for label in y_test]
from sklearn.metrics import f1_score as sk_f1
ft_macro_f1 = sk_f1(ft_labels, ft_preds, average='macro')
print(f"FastText Test Macro F1: {ft_macro_f1:.4f}")
print(classification_report(ft_labels, ft_preds))

# Save model
ft_model.save_model('models/model_fasttext.bin')
```

**Threshold decision:** If FastText Macro F1 > LinearSVC by more than 2 points, FastText is the production model. If not, test Model 3.

---

### Model 3: SentenceTransformers + Semantic Search (UNSPSC retrieval)

This approach does not train a classifier. Instead, it encodes each UNSPSC segment description using a pre-trained sentence encoder and finds the nearest segment for each product by cosine similarity.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model — same as ClassiCore used
model = SentenceTransformer('all-MiniLM-L6-v2')

# UNSPSC segment descriptions — these are the "anchors" for retrieval
UNSPSC_SEGMENTS = {
    "Electronic Components and Supplies": "electronic components semiconductors connectors cables circuit boards",
    "Apparel and Luggage and Personal Care Products": "clothing apparel shoes luggage bags fashion accessories",
    "Domestic Appliances and Supplies and Consumer Electronic Products": "home appliances kitchen equipment household products",
    "Building and Construction Machinery and Accessories": "construction tools hardware building materials equipment",
    "Industrial Machinery and Equipment": "industrial machinery manufacturing equipment factory tools",
    "Transportation and Storage and Mail Services": "automotive vehicles transport storage logistics",
    "Sports and Recreational Equipment and Supplies": "sports equipment outdoor recreation fitness toys games",
    "Food Beverage and Tobacco Products": "food grocery beverages nutrition snacks",
    "Pharmaceuticals and Healthcare Products": "health medical pharmacy personal care beauty cosmetics",
    "Animals and Birds and Fish": "pet supplies animal care veterinary",
    "Printed Media": "books publishing media literature education",
    "Audio and Visual Presentation and Composing Equipment": "audio video media entertainment music movies",
    "Information Technology Broadcasting and Telecommunications": "software computing technology telecommunications",
    "Office Equipment and Accessories and Supplies": "office supplies stationery furniture equipment",
    "Farming and Fishing and Forestry and Wildlife Machinery": "garden outdoor lawn farming agriculture",
}

# Encode UNSPSC segment descriptions
segment_names = list(UNSPSC_SEGMENTS.keys())
segment_descriptions = list(UNSPSC_SEGMENTS.values())
segment_embeddings = model.encode(segment_descriptions, batch_size=32, show_progress_bar=True)

# Encode test products (batch for speed)
print("Encoding test products...")
product_embeddings = model.encode(X_test, batch_size=256, show_progress_bar=True)

# Predict: nearest UNSPSC segment by cosine similarity
similarities = cosine_similarity(product_embeddings, segment_embeddings)
predicted_indices = np.argmax(similarities, axis=1)
y_pred_semantic = [segment_names[i] for i in predicted_indices]
y_test_labels = [le.classes_[label] for label in y_test]

# Evaluate
from sklearn.metrics import f1_score, classification_report
semantic_f1 = f1_score(y_test_labels, y_pred_semantic, average='macro')
print(f"Semantic Search Macro F1: {semantic_f1:.4f}")
print(classification_report(y_test_labels, y_pred_semantic))

# Save embeddings for production use
np.save('models/segment_embeddings.npy', segment_embeddings)
import json
with open('models/segment_names.json', 'w') as f:
    json.dump(segment_names, f)
```

---

### Model Comparison Table

After all three models, print:

```python
results = {
    'LinearSVC (TF-IDF + char n-grams)': f1_score(y_test, y_pred_svc, average='macro'),
    'FastText': ft_macro_f1,
    'SentenceTransformers (Semantic Search)': semantic_f1
}

print("\n=== MODEL COMPARISON ===")
for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: Macro F1 = {score:.4f}")

best_model = max(results, key=results.get)
print(f"\nBest model: {best_model}")
```

---

## PHASE 6: Evaluation — Step-by-Step

Run after the best model is identified.

### Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Use best model's predictions
# Replace y_pred_best with the predictions from the winning model
cm = confusion_matrix(y_test, y_pred_best)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 12))
sns.heatmap(
    cm_pct,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title('Confusion Matrix — Normalized by True Label')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
```

### Error Analysis

```python
# Build error dataframe
error_df = test_df.copy()
error_df['predicted'] = le.inverse_transform(y_pred_best)  # for SVC/FastText
error_df['correct'] = error_df['unspsc_segment'] == error_df['predicted']

print(f"Overall accuracy: {error_df['correct'].mean()*100:.1f}%")
print(f"Total errors: {(~error_df['correct']).sum():,}")

# Top misclassification pairs
errors = error_df[~error_df['correct']]
pair_counts = errors.groupby(['unspsc_segment', 'predicted']).size().sort_values(ascending=False)
print("\nTop 10 misclassification pairs (True → Predicted):")
print(pair_counts.head(10))

# Sample misclassified products for the top confused pair
top_true, top_pred = pair_counts.index[0]
sample_errors = errors[
    (errors['unspsc_segment'] == top_true) &
    (errors['predicted'] == top_pred)
]['text'].head(5).tolist()

print(f"\nSample products misclassified as '{top_pred}' (true: '{top_true}'):")
for s in sample_errors:
    print(f"  - {s[:120]}")
```

### Save Final Report

```python
from sklearn.metrics import classification_report
import json

report = classification_report(y_test, y_pred_best, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('evaluation/classification_report.csv')

print("\n=== FINAL EVALUATION REPORT ===")
print(f"Best Model: {best_model}")
print(f"Macro F1:    {report['macro avg']['f1-score']:.4f}")
print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
print(f"Accuracy:    {report['accuracy']:.4f}")
print("\nPer-class F1 (bottom 5 — problem areas):")
class_f1 = {k: v['f1-score'] for k, v in report.items() if k in le.classes_}
for cls, f1 in sorted(class_f1.items(), key=lambda x: x[1])[:5]:
    print(f"  {cls}: {f1:.4f}")
```

**Phase 6 deliverables:** `figures/confusion_matrix.png`, `evaluation/classification_report.csv`, best model saved in `models/`.

---

## Folder Structure

```
project_01_product_classifier/
├── data/
│   ├── train.parquet
│   ├── test.parquet
│   ├── train_ft.txt           (FastText format)
│   └── test_ft.txt
├── models/
│   ├── label_encoder.pkl
│   ├── model_svc.pkl
│   ├── model_fasttext.bin
│   └── segment_embeddings.npy
├── figures/
│   ├── category_distribution.png
│   ├── text_length_distribution.png
│   ├── top_words_overall.png
│   └── confusion_matrix.png
├── evaluation/
│   └── classification_report.csv
├── data_prep.ipynb
├── eda.ipynb
├── modeling_notebook.ipynb
├── CRISP_DM_Phase1_Phase2.md
└── PLAN_Phase3_onwards.md     (this file)
```

---

## Future: DeepMatcher for Project 2 — Duplicate SKU Detection

Once Project 1 is complete, DeepMatcher (anhaidgroup/deepmatcher) is the right tool for the next project. It takes pairs of product records and determines if they refer to the same item — exactly what's needed to find duplicate SKUs in Divya's item master ("SS Pipe 50mm" vs "SS PIPE 50 MM" vs "Stainless Pipe 2 inch"). File for later.
