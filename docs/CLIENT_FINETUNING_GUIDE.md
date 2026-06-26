# Client Fine-Tuning Guide
## How to Retrain or Fine-Tune on Client Data

---

## 1. Why Retraining Is Needed

The base model was trained on Amazon consumer product listings (~5.5M rows, 15 UNSPSC segments). It performs well on general consumer goods but will underperform on:

- **Industrial / B2B inventory** — raw materials, medical devices, packaging components
- **Highly abbreviated descriptions** — `D1010A`, `DS80-287`, `SS PIPE 50MM SCH40`
- **Client-specific taxonomy** — if the client uses their own category names instead of UNSPSC

The fix is simple: give the model labeled examples from the client's own data and retrain.

---

## 2. What Client Data You Need

### Minimum viable dataset

| Column | Required | Notes |
|---|---|---|
| `description` | Yes | Product name or description. More words = better. Combine name + description + any specs into one field. |
| `unspsc_segment` | Yes | The correct UNSPSC Segment label (L1). Must match the label names the model already knows exactly. |

### Additional columns that help (concatenate them into `description`)

| Column | Why it helps |
|---|---|
| Item Class / Type | "RAW-PACK", "FGS-NCE" add domain signal |
| Unit of measure | "KG", "EA", "MTR" helps with commodity vs. finished good |
| Long description | If available, always include — more text = better accuracy |
| Part number prefix | e.g. "D1010" appearing repeatedly teaches the model your naming conventions |

### Column you do NOT need

| Column | Why to exclude |
|---|---|
| Inventory ID / SKU code | Random alphanumeric codes add noise, not signal |
| Price | Numeric, no text signal |
| Status / Active flag | Same value for all rows, useless |
| Warehouse / Tax codes | Internal accounting codes, not product descriptions |

### Minimum rows required

| Situation | Minimum per label | Recommended |
|---|---|---|
| Fine-tune on top of base model | 50 rows per label | 200+ |
| Full retrain from client data only | 500 rows per label | 2,000+ |
| Adding a new label not in base model | 500 rows for the new label | 2,000+ |

If a label has fewer than 50 rows, either collect more data or merge it into the closest existing label.

---

## 3. Supported UNSPSC Segment Labels

These are the exact 15 label strings the current model knows. Client data must use these names exactly (case-sensitive):

```
Animals and Birds and Fish
Apparel and Luggage and Personal Care Products
Audio and Visual Presentation and Composing Equipment
Building and Construction Machinery and Accessories
Domestic Appliances and Supplies and Consumer Electronic Products
Electronic Components and Supplies
Farming and Fishing and Forestry and Wildlife Machinery
Food Beverage and Tobacco Products
Industrial Machinery and Equipment
Information Technology Broadcasting and Telecommunications
Office Equipment and Accessories and Supplies
Pharmaceuticals and Healthcare Products
Printed Media
Sports and Recreational Equipment and Supplies
Transportation and Storage and Mail Services
```

If the client's products don't map cleanly to these, see Section 6 (Adding New Labels).

---

## 4. Data Preparation

Create a CSV with two columns: `text` (concatenated product info) and `unspsc_segment` (the label).

```python
import pandas as pd
import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\-\/\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Load client data
df = pd.read_excel('client_inventory.xlsx', dtype=str)

# Concatenate all useful text columns into one
df['text'] = (
    df['Description'].fillna('') + ' ' +
    df['Item Class'].fillna('') + ' ' +
    df['Type'].fillna('')
).apply(clean_text)

# Drop rows with empty text or missing labels
df = df[df['text'].str.split().str.len() >= 3]
df = df[df['unspsc_segment'].notna()]
df = df[['text', 'unspsc_segment']]

# Check distribution
print(df['unspsc_segment'].value_counts())
print(f'\nTotal rows: {len(df)}')

df.to_parquet('data/client_data.parquet', index=False)
```

---

## 5. Fine-Tuning the Models

There are two strategies. **Strategy A** (recommended) combines base training data with client data and retrains. **Strategy B** trains only on client data when you have enough of it (500+ rows per label).

---

### Strategy A — Combined Retrain (Recommended)

Mix the existing 5.5M-row base data with client data and retrain. The model learns everything the base model knows, plus client-specific patterns. Client data is typically small so upsample it to balance influence.

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load base training data and client data
base_df   = pd.read_parquet('data/train.parquet')     # existing 5.5M rows
client_df = pd.read_parquet('data/client_data.parquet')

# Upsample client data so it has meaningful weight against 5.5M base rows.
# Rule of thumb: client data should be ~5-10% of total after upsampling.
UPSAMPLE_FACTOR = 50   # adjust based on how many client rows you have
client_upsampled = pd.concat([client_df] * UPSAMPLE_FACTOR, ignore_index=True)

combined = pd.concat([base_df[['text', 'unspsc_segment']], client_upsampled], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f'Combined rows: {len(combined):,}')
print(combined['unspsc_segment'].value_counts())

# Re-encode labels (in case new labels were added)
le = LabelEncoder()
combined['label'] = le.fit_transform(combined['unspsc_segment'])

train_df, test_df = train_test_split(
    combined, test_size=0.20, stratify=combined['label'], random_state=42
)

train_df.to_parquet('data/train_client.parquet', index=False)
test_df.to_parquet('data/test_client.parquet',   index=False)
joblib.dump(le, 'models/label_encoder_client.pkl')
print('Saved train_client.parquet, test_client.parquet, label_encoder_client.pkl')
```

---

### 5a. Retrain FastText

FastText does not support incremental fine-tuning of an existing `.bin` model for supervised classification. The standard approach is to retrain from scratch on combined data — it's fast enough (minutes on GPU).

```python
import fasttext
import joblib
import pandas as pd
from sklearn.metrics import f1_score, classification_report

PROJ = '/home/tayana-gpu/ML/project_01_product_classifier'

train_df = pd.read_parquet(f'{PROJ}/data/train_client.parquet')
test_df  = pd.read_parquet(f'{PROJ}/data/test_client.parquet')
le       = joblib.load(f'{PROJ}/models/label_encoder_client.pkl')

# Write FastText format files
def write_ft_file(df, path, le):
    with open(path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            label = le.classes_[row['label']].replace(' ', '_')
            f.write(f'__label__{label} {row["text"]}\n')

write_ft_file(train_df, f'{PROJ}/data/train_ft_client.txt', le)
write_ft_file(test_df,  f'{PROJ}/data/test_ft_client.txt',  le)

# Train — same hyperparameters as base model
ft_model = fasttext.train_supervised(
    input=f'{PROJ}/data/train_ft_client.txt',
    lr=0.5, epoch=15, wordNgrams=2,
    dim=100, minCount=2,          # lower minCount for small client vocab
    loss='softmax', thread=8
)

# Evaluate
preds = [ft_model.predict(t)[0][0].replace('__label__', '').replace('_', ' ')
         for t in test_df['text'].tolist()]
true  = [le.classes_[l] for l in test_df['label'].tolist()]

macro = f1_score(true, preds, average='macro')
print(f'FastText Macro F1: {macro:.4f}')
print(classification_report(true, preds))

# Save — overwrites the production model
ft_model.save_model(f'{PROJ}/models/model_fasttext.bin')
joblib.dump(le, f'{PROJ}/models/label_encoder.pkl')
print('Saved model_fasttext.bin and label_encoder.pkl')
```

---

### 5b. Retrain LinearSVC

LinearSVC also retrains from scratch — sklearn does not support warm-start for SVC. Subsample the combined data to 1M rows (same as the base pipeline) to keep training time under 10 minutes.

```python
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

PROJ = '/home/tayana-gpu/ML/project_01_product_classifier'

train_df = pd.read_parquet(f'{PROJ}/data/train_client.parquet')
test_df  = pd.read_parquet(f'{PROJ}/data/test_client.parquet')
le       = joblib.load(f'{PROJ}/models/label_encoder_client.pkl')

# Stratified subsample — SVC doesn't need the full 5.5M rows
SVC_SAMPLE = 1_000_000
if len(train_df) > SVC_SAMPLE:
    svc_train, _ = train_test_split(
        train_df, train_size=SVC_SAMPLE,
        stratify=train_df['label'], random_state=42
    )
else:
    svc_train = train_df

X_train = svc_train['text'].tolist()
y_train = svc_train['label'].tolist()
X_test  = test_df['text'].tolist()
y_test  = test_df['label'].tolist()

word_vec = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2),
    min_df=3, max_features=150_000, sublinear_tf=True   # lower min_df for small client vocab
)
char_vec = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5),
    min_df=3, max_features=100_000, sublinear_tf=True
)

pipeline = Pipeline([
    ('features', FeatureUnion([('word', word_vec), ('char', char_vec)])),
    ('clf', CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=2000, C=1.0),
        cv=3, method='isotonic', n_jobs=3
    ))
])

pipeline.fit(X_train, y_train)
preds  = pipeline.predict(X_test)
macro  = f1_score(y_test, preds, average='macro')
print(f'LinearSVC Macro F1: {macro:.4f}')
print(classification_report(y_test, preds, target_names=le.classes_))

joblib.dump(pipeline, f'{PROJ}/models/model_svc.pkl')
print('Saved model_svc.pkl')
```

---

## 6. Adding a New Label (Not in the Current 15)

If the client's products belong to a UNSPSC segment not in the current 15:

1. Add the new label name to the training data under `unspsc_segment`.
2. Ensure you have **at least 500 rows** for the new label (upsample if needed).
3. Run the combined retrain in Section 5 — the `LabelEncoder` will automatically pick up the new class.
4. The `label_encoder.pkl` and model must be updated together — never mix a new encoder with an old model.

UNSPSC Segment reference: https://www.unspsc.org/

---

## 7. Evaluation Checklist Before Deploying

Run through this before swapping in a new model:

- [ ] Macro F1 on client test set ≥ 0.75 (aim for ≥ 0.80)
- [ ] Per-class F1 for every label the client cares about ≥ 0.70
- [ ] No single label dominates predictions (check classification report — if one label gets 80%+ of rows, something is wrong with class balance)
- [ ] Test manually: paste 5–10 representative product descriptions into the single-product tab and verify the top prediction makes sense
- [ ] Confirm `label_encoder.pkl` and `model_fasttext.bin` were saved in the same training run (they must match)

---

## 8. Deploying the New Model

The Flask app loads models at startup from fixed paths. No config change needed — just replace the files:

```bash
# Back up current production models first
cp models/model_fasttext.bin  models/model_fasttext_base.bin
cp models/label_encoder.pkl   models/label_encoder_base.pkl
cp models/model_svc.pkl       models/model_svc_base.pkl

# Replace with client-tuned models
cp models/model_fasttext_client.bin  models/model_fasttext.bin   # if you renamed it
cp models/label_encoder_client.pkl   models/label_encoder.pkl
cp models/model_svc_client.pkl       models/model_svc.pkl

# Restart the app
# Ctrl+C the running process, then:
conda activate prod_classifier
cd app && python app.py

# Verify
curl http://localhost:5000/health
# → {"status": "ok", "model_ready": true, "classes": <new count>}
```

---

## 9. Quick Decision Tree

```
Client sends labeled data?
├── YES, 50–500 rows per label  → Strategy A (combined retrain), UPSAMPLE_FACTOR = 50
├── YES, 500+ rows per label    → Strategy A (combined retrain), UPSAMPLE_FACTOR = 10
├── YES, 2000+ rows per label   → Strategy B (client-only retrain, drop base data)
└── NO labeled data             → Ask client to label 50–100 rows per label manually
                                  (one afternoon of work for a domain expert)

Which model to deploy?
├── FastText Macro F1 > LinearSVC → deploy FastText (already the default)
└── LinearSVC wins               → update app/predictor.py to load model_svc.pkl instead
```

---

## 10. Files Reference

| File | Purpose |
|---|---|
| `data/client_data.parquet` | Prepared client data (text + label) |
| `data/train_client.parquet` | Combined base + client train split |
| `data/test_client.parquet` | Combined base + client test split |
| `models/label_encoder_client.pkl` | Encoder for client run (rename to `.pkl` to deploy) |
| `models/model_fasttext.bin` | Production FastText model (replace to deploy) |
| `models/model_svc.pkl` | Production LinearSVC model |
| `models/model_fasttext_base.bin` | Backup of pre-client base model |
