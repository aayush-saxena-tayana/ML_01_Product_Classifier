"""
app.py — Flask entry point.
Routes: GET /, POST /predict, POST /preview, POST /classify_batch, GET /health
"""

import io
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
from predictor import ProductClassifier

BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / 'models' / 'model_fasttext.bin'
ENCODER_PATH = BASE_DIR / 'models' / 'label_encoder.pkl'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB upload limit

classifier = ProductClassifier(str(MODEL_PATH), str(ENCODER_PATH))

ALLOWED = {'csv', 'xlsx', 'xls'}


def _allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


def _read(file) -> pd.DataFrame:
    name = file.filename.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file, dtype=str)
    return pd.read_excel(file, dtype=str)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    return jsonify({'predictions': classifier.predict(text, top_k=3)})


@app.route('/preview', methods=['POST'])
def preview():
    """Return column names and row count so the UI can build the column picker."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file attached'}), 400
    file = request.files['file']
    if not _allowed(file.filename):
        return jsonify({'error': 'Only CSV, XLSX and XLS files are supported'}), 400
    try:
        df = _read(file)
        return jsonify({
            'columns': df.columns.tolist(),
            'rows':    len(df),
            'preview': df.head(3).fillna('').to_dict(orient='records'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    """Classify every row in the uploaded file and return an enriched CSV."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file attached'}), 400
    file = request.files['file']
    col  = request.form.get('column', '').strip()
    if not _allowed(file.filename):
        return jsonify({'error': 'Only CSV, XLSX and XLS files are supported'}), 400
    try:
        df = _read(file)
        if col not in df.columns:
            return jsonify({'error': f'Column "{col}" not found in file'}), 400

        texts   = df[col].fillna('').astype(str).tolist()
        results = classifier.predict_batch(texts)

        df.insert(df.columns.get_loc(col) + 1, 'UNSPSC_Segment',
                  [r['segment']    for r in results])
        df.insert(df.columns.get_loc('UNSPSC_Segment') + 1, 'UNSPSC_Confidence_%',
                  [r['confidence'] for r in results])

        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(buf, mimetype='text/csv', as_attachment=True,
                         download_name='classified_output.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_ready': classifier.ready,
                    'classes': len(classifier.le.classes_)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
