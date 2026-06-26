"""
predictor.py — FastText-based product classifier (15 UNSPSC segments).
Instantiated once at startup; used by both single and batch routes.
"""

import joblib
import fasttext
from typing import List, Dict


class ProductClassifier:

    def __init__(self, model_path: str, label_encoder_path: str):
        self.model = fasttext.load_model(model_path)
        self.le    = joblib.load(label_encoder_path)
        self.ready = True

    def _clean(self, text: str) -> str:
        return str(text).strip().replace('\n', ' ').replace('\r', ' ')

    def _decode(self, label: str) -> str:
        return label.replace('__label__', '').replace('_', ' ')

    def predict(self, text: str, top_k: int = 3) -> List[Dict]:
        clean = self._clean(text)
        if not clean:
            return []
        labels, probs = self.model.predict(clean, k=top_k)
        return [
            {"segment": self._decode(lbl), "confidence": round(float(p) * 100, 1)}
            for lbl, p in zip(labels, probs)
        ]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            preds = self.predict(text, top_k=1)
            results.append(preds[0] if preds else {"segment": "", "confidence": 0.0})
        return results
