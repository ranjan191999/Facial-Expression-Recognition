# stage2/sentiment.py
"""
Lightweight sentiment wrapper using HuggingFace transformers pipeline.
Usage:
    from stage2.sentiment import get_sentiment
    get_sentiment("I love this!")
"""
from typing import Dict
try:
    from transformers import pipeline
except Exception as e:
    raise ImportError("Install transformers: pip install transformers") from e

# Create a singleton pipeline (loads model on first import)
_sentiment_pipe = pipeline("sentiment-analysis", truncation=True)

def get_sentiment(text: str) -> Dict[str, float]:
    """
    Returns a dict: {"label": "POSITIVE"/"NEGATIVE", "score": 0.99}
    """
    if not text:
        return {"label": "NEUTRAL", "score": 0.0}
    out = _sentiment_pipe(text[:512])[0]
    return {"label": out.get("label", ""), "score": float(out.get("score", 0.0))}

if __name__ == "__main__":
    print(get_sentiment("I absolutely love this product!"))
    print(get_sentiment("This was the worst experience ever."))