from flask import Flask, render_template, request
from transformers import pipeline
import os

app = Flask(__name__)

# Load smaller, more memory-efficient models
sentiment_pipeline = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
generation_pipeline = pipeline('text-generation', model="distilgpt2")
translation_pipeline = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
summarization_pipeline = pipeline('summarization', model="facebook/bart-large-cnn")
NER_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# ... rest of your code remains the same ...

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)