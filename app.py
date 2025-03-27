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

@app.route('/', methods=['GET', 'POST'])
def index():
    sent = None
    gen = None
    trans = None
    ner = None
    summ = None
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        task = request.form.get('task')

        if task == 'sentiment':
            sent = sentiment_pipeline(user_input)
            return render_template('index.html', sent=sent)
        elif task == 'generation':
            gen = generation_pipeline(user_input)
            return render_template('index.html', gen=gen)
        elif task == 'translation':
            trans = translation_pipeline(user_input)
            return render_template('index.html', trans=trans)
        elif task == 'summarization':
            summ = summarization_pipeline(user_input)
            return render_template('index.html', summ=summ)
        elif task == 'named_entity_recognition':
            ner = NER_pipeline(user_input)
            return render_template('index.html', ner=ner)

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)