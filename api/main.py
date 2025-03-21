from flask import Flask, render_template, request
from transformers import pipeline
from vercel_edge import EdgeFunction

app = Flask(__name__)
app = EdgeFunction(__name__)

# Load only essential pipelines with smaller models
sentiment_pipeline = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

@app.edge_function()
async def handle(request):
    if request.method == 'POST':
        data = await request.json()
        user_input = data.get('user_input')
        result = sentiment_pipeline(user_input)
        return {'result': result}
    
    return {'message': 'Send a POST request with text to analyze'}

# Load NLP pipelines
sentiment_pipeline = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
generation_pipeline = pipeline('text-generation')
translation_pipeline = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
summarization_pipeline = pipeline('summarization')
NER_pipeline = pipeline("ner", grouped_entities=True)

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