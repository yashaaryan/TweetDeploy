import os
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import nltk

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

app = FastAPI()

# Load BERT model and tokenizer
model_bert = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=2)
model_bert.load_state_dict(torch.load('bert_finetuned_model3.pth', map_location=torch.device('cpu')))
tokenizer_bert = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

stop_words_list = nltk.corpus.stopwords.words('english')

def preprocess_text(input_text):
    cleaned_text = re.sub(r'\W', ' ', str(input_text))
    cleaned_text = cleaned_text.lower()
    cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words_list])
    return cleaned_text

def predict_with_bert(sentence):
    sentence = preprocess_text(sentence)
    inputs = tokenizer_bert(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_bert(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "REAL" if predicted_class == 1 else "FAKE"

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return '''
    <html>
        <head><title>Disaster Tweets</title></head>
        <body>
            <h1>Disaster Tweets</h1>
            <form action="/predict" method="post">
                <input type="text" name="text" placeholder="Enter tweet" maxlength="280" required/>
                <button type="submit" name="model_type" value="bert">Predict BERT</button>
            </form>
        </body>
    </html>
    '''

@app.post('/predict', response_class=HTMLResponse)
async def predict(text: str = Form(...), model_type: str = Form(...)):
    # Ensure prediction is based on BERT
    prediction = predict_with_bert(text)
    
    # Return structured HTML with the prediction
    return f'''
    <html>
        <head><title>Prediction Result</title></head>
        <body>
            <h1>Prediction Result</h1>
            <p>The tweet is classified as: <strong>{prediction}</strong></p>
            <a href="/">Go back</a>
        </body>
    </html>
    '''

# Ensure the app binds to the right port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Get the port from environment variable, default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
