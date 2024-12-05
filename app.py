from flask import Flask, request, render_template_string
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import string
import nltk

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load BERT model and tokenizer
def load_model():
    model_bert = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=2)
    model_bert.load_state_dict(torch.load('bert_finetuned_model3.pth', map_location=torch.device('cpu')))
    tokenizer_bert = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    return model_bert, tokenizer_bert

model_bert, tokenizer_bert = load_model()

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

@app.route("/", methods=["GET", "POST"])
def main_page():
    if request.method == "POST":
        text = request.form['text']
        prediction = predict_with_bert(text)
        return f'''
        <html>
            <head>
                <title>Disaster Tweets</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        text-align: center;
                    }}
                    input[type="text"] {{
                        padding: 8px;
                        margin: 10px;
                        width: 250px;
                        border: 1px solid #ccc;
                    }}
                    button {{
                        padding: 8px 16px;
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        cursor: pointer;
                    }}
                    button:hover {{
                        background-color: #45a049;
                    }}
                </style>
            </head>
            <body>
                <h1>Disaster Tweets</h1>
                <form action="/" method="post">
                    <input type="text" name="text" placeholder="Enter tweet" maxlength="280" required/>
                    <button type="submit" name="model_type" value="bert">Predict BERT</button>
                </form>
                <h2>Prediction: {prediction}</h2>
            </body>
        </html>
        '''
    return '''
    <html>
        <head>
            <title>Disaster Tweets</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                }}
                input[type="text"] {{
                    padding: 8px;
                    margin: 10px;
                    width: 250px;
                    border: 1px solid #ccc;
                }}
                button {{
                    padding: 8px 16px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #45a049;
                }}
            </style>
        </head>
        <body>
            <h1>Disaster Tweets</h1>
            <form action="/" method="post">
                <input type="text" name="text" placeholder="Enter tweet" maxlength="280" required/>
                <button type="submit" name="model_type" value="bert">Predict BERT</button>
            </form>
        </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True)
