import torch
from transformers import BertForSequenceClassification, BertTokenizer
from flask import Flask, request, render_template_string
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained model and tokenizer
model_bert = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

# Load the fine-tuned model weights
model_bert.load_state_dict(torch.load('bert_finetuned_model3.pth', map_location=torch.device('cpu')))
model_bert.eval()

# Define home route with simple inline HTML
@app.route('/')
def home():
    html = """
    <h1>Text Prediction</h1>
    <form action="/predict" method="POST">
        <textarea name="text" placeholder="Enter text here..." rows="4" cols="50"></textarea><br>
        <button type="submit">Predict</button>
    </form>
    """
    return render_template_string(html)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from the form
    text = request.form.get('text', '')

    # Preprocess and tokenize input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        outputs = model_bert(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    html = f"""
    <h1>Text Prediction</h1>
    <form action="/predict" method="POST">
        <textarea name="text" placeholder="Enter text here..." rows="4" cols="50"></textarea><br>
        <button type="submit">Predict</button>
    </form>
    <h3>Prediction: {prediction}</h3>
    """
    return render_template_string(html)

# Main block
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
