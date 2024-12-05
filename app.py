import torch
from transformers import BertForSequenceClassification, BertTokenizer
from flask import Flask, request, jsonify
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define model and tokenizer loading inside the main block
if __name__ == '__main__':
    # Load pre-trained model and tokenizer
    model_bert = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    
    model_bert.load_state_dict(torch.load('bert_finetuned_model3.pth', map_location=torch.device('cpu')))
    model_bert.eval()

    # Define prediction route
    @app.route('/predict', methods=['POST'])
    def predict():
        # Get input text
        data = request.get_json()
        text = data['text']
        
        # Preprocess and tokenize input
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = model_bert(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return jsonify({'prediction': prediction})

    # Run the app
    app.run(debug=True)
