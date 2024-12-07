
# Model Deployment: Disaster Tweet Prediction

The **Disaster Tweet Prediction** project leverages Natural Language Processing (NLP) and machine learning to classify Twitter messages as either true emergencies or non-emergencies, even when disaster-related language is used. This application processes and analyzes tweet text to accurately distinguish between disaster-related and non-disaster tweets.

## Features

- **Predict Disaster Tweets**: Classifies tweets about real disaster events.
- **Interactive Web Form**: Allows users to input tweet data and get real-time predictions.
- Utilizes **BERT** for accurate tweet classification.

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, JavaScript (Vanilla)
- **Machine Learning Models**:  BERT
- **Deployment**: Compatible with Render, Heroku,AVM or any FastAPI-supporting server.

## Requirements

- Python 3.7+
- Libraries: 
  - FastAPI
  - Pandas
  - Uvicorn (for running the FastAPI server)
  - Scikit-learn
  - Python-multipart
  - Pydantic
  - TensorFlow
  - Starlette
  - NLTK
  - PyTorch
  - Transformers

## Setup Instructions

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**  
   Open your terminal and run:  
   ```bash
   git clone https://github.com/Esteebaan23/LSTM_Bert_Deployment.git
   ```

2. **Create a Virtual Environment**  
   Navigate to the project directory and create a virtual environment:  
   ```bash
   python -m venv myenv
   myenv\Scripts\activate
   ```

3. **Update File Paths**  
   Replace the dataset file paths in `app.py` with the appropriate paths on your system.

4. **Install Dependencies**  
   Install the required libraries using:  
   ```bash
   pip install -r requirements.txt
   ```

   If some dependencies (e.g., `scikit-learn`, `uvicorn`, `python-multipart`) are not installed correctly, install them manually:  
   ```bash
   pip install scikit-learn uvicorn python-multipart
   ```

5. **Run the Application**  
   Start the FastAPI server with:  
   ```bash
   uvicorn app:app --reload
   ```

6. **Access the Application**  
   Open your browser and visit:  
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Notes

- Ensure you have updated the file paths and configurations before running the app.  
- The project supports deployment on platforms like Render and Heroku. Make necessary adjustments in the deployment configurations.

---
