from flask import Flask, render_template, request, jsonify
import requests
import markdown as md


import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained RandomForest model (RF.pkl)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'CROP-RECOMMENDATION', 'RF.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        crop_model = pickle.load(f)
except Exception as e:
    crop_model = None
    print(f"Error loading crop model: {e}")

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/weather')
def weather():
    return render_template('weather.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')


# Crop recommendation route
@app.route('/crop', methods=['GET', 'POST'])
def crop():
    result = None
    error = None
    if request.method == 'POST':
        try:
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            if crop_model is not None:
                prediction = crop_model.predict(features)
                result = prediction[0]
            else:
                error = "Model not loaded. Please contact admin."
        except Exception as e:
            error = f"Error: {str(e)}"
    return render_template('crop.html', result=result, error=error)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot_api', methods=['POST'])
def chatbot_api():
    user_msg = request.json.get('message', '')
    # Predefined system prompt for all queries
    system_msg = "You are AgriSens, an expert AI assistant for agriculture and plant care. Provide clear, actionable, and friendly advice for farmers and gardeners."
    ollama_url = 'http://localhost:11434/api/generate'
    prompt = f"[INST] {system_msg} [/INST]\n{user_msg}"
    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        data = response.json()
        reply_raw = data.get('response', 'Sorry, no answer.')
        reply_html = md.markdown(reply_raw)
    except Exception as e:
        reply_html = f"<span style='color:red'>Error: {str(e)}</span>"
    return jsonify({"reply": reply_html})

if __name__ == '__main__':
    app.run(debug=True)
