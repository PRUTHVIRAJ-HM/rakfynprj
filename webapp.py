
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

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/weather')
def weather():
    return render_template('weather.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    prediction = None
    uploaded_image = None
    confidence = None
    recommendation = None
    next_steps = None
    
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                try:
                    import tensorflow as tf
                    from werkzeug.utils import secure_filename
                    
                    # Save uploaded file temporarily
                    upload_folder = os.path.join('static', 'uploads')
                    os.makedirs(upload_folder, exist_ok=True)
                    
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(upload_folder, filename)
                    file.save(filepath)
                    
                    # Load model and predict
                    model_path = os.path.join('PLANT-DISEASE-IDENTIFICATION', 'trained_plant_disease_model.keras')
                    model = tf.keras.models.load_model(model_path)
                    
                    image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
                    input_arr = tf.keras.preprocessing.image.img_to_array(image)
                    input_arr = np.array([input_arr])
                    predictions = model.predict(input_arr)
                    result_index = np.argmax(predictions)
                    confidence = float(np.max(predictions) * 100)
                    
                    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
                    
                    prediction = class_names[result_index].replace('___', ' - ').replace('_', ' ')
                    uploaded_image = '/' + filepath.replace('\\', '/')
                    
                    # Generate AI recommendations using Ollama
                    try:
                        ollama_url = 'http://localhost:11434/api/generate'
                        
                        # Generate recommendation
                        rec_prompt = f"You are an expert agricultural advisor. A plant has been identified with: {prediction}. Provide a brief, actionable recommendation (2-3 sentences) for treating this condition. Be specific and practical."
                        rec_payload = {
                            "model": "llama3.2:3b",
                            "prompt": rec_prompt,
                            "stream": False
                        }
                        rec_response = requests.post(ollama_url, json=rec_payload, timeout=30)
                        rec_data = rec_response.json()
                        recommendation = rec_data.get('response', 'Continue regular plant care and monitoring.')
                        
                        # Generate next steps
                        steps_prompt = f"You are an expert agricultural advisor. A plant has been identified with: {prediction}. Provide specific next steps (2-3 sentences) for the farmer to take immediately. Focus on practical actions."
                        steps_payload = {
                            "model": "llama3.2:3b",
                            "prompt": steps_prompt,
                            "stream": False
                        }
                        steps_response = requests.post(ollama_url, json=steps_payload, timeout=30)
                        steps_data = steps_response.json()
                        next_steps = steps_data.get('response', 'Monitor the plant closely and document any changes.')
                        
                    except requests.exceptions.ConnectionError:
                        recommendation = "Ollama Server is not running. Using default recommendations."
                        next_steps = "Start Ollama Server to get AI-powered next steps."
                    except Exception as e:
                        recommendation = "Continue regular plant care and monitoring."
                        next_steps = "Monitor the plant closely and consult with local agricultural experts."
                        
                except Exception as e:
                    prediction = f"Error: {str(e)}"
    
    return render_template('disease.html', prediction=prediction, uploaded_image=uploaded_image, 
                         confidence=confidence, recommendation=recommendation, next_steps=next_steps)


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
    system_msg = "You are AgriSens, an expert AI assistant for agriculture and plant care. Provide clear, actionable, and friendly advice for farmers and gardeners, You will not answer questions that are related to other domains or fields under any circumstances "
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
        # Enable markdown tables extension
        reply_html = md.markdown(reply_raw, extensions=['tables'])
    except requests.exceptions.ConnectionError:
        reply_html = "<span style='color:red'>Ollama Server isn't started. Please run it in order to start analytics.</span>"
    except Exception as e:
        reply_html = f"<span style='color:red'>Error: {str(e)}</span>"
    return jsonify({"reply": reply_html})

if __name__ == '__main__':
    app.run(debug=True)
