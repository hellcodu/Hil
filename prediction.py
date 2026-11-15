import os
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
MODEL_DIR = 'models'

def load_prediction_model(model_name):
    model_filenames = {
        'cnn': 'brain_tumor_cnn.keras',
        'cnn_lstm': 'brain_tumor_cnn_lstm.keras',
        'mobilenetv2': 'brain_tumor_mobilenetv2.keras'
    }
    
    filename = model_filenames.get(model_name, 'brain_tumor_mobilenetv2.keras')
    model_path = os.path.join(MODEL_DIR, filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, model_name):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # Expand dimensions to create batch of 1: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # MobileNetV2 requires [-1, 1] preprocessing, while CNN and CNN-LSTM use [0, 1] rescaling inside the model
    if model_name == 'mobilenetv2':
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
    return img_array

def predict_tumor(image_path, model_name='mobilenetv2'):
    try:
        model = load_prediction_model(model_name)
        processed_img = preprocess_image(image_path, model_name)
        
        preds = model.predict(processed_img)[0]
        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index]) * 100
        predicted_category = CLASS_NAMES[predicted_index]
        
        # Format class probabilities
        probabilities = {CLASS_NAMES[i]: float(preds[i]) * 100 for i in range(len(CLASS_NAMES))}
        
        return {
            'success': True,
            'prediction': predicted_category,
            'confidence': round(confidence, 2),
            'probabilities': probabilities,
            'model_used': model_name.upper()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }