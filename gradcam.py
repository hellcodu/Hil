import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

RESULTS_DIR = os.path.join('static', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_last_conv_layer_name(model, model_name):
    # Automatically identify the last convolutional layer based on architecture
    if model_name == 'mobilenetv2':
        return 'Conv_1' # standard last conv layer name in MobileNetV2
    
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
            
    # Fallback for models without direct Conv2D (e.g. CNN-LSTM)
    return None

def generate_gradcam(image_path, model, model_name, pred_index=None):
    try:
        img_size = (224, 224)
        original_img = Image.open(image_path).convert('RGB').resize(img_size)
        img_array = np.array(original_img)
        
        # Preprocess input depending on model
        processed_input = np.expand_dims(img_array, axis=0)
        if model_name == 'mobilenetv2':
            processed_input = tf.keras.applications.mobilenet_v2.preprocess_input(processed_input)
        else:
            processed_input = processed_input / 255.0

        last_conv_name = get_last_conv_layer_name(model, model_name)
        
        # If standard conv layer is missing (e.g. CNN-LSTM internal reshape), use fallback visualization
        if not last_conv_name and model_name == 'cnn_lstm':
            return generate_fallback_heatmap(original_img, img_array, image_path)

        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_input)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Gradient calculation
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Superimpose heatmap on original image
        heatmap_colored = plt.jet(heatmap)[:, :, :3]
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).resize(img_size)
        
        superimposed = Image.blend(original_img.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.4)
        
        base_name = os.path.basename(image_path)
        result_filename = f"gradcam_{base_name}"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        superimposed.convert('RGB').save(result_path)
        
        return os.path.join('static', 'results', result_filename).replace('\\', '/')

    except Exception as e:
        print(f"[WARNING] Grad-CAM generation failed: {e}. Using fallback.")
        return generate_fallback_heatmap(original_img, img_array, image_path)

def generate_fallback_heatmap(original_img, img_array, image_path):
    # Fallback visualization for architectures without standard spatial conv maps
    gray = np.mean(img_array, axis=2)
    norm_gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(original_img)
    plt.imshow(norm_gray, cmap='jet', alpha=0.3)
    plt.axis('off')
    
    base_name = os.path.basename(image_path)
    result_filename = f"gradcam_fallback_{base_name}"
    result_path = os.path.join(RESULTS_DIR, result_filename)
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return os.path.join('static', 'results', result_filename).replace('\\', '/')