import os
import pickle
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Configuration constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_TEST = os.path.join('dataset', 'Testing')
MODEL_DIR = 'models'
RESULTS_DIR = 'training_results'
STATIC_RESULTS = os.path.join('static', 'results')

os.makedirs(STATIC_RESULTS, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_test_dataset():
    print("[INFO] Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_TEST,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    class_names = test_ds.class_names
    
    # Extract all true labels and images for metric calculations
    images = []
    labels = []
    for img_batch, lbl_batch in test_ds:
        images.append(img_batch.numpy())
        labels.append(lbl_batch.numpy())
        
    X_test = np.vstack(images)
    y_test = np.concatenate(labels)
    
    return X_test, y_test, class_names

def plot_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Annotate values
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(STATIC_RESULTS, f'confusion_matrix_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

if __name__ == '__main__':
    X_test, y_test, class_names = load_test_dataset()
    
    models_to_evaluate = {
        'cnn': os.path.join(MODEL_DIR, 'brain_tumor_cnn.keras'),
        'cnn_lstm': os.path.join(MODEL_DIR, 'brain_tumor_cnn_lstm.keras'),
        'mobilenetv2': os.path.join(MODEL_DIR, 'brain_tumor_mobilenetv2.keras')
    }
    
    evaluation_results = {}
    
    for name, path in models_to_evaluate.items():
        if not os.path.exists(path):
            print(f"[WARNING] Model file {path} not found. Skipping...")
            continue
            
        print(f"\n[INFO] Evaluating {name.upper()} model...")
        model = tf.keras.models.load_model(path)
        
        # Get model file size in MB
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Measure inference time
        start_time = time.time()
        preds = model.predict(X_test)
        inference_time = time.time() - start_time
        
        y_pred = np.argmax(preds, axis=1)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Save confusion matrix plot
        cm_path = plot_confusion_matrix(cm, class_names, name)
        
        evaluation_results[name] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'model_size_mb': float(file_size_mb),
            'inference_time_sec': float(inference_time),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_path': cm_path
        }
        
        print(f" -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | Size: {file_size_mb:.2f} MB")

    # Save evaluation metrics to results directory
    with open(os.path.join(RESULTS_DIR, 'evaluation_metrics.pkl'), 'wb') as f:
        pickle.dump(evaluation_results, f)
        
    print("\n[SUCCESS] Model evaluation complete! Results saved successfully.")