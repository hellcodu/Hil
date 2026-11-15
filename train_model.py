import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Optimized for Intel i3 CPU
EPOCHS = 12      # Balanced epochs for training time vs accuracy
DATASET_TRAIN = os.path.join('dataset', 'Training')
MODEL_DIR = 'models'
RESULTS_DIR = 'training_results'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_datasets():
    print("[INFO] Loading training and validation datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_TRAIN,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_TRAIN,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"[INFO] Classes detected: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

# --- MODEL 1: Custom CNN ---
def build_cnn(num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- MODEL 2: CNN-LSTM (RNN) ---
# Explanation: Converts image spatial feature maps into a sequence of feature vectors 
# along height/width strips before feeding them into LSTM layers for sequential dependency modeling.
def build_cnn_lstm(num_classes):
    input_tensor = layers.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1./255)(input_tensor)
    
    # CNN Feature Extractor backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x) # Shape: (batch, 56, 56, 64)
    
    # Reshape feature maps into sequential timesteps for LSTM
    # We treat height dimension (56) as timesteps and features (56 * 64) as feature vector per timestep
    shape = tf.keras.backend.int_shape(x)
    x = layers.Reshape(target_shape=(shape[1], shape[2] * shape[3]))(x)
    
    # LSTM Layers
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.5)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- MODEL 3: MobileNetV2 (Transfer Learning) ---
def build_mobilenetv2(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base initially for CPU efficiency
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    # MobileNetV2 specific preprocessing: scales pixels from [0, 255] to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    train_ds, val_ds, class_names = load_datasets()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    # Train CNN
    print("\n--- Training Model 1: Custom CNN ---")
    cnn = build_cnn(len(class_names))
    hist_cnn = cnn.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    cnn.save(os.path.join(MODEL_DIR, 'brain_tumor_cnn.keras'))
    
    # Train CNN-LSTM
    print("\n--- Training Model 2: CNN-LSTM ---")
    cnn_lstm = build_cnn_lstm(len(class_names))
    hist_lstm = cnn_lstm.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    cnn_lstm.save(os.path.join(MODEL_DIR, 'brain_tumor_cnn_lstm.keras'))
    
    # Train MobileNetV2
    print("\n--- Training Model 3: MobileNetV2 ---")
    mobilenet = build_mobilenetv2(len(class_names))
    hist_mobilenet = mobilenet.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    mobilenet.save(os.path.join(MODEL_DIR, 'brain_tumor_mobilenetv2.keras'))
    
    # Save histories for evaluation dashboard later
    histories = {
        'cnn': hist_cnn.history,
        'cnn_lstm': hist_lstm.history,
        'mobilenetv2': hist_mobilenet.history,
        'class_names': class_names
    }
    with open(os.path.join(RESULTS_DIR, 'training_histories.pkl'), 'wb') as f:
        pickle.dump(histories, f)
        
    print("\n[SUCCESS] All three models trained and saved successfully!")