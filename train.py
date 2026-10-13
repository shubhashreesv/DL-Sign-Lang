import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import create_cnn_model

# Configuration
DATA_DIR = 'data/ASL_SemCom'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_SAVE_PATH = 'models/sign_language_model.h5'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

def prepare_data():
    """
    Prepare training and test data generators.
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        validation_split=0.2  # Use 20% for validation
    )
    
    # Test data generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def train_model():
    """
    Train the CNN model and evaluate it.
    """
    print("Preparing data...")
    train_gen, val_gen, test_gen = prepare_data()
    
    num_classes = len(train_gen.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_gen.class_indices}")
    
    # Create model
    print("Creating model...")
    model = create_cnn_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=num_classes
    )
    
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions for confusion matrix
    print("Generating confusion matrix...")
    test_gen.reset()
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    class_names = list(train_gen.class_indices.keys())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    print("Confusion matrix saved to models/confusion_matrix.png")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, 
                                target_names=class_names))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Save class indices
    import json
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)
    print("Class indices saved to models/class_indices.json")
    
    return model, history, test_accuracy

if __name__ == '__main__':
    train_model()