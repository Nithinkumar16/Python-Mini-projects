import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Function to load images and labels from a specified folder
def load_images_from_folder(folder, introduce_noise=False, noise_level=0.1):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if not os.path.isfile(img_path):
            print(f"File {img_path} does not exist or is not a valid file.")
            continue  # Skip invalid files

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to match the model input
            images.append(img)

            # Example label extraction (modify according to your naming convention)
            label = 1 if 'abnormal' in filename.lower() else 0  # Case-insensitive check

            # Introduce label noise
            if introduce_noise:
                if np.random.rand() < noise_level:
                    label = 1 - label  # Flip label
            labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset from a given folder
def load_data(image_folder, introduce_noise=False, noise_level=0.1):
    X, y = load_images_from_folder(image_folder, introduce_noise, noise_level)
    X = X.reshape(-1, 128, 128, 1) / 255.0  # Normalize and reshape
    return X, y

# Define an enhanced CNN model for classification with more layers
def enhanced_cnn_model(input_size=(128, 128, 1), num_classes=2):
    inputs = layers.Input(input_size)

    # First Convolutional Block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Second Convolutional Block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Third Convolutional Block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Flatten and Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Load data from folder specified by the user
def main(image_folder):
    # Introduce label noise by setting introduce_noise=True and adjusting noise_level as needed
    X, y = load_data(image_folder, introduce_noise=True, noise_level=0.1)  # 10% label noise

    # Compile the model with an adjusted learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjusted learning rate
    model = enhanced_cnn_model()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

    # Train the model with Early Stopping and Learning Rate Reduction
    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=10,  # Increased epochs to allow learning
        batch_size=8,  # Adjusted batch size
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate the model
    y_pred = np.argmax(model.predict(X), axis=1)  # Get predicted class labels
    accuracy = np.mean(y_pred == y)  # Calculate accuracy manually

    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Calculate F1 score
    f1 = f1_score(y, y_pred, average='binary')  # 'binary' since there are 2 classes
    print(f'F1 Score: {f1:.2f}')

    # Display classification report (optional, for precision, recall, etc.)
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Example of making predictions on new images
    def predict_and_display(model, new_images):
        predictions = model.predict(new_images)
        predicted_classes = np.argmax(predictions, axis=1)
        num_images = len(predicted_classes)

        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)  # rows, columns, position of plot
            plt.imshow(new_images[i].reshape(128, 128), cmap='gray')  # visualize to 2D
            plt.title(f'Pred: {predicted_classes[i]}')
            plt.axis('off')
        plt.show()

    # Load new images for prediction from a specified folder
    new_image_folder = r"C:\Users\HEMANTH KUMAR U\PycharmProjects\Nithin\augmented_L1"  # Ensure correct path
    new_images, _ = load_images_from_folder(new_image_folder)
    if len(new_images) == 0:
        print("No images found in the new_image_folder for prediction.")
        return
    new_images = new_images.reshape(-1, 128, 128, 1) / 255.0

    # Make predictions
    predict_and_display(model, new_images)

# Main execution point
if __name__ == "__main__":
    image_folder = input("Enter path to the folder containing training images: ")  # User inputs folder path
    if not os.path.isdir(image_folder):
        print(f"The provided path '{image_folder}' is not a valid directory.")
    else:
        main(image_folder)
