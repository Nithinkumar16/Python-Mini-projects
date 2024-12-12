import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# Function to load images and labels from a specified folder
def load_images_from_folder(folder):
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
            label = 1 if 'abnormal' in filename else 0  # Adjust based on your dataset
            labels.append(label)
    return np.array(images), np.array(labels)  #preprocessing and training

# Load dataset from a given folder
def load_data(image_folder):
    X, y = load_images_from_folder(image_folder)
    X = X.reshape(-1, 128, 128, 1) / 255.0  # Normalize and reshape
    return X, y

# Define CNN model for classification
def simple_cnn_model(input_size=(128, 128, 1), num_classes=2):
    inputs = layers.Input(input_size)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs) #extract features
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  #helps in over fitting
    x = layers.Flatten()(x)  #data to connect dense layer
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[inputs], outputs=[outputs])  #training
    return model

# Load data from folder specified by the user
def main(image_folder):
    X, y = load_data(image_folder)  #images and labels

    # Compile the model
    model = simple_cnn_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  #adapts learningb rate and multi class
    #minimize loss fuction
    # Train the model
    history = model.fit(X, y, validation_split=0.2, epochs=10, batch_size=8)

    # Plot training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()   #training and validation
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
        for i in range(len(predicted_classes)):
            plt.subplot(1, len(predicted_classes), i + 1)   #rows,columns,position of plot
            plt.imshow(new_images[i].reshape(128, 128), cmap='gray')  #visualize to 2D
            plt.title(f'Pred: {predicted_classes[i]}')
            plt.axis('off')
        plt.show()

    # Load new images for prediction from a specified folder
    new_image_folder = r"C:\Users\HEMANTH KUMAR U\PycharmProjects\Nithin\augumented L1"  # Use raw string for path
    new_images, _ = load_images_from_folder(new_image_folder)
    new_images = new_images.reshape(-1, 128, 128, 1) / 255.0

    # Make predictions
    predict_and_display(model, new_images)

# Main execution point
if __name__ == "__main__":
    image_folder = input("Enter path to the folder containing training images: ")  # User inputs folder path
    main(image_folder)