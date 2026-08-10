import os
print("Loading imports...")
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
print("Loading TensorFlow (this may take a moment)...")
from tensorflow.keras.utils import to_categorical
import joblib  # <-- import joblib
print("✓ All imports loaded")

IMG_HEIGHT = 64
IMG_WIDTH = 64

def load_data(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            label = filename.split('_')[0]
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')  # grayscale
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(train_path, test_path):
    print("Loading training data...")
    x_train, y_train = load_data(train_path)
    print(f"Loaded {len(x_train)} training images")
    
    print("Loading test data...")
    x_test, y_test = load_data(test_path)
    print(f"Loaded {len(x_test)} test images")

    # Add channel dimension
    print("Adding channel dimension...")
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    y_train_enc = to_categorical(le.fit_transform(y_train))
    y_test_enc = to_categorical(le.transform(y_test))

    # Save label encoder for prediction use
    print("Saving label encoder...")
    joblib.dump(le, "label_encoder.pkl")

    return x_train, y_train_enc, x_test, y_test_enc

# Example usage
if __name__ == "__main__":
    train = r'C:\Users\HP\Downloads\New folder (2)\train'
    test = r'C:\Users\HP\Downloads\New folder (2)\test'
    x_train, y_train, x_test, y_test = preprocess_data(train, test)
    print("Saving processed data...")
    np.savez("processed_data.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("✓ Preprocessing complete! Files saved: processed_data.npz and label_encoder.pkl")