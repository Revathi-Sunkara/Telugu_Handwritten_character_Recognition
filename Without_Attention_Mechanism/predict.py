import os
import numpy as np
print("Loading imports...")
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Note: No custom layers needed for this model
print("✓ All imports loaded")

# ---------------- Configuration ---------------- #
IMG_HEIGHT = 64
IMG_WIDTH = 64
TEST_FOLDER = r'C:\Users\HP\Downloads\New folder (2)\test'

# ---------------- Load Model and Label Encoder ---------------- #
print("\nLoading trained model...")
model = load_model('telugu_model.h5')
print("✓ Model loaded")
print("Loading label encoder...")
label_encoder = joblib.load('label_encoder.pkl')
print("✓ Label encoder loaded")

# ---------------- Image Preprocessing ---------------- #
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0
    img = img[..., np.newaxis]
    return np.expand_dims(img, axis=0)

# ---------------- OCR Prediction ---------------- #
print("\nStarting predictions on test images...")
correct = 0
total = 0
y_true = []
y_pred = []
y_scores = []

test_files = [f for f in os.listdir(TEST_FOLDER) if f.endswith('.png')]
print(f"Found {len(test_files)} test images")

for idx, filename in enumerate(test_files, 1):
    total += 1
    file_path = os.path.join(TEST_FOLDER, filename)
    true_label = filename.split('_')[0]

    img = preprocess_image(file_path)
    prediction = model.predict(img, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    if predicted_label == true_label:
        correct += 1

    print(f"{filename} --> Predicted: {predicted_label} | Actual: {true_label}")

    y_true.append(true_label)
    y_pred.append(predicted_label)
    y_scores.append(prediction)

# ---------------- Accuracy ---------------- #
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n✅ Prediction Accuracy: {accuracy:.3f}% ({correct}/{total} correct)")

# ---------------- Confusion Matrix ---------------- #
labels = sorted(list(set(y_true)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✓ Confusion matrix saved")
plt.close()

# ---------------- Classification Report ---------------- #
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels, digits=3))

# ---------------- ROC Curve ---------------- #
print("Generating ROC curve...")
try:
    y_true_num = label_encoder.transform(y_true)
    y_true_bin = label_binarize(y_true_num, classes=range(len(label_encoder.classes_)))
    y_score = np.array(y_scores)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(label_encoder.classes_)

    plt.figure(figsize=(12, 10))
    colors = cycle(plt.cm.tab20.colors)

    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        class_name = label_encoder.classes_[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="best", fontsize='x-small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_Attention.png")
    print("✓ ROC curve saved")
    plt.close()

except Exception as e:
    print("⚠ ROC curve generation failed:", e)

# ---------------- Accuracy & Loss Plot with Test Metrics ---------------- #
print("Generating accuracy and loss plots...")
try:
    # Load training history
    history = joblib.load('training_history.pkl')

    # Load test data to evaluate final test accuracy/loss
    data = np.load("processed_data.npz")
    x_test = data['x_test']
    y_test = data['y_test']
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"\n✅ Final Test Accuracy: {test_accuracy * 100:.3f}%")
    print(f"📉 Final Test Loss: {test_loss:.3f}")

    # -------- Accuracy Plot --------
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='cyan')
    plt.axhline(y=test_accuracy, color='red', linestyle='--', label='Test Accuracy')
    plt.title('Train/Val/Test Accuracy_with_attention')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot_Attention.png")
    print("✓ Accuracy plot saved")
    plt.close()

    # -------- Loss Plot --------
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', color='orange')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.axhline(y=test_loss, color='purple', linestyle='--', label='Test Loss')
    plt.title('Train/Val/Test Loss_with_attention')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot_Attention.png")
    print("✓ Loss plot saved")
    plt.close()
    print("\n✅ All predictions and visualizations complete!")

except Exception as e:
    print("⚠ Could not compute or plot test accuracy/loss:", e)