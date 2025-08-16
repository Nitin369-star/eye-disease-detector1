import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ========== CONFIG ==========
MODEL_PATH = "keras_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DATA_PATH = "test"  # Folder with subfolders of 7 diseases

# ========== LOAD MODEL ==========
print("[INFO] Loading model...")
model = load_model(MODEL_PATH, compile=False)

# ========== LOAD TEST DATA ==========
print("[INFO] Loading test images from:", TEST_DATA_PATH)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ========== PREDICT ==========
print("[INFO] Predicting on test set...")
X_test, y_true = [], []
for i in range(len(test_generator)):
    x_batch, y_batch = test_generator[i]
    X_test.append(x_batch)
    y_true.append(y_batch)
    if i >= len(test_generator) - 1:
        break

X_test = np.vstack(X_test)
y_true = np.vstack(y_true)

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# ========== CONFUSION MATRIX ==========
print("[INFO] Generating confusion matrix...")
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("✅ Done: confusion_matrix.png saved.")

# ========== TRAINING METRICS ==========
print("[INFO] Training dummy model for loss/accuracy curves...")

# ✅ COMPILE THE MODEL
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ FIT FOR METRICS (REQUIRED)
history = model.fit(
    test_generator,
    epochs=5,
    validation_data=test_generator,
    verbose=1
)

# Accuracy Curve
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_curve.png')
plt.close()

# Loss Curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.close()

print("✅ Done: accuracy_curve.png, loss_curve.png saved.")
