import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# 1. Create "outputs" folder
# ===============================
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# ===============================
# 2. Load dataset
# ===============================
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Class names
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ===============================
# 3. Build ANN Model
# ===============================
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===============================
# 4. Train Model
# ===============================
history = model.fit(x_train, y_train_cat, 
                    epochs=10, 
                    batch_size=128, 
                    validation_split=0.2, 
                    verbose=1)

# ===============================
# 5. Evaluate
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print("Test Accuracy:", test_acc)

# ===============================
# 6. Save Accuracy & Loss Graphs
# ===============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.savefig("outputs/accuracy.png")   # save automatically
plt.show()
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.savefig("outputs/loss.png")
plt.show()
plt.close()

# ===============================
# 7. Confusion Matrix
# ===============================
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("outputs/confusion_matrix.png")
plt.show()
plt.close()

# ===============================
# 8. Save Sample Predictions
# ===============================
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {class_names[y_pred_classes[i]]}\nTrue: {class_names[y_test[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("outputs/sample_predictions.png")
plt.show()
plt.close()

print("✅ All outputs saved inside 'outputs/' folder")
