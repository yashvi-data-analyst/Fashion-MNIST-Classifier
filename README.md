Fashion MNIST Classifier 👕👗👟

This project is a simple Artificial Neural Network (ANN) built using TensorFlow & Keras to classify the Fashion MNIST dataset (10 categories of clothing and accessories).


📌 Project Workflow

1. Load and preprocess the Fashion MNIST dataset

2. Build an ANN model with hidden layers + Dropout

3. Train the model on training data

4. Evaluate on test data

5. Save charts (Accuracy, Loss, Confusion Matrix, Predictions) inside outputs/ folder



📊 Results

Achieved around ~85–90% accuracy on test data

Model learns to distinguish clothing items like T-shirt, Trouser, Dress, Sneaker, etc.



📂 Repository Structure

Fashion-MNIST-Classifier/
│
├── Fashion MNIST classifier.py   # Main Python script
├── outputs/                      # Saved charts & predictions
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png



📷 Sample Outputs

Model Accuracy

Model Loss

Confusion Matrix

Sample Predictions



🛠️ Tech Stack

Python 3

TensorFlow / Keras

NumPy, Matplotlib, Seaborn

Scikit-learn


🚀 How to Run

1. Clone this repo

2. Install dependencies:

pip install tensorflow numpy matplotlib seaborn scikit-learn

3. Run the script:

python "Fashion MNIST classifier.py"

4. All charts will be saved inside the outputs/ folder


✨ This was part of my learning project on Deep Learning with Keras.



From idea to execution, made with ❤️ by Yashvi Verma.
