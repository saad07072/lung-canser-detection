🫁 Mini Lung Cancer Detection (Demo Project)

This repository contains a partial demonstration of a Lung Cancer Detection system built with deep learning and a Streamlit web interface.
It is designed as a mini-version for academic presentation (using a small synthetic dataset).
👉 Replace the synthetic dataset with real CT/X-ray images to extend it into a research/clinical project.

✨ Features

Synthetic demo dataset generator (creates small "cancer" vs "normal" image set).

CNN model training script (Keras/TensorFlow).

Streamlit web app for prediction with uploaded chest images.

Grad-CAM visualization to highlight regions the model focuses on.

Evaluation script with confusion matrix & classification report.

📂 Project Structure
lung-cancer-demo/
│
├── create_dataset.py     # Generates synthetic demo dataset
├── train_model.py        # Trains CNN and saves model (mini_model.h5)
├── app.py                # Streamlit app for predictions + Grad-CAM
├── evaluate.py           # Evaluate model performance on dataset
├── mini_dataset/         # (Auto-created) contains 'normal' & 'cancer' folders
└── mini_model.h5         # Saved CNN model after training

⚡ Quickstart
1. Clone repository
git clone https://github.com/YOUR-USERNAME/lung-cancer-demo.git
cd lung-cancer-demo

2. Install dependencies
pip install tensorflow streamlit pillow opencv-python scikit-learn matplotlib

3. Generate dataset & train model
python train_model.py


This will auto-create mini_dataset/ and train a small CNN, saving it as mini_model.h5.

4. Run the demo web app
streamlit run app.py


Upload a PNG/JPG chest image.

The app displays cancer probability and a Grad-CAM heatmap.

📊 Example Output

Prediction: Cancer probability ~0.86 → Predicted: CANCER

Grad-CAM Heatmap: Highlights suspected abnormal regions.

Evaluation: Run python evaluate.py to view accuracy, precision/recall, and confusion matrix.

🛠 Future Improvements

Replace synthetic dataset with real CT/X-ray scans.

Apply transfer learning (e.g., VGG16, ResNet).

Add ROC curve and threshold tuning in the Streamlit app.

Use proper preprocessing for DICOM medical images.

⚠️ Disclaimer

This project is for educational demonstration only.
It is not trained on real patient data and must not be used for clinical diagnosis.
