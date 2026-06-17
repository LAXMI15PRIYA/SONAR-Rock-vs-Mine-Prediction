# SONAR Rock vs Mine Prediction

This project uses **Logistic Regression** to classify underwater objects as **Rock** or **Mine** based on SONAR signal data. It's a classic binary classification problem inspired by real-world sonar/radar applications used in naval and underwater exploration systems.

## 📌 Problem Statement

Submarines and underwater vehicles use SONAR (Sound Navigation and Ranging) signals to detect objects underwater. The challenge is to distinguish between rocks and metal mines based on the way sound waves bounce off them. This project builds a Machine Learning model that automates this classification using historical SONAR signal readings.

## 📂 Dataset

- **Source:** SONAR dataset (Connectionist Bench - Sonar, Mines vs Rocks)
- **Samples:** 208 SONAR readings
- **Features:** 60 numerical attributes representing energy levels of sonar signals at different frequencies
- **Label:** `R` (Rock) or `M` (Mine)

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Google Colab

## ⚙️ Project Workflow

1. **Data Loading** – Loaded the SONAR dataset into a Pandas DataFrame
2. **Data Exploration** – Checked shape, statistical summary, and class distribution (Rock vs Mine counts)
3. **Data Preprocessing** – Separated features (X) and labels (Y)
4. **Train-Test Split** – Split data into training (90%) and test (10%) sets using stratified sampling to maintain class balance
5. **Model Training** – Trained a Logistic Regression model on the training data
6. **Model Evaluation** – Measured accuracy on both training and test data
7. **Predictive System** – Built a system that takes new SONAR readings and predicts whether the object is a Rock or a Mine

## 📊 Results

| Metric | Score |
|---|---|
| Training Accuracy | **83.4%** |
| Test Accuracy | **76.2%** |

**Train/Test Split:** 90% train (187 samples) / 10% test (21 samples), stratified

## 🔍 Sample Prediction

Given a new SONAR reading, the model predicts whether the object is a Rock or a Mine in real time:

```python
prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a Mine')
```

## 🚀 How to Run

1. Clone this repository
```bash
   git clone https://github.com/LAXMI15PRIYA/SONAR-Rock-vs-Mine-Prediction.git
```
2. Open `Rock_vs_Mine_Prediction.ipynb` in Google Colab or Jupyter Notebook
3. Upload `Copy of sonar data.csv` to the same environment (or update the file path in the code)
4. Run all cells in order

## 🔧 Future Improvements

- Apply feature scaling (StandardScaler) to potentially improve model performance
- Experiment with other classifiers (Random Forest, SVM, KNN) for comparison
- Add cross-validation for a more robust accuracy estimate
- Build a simple web interface (e.g., using Flask or Streamlit) for live predictions

## 👩‍💻 Author

**Lakshmi Priya V**
M.Tech – Artificial Intelligence & Data Science
SRM Institute of Science and Technology (SRMIST)
