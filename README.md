
# 🏠 Real Estate Price Prediction

This project predicts the **house price per unit area** based on various real estate features using a **Linear Regression** model. The application is built with **Dash**, a Python framework for building interactive web apps.

## 🚀 Features

- Interactive web interface to input real estate data:
  - Distance to MRT Station
  - Number of Convenience Stores
  - Latitude and Longitude
- Predicts house price per unit area in real-time
- Clean and responsive UI

## 📂 Dataset

The dataset is sourced from `Real_Estate.csv`, which includes features like:
- Distance to the nearest MRT station
- Number of convenience stores
- Geographic coordinates (latitude & longitude)
- House price per unit area (target)

## 🧠 Machine Learning Model

- Algorithm: **Linear Regression**
- Library: **Scikit-learn**
- Model is trained on 80% of the dataset and tested on 20%

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RealEstate_Price_Prediction.git
   cd RealEstate_Price_Prediction
`

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Run the App

```bash
python main.py
```

Then open your browser and go to:

```
http://127.0.0.1:8050/
```

## 📦 Dependencies

* pandas
* scikit-learn
* dash


