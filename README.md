# Titanic Survival Prediction Web App

This is an end-to-end machine learning project that predicts the survival probability of passengers on the Titanic.  
The project combines data science and full stack development to demonstrate building, deploying, and interacting with a trained ML model.

It includes:

- A Jupyter notebook for exploratory data analysis (EDA) and model experimentation.
- A Python pipeline that performs feature engineering and trains an XGBoost classifier (Accuracy: 82%).
- A Flask API backend that serves the trained model for real-time predictions.
- A React.js frontend that allows users to input passenger details and receive survival predictions.

---

## How to run the project

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Titanic-EDA-Classification.git
cd Titanic-EDA-Classification
```

### 2. Set up a virtual environment and install dependencies
(Recommended to avoid dependency conflicts)
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build the machine learning pipeline
This step loads the Titanic dataset, processes the data, trains the model, and saves the pipeline.
```bash
make run-pipeline
```

### 4. Run the backend API
This starts the Flask server.
```bash
make run-backend
```

### 5. Run the React frontend
In a separate terminal, start the React application.
```bash
make run-frontend
```

### 6. Use the web app
Open your browser and go to:
```arduino
http://localhost:3000
```


