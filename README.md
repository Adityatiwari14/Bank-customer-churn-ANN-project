#Bank Customer Churn Prediction

## Overview
This project aims to predict customer churn using a deep learning model built with TensorFlow. It includes data preprocessing, model training, and deployment using a Streamlit web application.

## Project Structure
- `app.py` - The Streamlit web application for predicting customer churn.
- `Churn_Modelling.csv` - The dataset containing customer information.
- `experiments.ipynb` - Jupyter Notebook used for data exploration and model experimentation.
- `prediction.ipynb` - Jupyter Notebook for making predictions using the trained model.
- `requirements.txt` - Dependencies required to run the project.

## Dataset Details
The dataset `Churn_Modelling.csv` contains customer information from a bank and includes the following features:
- `RowNumber`: Index of the row.
- `CustomerId`: Unique ID assigned to a customer.
- `Surname`: Last name of the customer.
- `CreditScore`: Credit score of the customer.
- `Geography`: Country of residence.
- `Gender`: Gender of the customer.
- `Age`: Age of the customer.
- `Tenure`: Number of years the customer has been with the bank.
- `Balance`: Account balance of the customer.
- `NumOfProducts`: Number of bank products used by the customer.
- `HasCrCard`: Whether the customer has a credit card (0 or 1).
- `IsActiveMember`: Whether the customer is an active member (0 or 1).
- `EstimatedSalary`: Estimated annual salary of the customer.
- `Exited`: Target variable indicating whether the customer churned (1) or not (0).

## Explanation of Notebooks
### `experiments.ipynb`
This notebook is used for exploratory data analysis and initial model experimentation. The steps include:
1. **Loading the dataset**: Using Pandas to read `Churn_Modelling.csv`.
2. **Data Cleaning**: Handling missing values and inconsistencies.
3. **Feature Engineering**:
   - Encoding categorical variables (`Geography`, `Gender`).
   - Normalizing numerical features (`CreditScore`, `Age`, `Balance`, etc.).
4. **Model Training**:
   - Implementing various machine learning models (Logistic Regression, Decision Trees, Neural Networks) to compare performance.
   - Hyperparameter tuning.
5. **Evaluation**:
   - Analyzing accuracy, precision, recall, and F1-score.
   - Using confusion matrices and ROC curves.

### `prediction.ipynb`
This notebook focuses on using the trained deep learning model to make predictions on new data. The steps include:
1. **Loading the trained model**: Importing the pre-trained TensorFlow model (`model.h5`).
2. **Preprocessing new data**:
   - Encoding categorical variables.
   - Scaling numerical variables using the pre-trained scaler.
3. **Making Predictions**:
   - Feeding new customer data into the model.
   - Obtaining churn probability and interpreting the results.
4. **Post-processing**:
   - Formatting the results for easy analysis.

## Features
- Reads customer data and processes it for prediction.
- Uses a trained neural network to predict the probability of customer churn.
- Deploys the model using Streamlit for easy user interaction.
- Utilizes feature engineering techniques like encoding and scaling.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Input customer details to get a churn prediction.

## Model Details
- The model is a deep learning neural network built with TensorFlow.
- Encodes categorical variables using label encoding and one-hot encoding.
- Scales numerical variables before feeding into the model.
- Outputs the probability of customer churn.

## Contributing
Contributions are welcome! Feel free to submit issues and pull requests to improve the project.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
