import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso ,Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    confusion_matrix, accuracy_score, recall_score
import seaborn as sns
import pickle

# Set page title
st.title("Machine Learning Application")
st.markdown("<h1 style='text-align: center;'>Machine Learning Application</h1>", unsafe_allow_html=True)

# Ask user to upload or use default dataset
upload_option = st.radio("Choose an option:", ("Upload Dataset", "Use Default Dataset"))

# Load dataset
if upload_option == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    dataset_option = st.sidebar.selectbox("Choose a default dataset:", ("Titanic", "Iris", "Diamond"))

    if dataset_option == "Titanic":
        data = sns.load_dataset("titanic")
    elif dataset_option == "Iris":
        data = sns.load_dataset("iris")
    elif dataset_option == "Diamond":
        data = sns.load_dataset("diamonds")

# Display basic information about the dataset
if st.sidebar.button("Head"):
    st.write(data.head())

if st.sidebar.button("Tail"):
    st.write(data.tail())

if st.sidebar.button("Info"):
    st.write(data.info())

if st.sidebar.button("Summary Statistics"):
    st.write(data.describe())

if st.sidebar.button("Column Names"):
    st.write(data.columns)

if st.sidebar.button("Shape"):
    st.write(data.shape)

if st.sidebar.button("Data Types"):
    st.write(data.dtypes)

if st.sidebar.button("Missing Values"):
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_data = pd.concat([missing_values, missing_percentage], axis=1, keys=['Missing Values', 'Percentage'])
    st.write(missing_data)

if st.sidebar.button("Duplicates"):
    duplicate_rows = data[data.duplicated()]
    st.write(duplicate_rows)

# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include=['object','category']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Impute missing values
# ...

# Impute missing values
imputer = IterativeImputer()   
input_data = pd.DataFrame(imputer.transform(data.astype(data.dtypes
    [selected_features].to_dict())), columns=data.columns)

# ...

# Scale data
scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Select features and target variables
selected_features = st.multiselect("Select Features", data.columns)
selected_target = st.selectbox("Select Target Variable", data.columns)

# Determine problem type
if data[selected_target].dtype == np.float64:
    problem_type = "Regression"
else:
    problem_type = "Classification"

# Select algorithm
if problem_type == "Regression":
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "Lasso Regression": Lasso(),
        "Ridge Regression": Ridge(),
        "K-Nearest Neighbors Regression": KNeighborsRegressor(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor()
    }
else:
    algorithms = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "Random Forest Classifier": RandomForestClassifier(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier()
    }

selected_algorithm = st.selectbox("Select Algorithm", list(algorithms.keys()))

# Train-test split ratio
train_test_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, 0.7, 0.1)

# Split data into training and testing sets
X = data[selected_features]
y = data[selected_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_test_ratio, random_state=42)

# Fit the model
model = algorithms[selected_algorithm]
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Display model performance```python
if problem_type == "Regression":
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))
else:
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
    st.write("Recall Score:", recall_score(y_test, y_pred))

# Download the model
if st.button("Download Model"):
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    st.success("Model downloaded successfully!")

# Prediction
prediction_option = st.sidebar.checkbox("Make a Prediction")

if prediction_option:
    input_features = {}
    for feature in selected_features:
        value = st.sidebar.text_input(f"Enter {feature}")
        input_features[feature] = value

    input_data = pd.DataFrame(input_features, index=[0])
    input_data = input_data.astype(data.dtypes[selected_features].to_dict())

    # Inverse transform categorical columns
    for column in input_data.select_dtypes(include=['object']):
        input_data[column] = label_encoders[column].inverse_transform(input_data[column])

    # Impute missing values
    input_data = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)

    # Scale input data
    input_data[input_data.columns] = scaler.transform(input_data[input_data.columns])

    # Predict using the model
    prediction = model.predict(input_data)

    # Inverse transform target variable for classification
    if problem_type == "Classification":
        prediction = label_encoders[selected_target].inverse_transform(prediction)

    st.write("Prediction:", prediction)