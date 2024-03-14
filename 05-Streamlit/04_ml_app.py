import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from xgboost import XGBRegressor, XGBClassifier
# from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score, recall_score
import pickle

# Set page title
st.title("Machine Learning Application")
st.markdown("<h1 style='text-align: center;'>Machine Learning Application</h1>", unsafe_allow_html=True)

# Ask user for dataset upload or default dataset
dataset_option = st.radio("Choose dataset option:", ("Upload Dataset", "Default Dataset"))

# Load default dataset if selected
if dataset_option == "Default Dataset":
    default_dataset = st.selectbox("Choose a default dataset:", ("Titanic", "Iris", "Diamond"))
    if default_dataset == "Titanic":
        data = sns.load_dataset("titanic")
    elif default_dataset == "Iris":
        data = sns.load_dataset("iris")
    elif default_dataset == "Diamond":
        data = sns.load_dataset("diamonds")
else:
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.stop()

# Sidebar options
show_head = st.sidebar.button("Show Head")
show_tail = st.sidebar.button("Show Tail")
show_info = st.sidebar.button("Show Info")
show_summary = st.sidebar.button("Show Summary Statistics")
show_columns = st.sidebar.button("Show Columns")
show_missing_percentage = st.sidebar.button("Missing Percentage")
show_duplicates = st.sidebar.button("Show Duplicated Rows")

if show_head:
    st.write("## Head")
    st.write(data.head())

if show_tail:
    st.write("## Tail")
    st.write(data.tail())

if show_info:
    st.write("## Info")
    st.write(data.info)

if show_summary:
    st.write("## Summary Statistics")
    st.write(data.describe())

if show_columns:
    st.write("## Columns")
    st.write(data.columns)

if show_missing_percentage:
    st.write("## Missing Percentage")
    missing_percentage = (data.isnull().mean() * 100).round(2)
    st.write(missing_percentage)

if show_duplicates:
    st.write("## Duplicated Rows")
    duplicated_rows = data[data.duplicated()]
    st.write(duplicated_rows)

# Encode categorical/object columns
object_columns = data.select_dtypes(include=['object']).columns
for col in object_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Impute missing values
imputer = IterativeImputer()
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Scale data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)

# Select features and target variable
st.write("## Select Features and Target Variable")
feature_cols = st.multiselect("Select Features", data.columns)
target_col = st.selectbox("Select Target Variable", data.columns)

# Check problem type (Regression/Classification)
if data[target_col].dtype == 'float64':
    problem_type = "Regression"
    algorithms = ["Linear Regression", "Decision Tree Regression", "XGB Regression", "Lasso Regression",
                  "Ridge Regression", "K Neighbors Regressor", "Random Forest Regression", "Gradient Boosting Regression"]
else:
    problem_type = "Classification"
    algorithms = ["Logistic Regression", "Naive Bayes", "XGB Classifier", "Random Forest Classifier",
                  "LightGBM Classifier", "CatBoost Classifier", "K Neighbors Classifier"]

st.write(f"Problem Type: {problem_type}")

# Select algorithm
selected_algorithm = st.selectbox("Select Algorithm", algorithms)

# Split data into train and test sets
test_size = st.slider("Select Train-Test Split Ratio", 0.1, 0.9, 0.2)
X_train, X_test, y_train, y_test = train_test_split(data_scaled[feature_cols], data_scaled[target_col], test_size=test_size, random_state=42)

# Initialize and train model
if "Regression" in problem_type:
    if selected_algorithm == "Linear Regression":
        model = LinearRegression()
    elif selected_algorithm == "Decision Tree Regression":
        model = DecisionTreeRegressor()
    # elif selected_algorithm == "XGB Regression":
    #     model = XGBRegressor()
    # elif selected_algorithm == "Lasso Regression":
    #     model = Lasso()
    # elif selected_algorithm == "Ridge Regression":
    #     model = Ridge()
    elif selected_algorithm == "K Neighbors Regressor":
        model = KNeighborsRegressor()
    elif selected_algorithm == "Random Forest Regression":
        model = RandomForestRegressor()
    elif selected_algorithm == "Gradient Boosting Regression":
        model = GradientBoostingRegressor()
else:
    if selected_algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif selected_algorithm == "Naive Bayes":
        model = GaussianNB()
    # elif selected_algorithm == "XGB Classifier":
    #     model = XGBClassifier()
    elif selected_algorithm == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif selected_algorithm == "LightGBM Classifier":
        model = GradientBoostingClassifier()
    # elif selected_algorithm == "CatBoost Classifier":
    #     model = CatBoostClassifier()
    elif selected_algorithm == "K Neighbors Classifier":
        model = KNeighborsClassifier()

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model evaluation
st.write("## Model Evaluation")
if "Regression" in problem_type:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"R2 Score: {r2}")
else:
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
    st.write(f"Accuracy Score: {accuracy}")
    st.write(f"Recall Score: {recall}")

# Option to download model
if st.button("Download Model"):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    st.success("Model downloaded successfully!")

# Make prediction
if st.button("Make Prediction"):
    input_features = []
    for col in feature_cols:
        val = st.number_input(f"Enter value for {col}", step=0.)
        input_features.append(val)
    input_data = pd.DataFrame([input_features], columns=feature_cols)
    prediction = model.predict(input_data)
    st.write("Prediction:")
    st.write(prediction)



