import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score, f1_score
from sklearn.impute import IterativeImputer, SimpleImputer
import pickle

# 1. Ask the user if he want upload the dataset or not
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv", "xlsx", "tsv"])

# 2. If the user doesn't upload the data then provide a default dataset selection box
if uploaded_file is None:
    dataset_name = st.selectbox('Select a default dataset', ('titanic', 'tips', 'iris'))
    if dataset_name == 'titanic':
        data = sns.load_dataset('titanic')
    elif dataset_name == 'tips':
        data = sns.load_dataset('tips')
    else:
        data = sns.load_dataset('iris')
else:
    data = pd.read_csv(uploaded_file)

# 3. Print the basic information of selected or uploaded dataset
st.write(data.head())
st.write(data.tail())
st.write(data.shape)
st.write(data.describe())
st.write(data.info())
st.write(data.columns)

# 4. Ask the user to select variables as features and target
features = st.multiselect('Select Features', data.columns.tolist())
target = st.selectbox('Select Target', data.columns.tolist())

# 5. Identify the problem type (regression or classification) based on the target variable
problem_type = 'classification'
if data[target].dtype in ['int64', 'float64']:
    problem_type = 'regression'

# 6. Preprocess the data
num_features = data[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = data[features].select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

imputer_num = IterativeImputer()
data[num_features] = imputer_num.fit_transform(data[num_features])

imputer_cat = SimpleImputer(strategy='most_frequent')
data[cat_features] = imputer_cat.fit_transform(data[cat_features])

scaler = StandardScaler()
data[num_features] = scaler.fit_transform(data[num_features])

le_dict = {}
for col in cat_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

# 7. Ask the user to select model from the sidebar
model_name = st.sidebar.selectbox('Select a model', ('Logistic Regression', 'Linear Regression'))

# 8. Split the data into train and test
test_size = st.slider('Test size', 0.1, 0.9, 0.2)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=test_size)

# 9. Fit the model on the training data
if model_name == 'Logistic Regression':
    model = LogisticRegression()
elif model_name == 'Linear Regression':
    model = LinearRegression()

model.fit(X_train, y_train)

# 10. Evaluate the performance of the model
y_pred = model.predict(X_test)
if problem_type == 'regression':
    st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    st.write('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    st.write('R2 Score:', r2_score(y_test, y_pred))
else:
    st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred))
    st.write('Accuracy Score:', accuracy_score(y_test, y_pred))
    st.write('F1 Score:', f1_score(y_test, y_pred))

# 11. Display the performance of each model based on the evaluation matrices
# This is done in the previous step

# 12. Ask the user if he want download the model
if st.button('Download Model As Pickle'):
    pickle.dump(model, open('model.pkl', 'wb'))

# 13. Ask the user if he wants to the prediction
if st.button('Predict'):
    inputs = {}
    for feature in features:
        dtype = str(data[feature].dtype)
        if dtype in ['int64', 'float64']:
            inputs[feature] = st.number_input(f'Enter value for {feature}')
        else:
            options = sorted(list(data[feature].unique()))
            inputs[feature] = st.selectbox(f'Select value for {feature}', options)
    inputs_df = pd.DataFrame([inputs])
    inputs_df[num_features] = imputer_num.transform(inputs_df[num_features])
    inputs_df[cat_features] = imputer_cat.transform(inputs_df[cat_features])
    inputs_df[num_features] = scaler.transform(inputs_df[num_features])
    for col in cat_features:
        inputs_df[col] = le_dict[col].transform(inputs_df[col])
    prediction = model.predict(inputs_df)
    st.write('Prediction:', prediction)

# 14. Show prediction to the user
# This is done in the previous step

# 14. Show prediction to the user
# This is done in the previous step




