Hey, chatgpt act as a excellent  web application developer. I want you to develop an amazing machine learning application using scikit learn and streamlit in python. Please follow these steps  and generate me complete  without any errors.  

1. The title of the application will be Machine learning Application, Display it in center of the page . 
2. Then please ask   the user,  if he want to upload dataset or he want to  deafault dataset.
3. if the user upload the data then proceed with that data, else ask the user to choose one the default dataset. The default dataset must include titanic ,iris, diamond, featch these dataset from seaborn library.
4. then provide the buttons in sidebar to display basic information about the dataset. 
The basic information must include head, tail, info, summary statistic,column names shape and datatype of columns,precentage of missing values in each feature and display the rows with duplicated data.
5. then please encode the categorical or object datatype of columns using label encoder from scikit learn. please different label encoder for each column in order inverse tranform at the end. 
6. then please impute the  features having missing values using iterative imputer from scikit learn.
7. Then please scaled the data using standard scaler from scikit learn library.
8. Then please  aks the user to  select featues and target variable and display him the names of the columns of respective data 
9. if the target featues is of float data type. Then display it is a regression problem else it's classification. 
if the the problem is regression then display a list of regression alogrithm this list must include alogrithm linear regression, decision tree regression, xgb regression, lasso regression ,ridge regression,k neigbhour regressor,random forset regression, gradient boosting regression. else display of list classification algorithm like logistic regression, naive beyes algorithm, xgb classifier, random forest classifier, light classifier catboost classifier, k neibhour classifier,
10. please ask the user to select one of algoritm from respective list. 
11. then please ask the user to select train to test split ration. please show this in sidebar by using slider option. 
12. please take user ratio of train and test and split the data according to that ratio.
13. Then please fit the user selected model on the  training data using fit method
14. then  predict the output for the testing data using predict method.
15. please display the performance of the model using metrices. if problem is regression then use ,mean squared error , mean absolute error ,r2_2 score. else please use confusion matrix,accuracy score, recall score.
15. please display the evluation of the model using these evluation metrices 
16. please provide button to download the model using pickle 
17. Ask the user if he wants make predication using this trained. if yes  then ask him to provide a input featues ans show the predication to user which model has predicated by inverse tranforming the data.
