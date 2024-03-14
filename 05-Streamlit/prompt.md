# Prompt

Hey Chat GPt, act as an application developer expert, and help me with the following problem in python
using streamlit, and build an machine learning application by using scikit learn with the following workflow.
please avoid all the error and provide me the code in working from without any error in low code method and please include all the funtionality which is methioned below.

1. Ask the user if he want upload the  dataset or not. If yes then ask for a file to be uploaded in csv, xlsx,tsv or any other possible data format
if user does not upload dataset then used default dataset which user select from dropdown list.
otherwise user provided dataset for futher processing
2. If the user doesn't upload the data then provide a deafault dataset selection box. this selection box should download data from sns. this selection box should include dataset like titanic ,tips or iris.
3. print the basic information of selected or uploaded dataset such as data head, data tial,data shape, data description,and data info and coloumn.
4. Ask the user the select variables as features and target. 
5. Identify the problem  type (regression or classification) based on the target variable selected by the user.
or if the target variable is contionous data type or fload type suggest to use regression else  classification.
6. preprocess the data by following steps. impute missing values by using iterative imputer from scikit learn. first this function. 
if the features aren't in the same scale then use standard scaler from scikit learn to scale them.
use label encoder to encode the categorical features from scikit learn. please make sure to use different label encoder for different variable in order inverse transform in the end.

7. Ask the user to select model from the sidebar. the model list should include all the regression and classification models. 
8. Ask the user to split the data into train and test  provide a line to set the value of train and test split. 
if user doesn't give train and test split use default 20 to 80 ration. 20 for test 80 for training.
9. Fit the model on the training data. And evulavate the perform of the model testing data.
10. if the problem is regression then use mean square error, mean absolute error, r2_score as an evluation metrics or if the  problem is classification use confusion matrix, accuracy score, f1_score. 
11. Display the performance of each model based on the evluation matrices. 
12. Ask the user if he want download the model,if yes then download the model in pickle format. 
13. Ask the user if he wants to the prediction, if yes then ask user to provide the input data and make predication using the best model
14. show prediction to the user.