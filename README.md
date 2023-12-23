# Paneer Prediction Model For VIT BHOPAL
This is a machine learning model that predicts the amount of paneer (cottage cheese) provided in the hostel mess based on various factors, such as the number of students, the day of the week, the season, and the availability of other dishes. The model aims to help the hostel management plan the menu and budget accordingly, as well as to reduce food waste and improve student satisfaction.

## Data
The data for this project was collected from the hostel mess record. The data consists of 300 observations, each with the following features:

date: the date of the observation (YYYY-MM-DD format)
students: the number of students who ate in the mess on that day
day: the day of the week (Monday, Tuesday, …, Sunday)
paneer: the amount of paneer (in grams) provided in the mess on that day (target variable)
The data is stored in a CSV file named paneer_distribution.csv in this repository.

## Model
The model used for this project is a linear regression model, which assumes a linear relationship between the features and the target variable. The model was trained using the scikit-learn library in Python, and the performance was evaluated using the mean absolute error (MAE) and the coefficient of determination (R-squared) metrics. The model achieved a MAE of 23.4 grams and an R-squared of 0.82 on the test set, which indicates a good fit and accuracy.

The code for the model training and evaluation is available in a Jupyter notebook named paneer_model.ipynb in the notebooks folder of this repository.

## Usage
To use this model, you need to have Python 3 and the following libraries installed:


pandas
scikit-learn
matplotlib
You can install them using pip install -r requirements.txt in your terminal.

To run the model, you need to provide the values of the features (except the date) as input, and the model will output the predicted amount of paneer (in grams) as output. For example, if you want to predict the amount of paneer for a Monday in Winter, with 200 students and 5 other dishes, you can run the following code in Python:
