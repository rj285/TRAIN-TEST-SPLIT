import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    'age':[30,35,40,45,50,55,60,65,70,75],
    'income':[50000,55000,60000,65000,70000,75000,80000,85000,90000,95000],
    'buy_car':[0,0,1,1,1,1,1,1,1,1]
}

# 1 represent 'will buy a car', 0 represents 'will not buy car'

df = pd.DataFrame(data)
# print(df)
age = df['age']
income = df['income']
buy_car = df['buy_car']
# print(age,income,buy_car)

# plt.figure(figsize=(8,6))
# plt.scatter(age,buy_car, color='red', marker='+')
# plt.title('car-dealership; predicting customers will or will not buy a car')
# plt.xlabel('AGE')
# plt.ylabel('CHANCE OF BUYING A CAR')
# plt.savefig('QA_car_dealership-logistic_regression.png')
# plt.show()

x = df[['age','income']]
y = df[['buy_car']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# print(x_train)
# print(len(x_train)) #7
# print(x_test)
# print(len(x_test)) #3
# print(y_train)
# print(len(y_train)) #7
# print(y_test)
# print(len(y_test))  #3

model = LogisticRegression()

model.fit(x_train, y_train)
predictions = model.predict(x_test)
# print(f"Prediction is : {predictions}")
# print('--------------------')
# print(x_test)
# print('--------------------')
# print(y_test)
input_age = int(input('Enter the age to predict: '))
input_income = int(input('Enter the income to predict: '))
answer = model.predict([[input_age, input_income]])
if answer == 1:
    print(f"For age {input_age} and income {input_income}, the predicted result is: Will buy a car.")
else:
    print(f"For age {input_age} and income {input_income}, the predicted result is: Will not buy a car.")
