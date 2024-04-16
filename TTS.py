import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #pip install scikit-learn
from sklearn.linear_model import LinearRegression

data = {
    "SquareFeet": [2000, 1500, 2200, 1200, 1800, 2500, 1900, 2300, 2600, 2100, 2800, 1700, 2400, 2000, 3000, 1600, 2800, 3200, 1800, 2000],
    "HouseAge": [5, 3, 6, 2, 4, 5, 5, 7, 8, 6, 7, 4, 5, 3, 6, 2, 8, 7, 4, 5],
    "Price($)": [300000, 250000, 320000, 200000, 280000, 350000, 270000, 330000, 360000, 310000, 380000, 240000, 320000, 300000, 400000, 230000, 380000, 420000, 260000, 280000]
}

'''
Separate features (X) and target variable (y)
X = df[['SquareFeet', 'HouseAge']]  Features  (independent variables) 
y = df['Price($)']  Target variable (dependent variable)
'''

df = pd.DataFrame(data)
# print(df)

plt.figure(figsize=(8,6))
plt.scatter(data["SquareFeet"],data["Price($)"], color='red', marker='x')
plt.title('RELATIONSHIP BETWEEN SQUAREFEET AND SELLING PRICE')
plt.xlabel('SquareFeet')
plt.ylabel('Price($)')
plt.savefig('RBSASP.png', format='png')
plt.show()

'''
30% for test sample and 70% is training sample
'''
x = df[['SquareFeet','HouseAge']]
y = df[['Price($)']]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

'''
Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Here, X_train and y_train are the training features and target variable, respectively.
Similarly, X_test and y_test are the testing features and target variable, respectively.
test_size is the proportion of the dataset to include in the test split (0.3 means 30% of the data will be used for testing).
random_state is a seed for the random number generator used to split the data, ensuring reproducibility.
'''
# print(len(X_train)) #14
# print(len(X_test)) #6
# print(len(Y_train)) #14
# print(len(Y_test)) #6

#LinearRegression
clf = LinearRegression()
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
print(f"Prediction is : {predictions}")
print(f"Actual price is : {Y_test}")

score = clf.score(X_test,Y_test)
print(f"The score : {score}")
