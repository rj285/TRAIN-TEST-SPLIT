import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Volkswagen_Polo_gt_olx.csv')
# print(data)

df = pd.DataFrame(data)
# print(df.isna().sum())
# print(df)

# plt.figure(figsize=(8,6))
# plt.scatter(data['Year'],data['Price'], color='red', marker='x')
# plt.title('VOLKSWAGEN POLO GT PRICE BASED ON YEAR')
# plt.xlabel('YEAR')
# plt.ylabel('PRICE')
# plt.savefig('Volkswagen_Polo_gt_olx.png', format='png')
# plt.show()

'''
30% for test sample and 70% is training sample
'''

x = df[['Year','Kilometers']]
y = df[['Price']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# print(x_train)
# print(len(x_train)) #35
# print(x_test)
# print(len(x_test)) #16
# print(y_train)
# print(len(y_train)) #35
# print(y_test)
# print(len(y_test)) #16

#LinearRegression
clf = LinearRegression()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print(f"prediction is : {predictions}")
print(f"actual price is {y_test}")

score = clf.score(x_test,y_test)
print(f"The score: {score}")
