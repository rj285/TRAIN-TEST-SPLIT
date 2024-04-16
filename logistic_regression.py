import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

while True:
    data = {
        'age': [22, 25, 47,  52,  46,  56,  55,  60,  62,  61,  18,  28,  27,  29,  49,  55,  25, 58,  19,  18,  21,  26,  40,  45,  50,  54,  23],
        'bought_insurance': [ 0, 0, 1,  0, 1,1, 0,1, 1, 1, 0, 0,0,0, 1, 1, 1, 1, 0,0,0,0,1,1,1,1, 0]
        }
    # print(len(data['age'])) #27
    # print(len(data['bought_insurance'])) #27

    df = pd.DataFrame(data)
    # print(df)

    # plt.figure(figsize=(8,6))
    # plt.scatter(data["age"],data['bought_insurance'], color='red', marker='x')
    # plt.title('RELATIONSHIP BETWEEN AGE AND INSURRANCE')
    # plt.xlabel('age')
    # plt.ylabel('bought_insurance')
    # plt.savefig('RBAAI.png', format='png')
    # plt.show()

    x = df[['age']]
    y = df[['bought_insurance']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    # print(x_train)
    # print(len(x_train)) #24
    # print(x_test)
    # print(len(x_test)) #3
    # print(y_train)
    # print(len(y_train)) #24
    # print(y_test)
    # print(len(y_test))  #3

    #LogisticRegression
    model = LogisticRegression()

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"Prediction is : {predictions}")
    print(x_test)
    print('--------------------')
    print(y_test)
    input = int(input('Eneter the age to predict:- '))
    answer = model.predict([[input]])
    print(f"predicted for age {input} is : {answer}")
    print('----------------------------------------------')
    chances = model.predict_proba(x_test)
    print(chances)
    
    