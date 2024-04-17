import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
'''
sklearn.datasets is a module in scikit-learn that provides utility functions to load standard datasets for machine learning. 
load_digits is one such function that loads the handwritten digits dataset, which is commonly used for classification tasks. 
It returns a dictionary-like object containing the data and metadata.
'''

from sklearn.linear_model import LogisticRegression
'''
sklearn.linear_model.LogisticRegression is a class in scikit-learn for logistic regression, a popular method for binary classification tasks. 
It learns a logistic function to predict the probability of a binary outcome based on input features. 
It's commonly used due to its simplicity, interpretability, and effectiveness for linearly separable data.
'''

from sklearn.model_selection import train_test_split
'''
`sklearn.model_selection.train_test_split` splits your dataset into training and testing sets so you can evaluate your model's performance. 
It's like dividing your data into two parts: one for training the model and the other for testing its accuracy. 
Just specify your data and how much to allocate for testing, and it handles the rest.
'''

digits = load_digits()
# print(digits)
# print(dir(digits)) #['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']
'''
dir() is a built-in function that returns a list of valid attributes for the specified object. 
If no object is provided, it returns the list of names in the current local scope. 
'''
# for i in range(10):
#     plt.matshow(digits['images'][i])
#     plt.gray()
# plt.show()

# plt.matshow(digits['images'][0])
# plt.gray()
# # plt.savefig('load_digit(0).png')
# plt.show()

# print(digits['images'][0])
'''
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
'''
# print(len(digits['images'])) #1797
# print(digits.target[:5]) #[0 1 2 3 4]

model = LogisticRegression(max_iter=1000)
'''
creating a logistic regression model with a maximum of 1000 iterations. 
It's a common choice to set a maximum number of iterations to ensure that the model converges to a solution, 
especially when dealing with large datasets or complex relationships between features and the target variable.
'''

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1)
# print(x_train)
# print(len(x_train)) #1617
# print(x_test)
# print(len(x_test)) #180
# print(y_train)
# print(len(y_train))  #1617
# print(y_test)
# print(len(y_test))  #180

# print(digits.data)
'''
[[ 0.  0.  5. ...  0.  0.  0.]
 [ 0.  0.  0. ... 10.  0.  0.]
 [ 0.  0.  0. ... 16.  9.  0.]
 ...
 [ 0.  0.  1. ...  6.  0.  0.]
 [ 0.  0.  2. ... 12.  0.  0.]
 [ 0.  0. 10. ... 12.  1.  0.]]
'''
# print(len(digits.data)) #1797
# print(digits.target) #[0 1 2 ... 8 9 8]
# print(len(digits.target)) #1797

model.fit(x_train,y_train)
predictions = model.predict(x_test)

# print("The Prediction:- ",predictions)
# print('------------------------------------------')
# print(y_test)

'''
#just checking and comparing thepredicted data with test data
x = [4, 0, 3, 6, 1, 0, 0, 1, 5, 0, 3, 1, 8, 7, 4, 0, 8, 1, 5, 8, 3, 3, 4, 0, 8, 7, 4, 1, 2, 5, 2, 6, 8, 1, 0, 2, 4, 5, 2, 6, 0, 4, 1, 1, 4, 6, 2, 9, 9, 4, 0, 9, 0, 9, 6, 7, 6, 2, 0, 2, 0, 2, 0, 2, 9, 3, 6, 3, 1, 1, 7, 4, 7, 6, 1, 8, 4, 4, 9, 7, 2, 2, 2, 0, 9, 6, 6, 2, 0, 3, 1, 9, 3, 1, 4, 4, 1, 3, 6, 6, 8, 5, 5, 4, 2, 8, 7, 0, 7, 5, 6, 2, 2, 3, 0, 5, 4, 0, 7, 6, 5, 4, 4, 6, 0, 6, 5, 3, 3, 6, 1, 9, 8, 3, 2, 9, 8, 4, 5, 5, 8, 0, 0, 6, 4, 4, 6, 3, 6, 1, 3, 7, 1, 5, 6, 9, 6, 6, 7, 6, 2, 8, 3, 1, 4, 7, 4, 5, 6, 3, 9, 5, 6, 2, 1, 1, 0, 4, 5, 2]
y = [4, 0, 3, 6, 1, 0, 0, 1, 5, 0, 3, 1, 9, 7, 4, 0, 8, 1, 5, 8, 3, 3, 4, 0, 8, 7, 4, 1, 2, 5, 2, 6, 8, 1, 0, 2, 4, 5, 2, 6, 0, 4, 1, 1, 4, 6, 2, 9, 9, 4, 0, 9, 0, 7, 6, 7, 6, 2, 0, 2, 0, 2, 0, 2, 9, 3, 6, 3, 1, 1, 7, 4, 7, 6, 1, 8, 4, 4, 9, 7, 2, 2, 2, 0, 9, 6, 6, 2, 0, 3, 8, 9, 3, 1, 4, 5, 1, 7, 6, 6, 8, 5, 5, 4, 2, 8, 7, 0, 7, 5, 6, 2, 2, 3, 0, 5, 4, 0, 7, 6, 5, 4, 4, 6, 0, 6, 5, 3, 2, 6, 1, 9, 8, 3, 2, 9, 6, 4, 5, 5, 8, 0, 0, 6, 4, 4, 6, 2, 6, 1, 3, 7, 1, 5, 6, 9, 6, 6, 7, 6, 2, 8, 3, 1, 4, 7, 4, 5, 6, 3, 9, 5, 6, 2, 1, 1, 0, 4, 5, 2]

true_count = 0
false_count = 0

for i, j in zip(x, y):
    if i == j:
        print(True)
        true_count += 1
    else:
        print(False)
        false_count += 1

print("Number of True:", true_count)
print("Number of False:", false_count)
print(len(x))
'''