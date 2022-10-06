import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

#load dataset
iris = datasets.load_iris()
#print(iris.data)
'''print(iris.target)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

0-setosa
1-versicolor
2-virginia

 '''

#dependent and independent variables
x = iris.data
y = iris.target

#split data into training and testing setosa
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


#fit model_
model = GradientBoostingClassifier()
model.fit(x_train, y_train)
model_score = model.score(x_test, y_test)
# print("model score is:{}".format(round(model_score, 2)))

pred = model.predict(x_test)
# print(pred)
print("\n confusion matrix", confusion_matrix(pred, y_test))

print("\n accuracy :", round(accuracy_score(pred, y_test), 2))

#save model
pickle.dump(model, open("flowerpredictor.pkl", "wb"))
print("model saved successfully!!")
