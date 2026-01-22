import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\Mohamed\Desktop\AIProject\shopping_behavior.csv')
X = dataset.iloc[:, 1:-1].values #x holds column 1 to 14 (age to payment method)
y = dataset.iloc[:, -1].values #y holds subscription status

#handle missing values (age and review rating)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer = imputer.fit(X[:, [0, 9]]) #X is an array so column 1 (age) is index 0
X[:, [0, 9]] = imputer.transform(X[:, [0, 9]])

#encode strings to numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(X.shape[1]):
    try:
        if isinstance(X[0, i], str) or type(X[0, i]) == str:
            X[:, i] = labelencoder_X.fit_transform(X[:, i])
    except:
        pass
#encode the target
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#model training
#naive bais
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_pred_nb = classifier_nb.predict(X_test)

#decision tree
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_dt.fit(X_train, y_train)
y_pred_dt = classifier_dt.predict(X_test)

#SVC (Support Vector Classifier)
from sklearn.svm import SVC
classifier_svc = SVC(kernel='linear', random_state=0)
classifier_svc.fit(X_train, y_train)
y_pred_svc = classifier_svc.predict(X_test)


#evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
#naive bais
acc_nb = accuracy_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f"Naive Bais Accuracy: {acc_nb}")
print(cm_nb)

#decision Tree
acc_dt = accuracy_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(f"\nDecision Tree Accuracy: {acc_dt}")
print(cm_dt)

#SVC
acc_svc = accuracy_score(y_test, y_pred_svc)
cm_svc = confusion_matrix(y_test, y_pred_svc)
print(f"\nSVC Accuracy: {acc_svc}")
print(cm_svc)