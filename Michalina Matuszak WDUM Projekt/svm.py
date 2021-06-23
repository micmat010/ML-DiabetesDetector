import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#wczytanie danych
df = pd.read_csv('C:/Users/MSI-PC/WDUM/diabetes/diabetes.csv')

#Wstawianie NaN zamiast zerowych wartosci:
df.Insulin.replace(0, np.nan, inplace=True)
df.SkinThickness.replace(0, np.nan, inplace=True)
df.BMI.replace(0, np.nan, inplace=True)

#wstawianie mediany zamiast brakujacych wartosci
#Zaokrąglenie float do int, aby zachować spójność typów zmiennych
df = df.fillna(df.median()).round(0).astype(int)

#cechy jako zbiór X
X = df.drop('Outcome', axis=1)

#wynik jako Y
Y = df['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

#import modułów
from sklearn import svm
from sklearn.svm import SVC

#normalizacja danych
#MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#import modułów
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

#wybieranie optymalnego parametru C dla algorytmu SVM
svm_score_list = []
n_list = range(1, 10)

for n in n_list:
    clf = svm.SVC(C=n, kernel="rbf")
    clf.fit(X_train_scaled, Y_train)
    svm_score_list.append(clf.score(X_test_scaled, Y_test))

#wyświetlenie optymalnego parametru
print("(C, score)")
n_list[np.argmax(svm_score_list)], svm_score_list[np.argmax(svm_score_list)]

#trenowanie modelu
clf = svm.SVC(C = 1, kernel='rbf') # C-Support Vector Classification
clf.fit(X_train_scaled,Y_train)
minmax_score = clf.score(X_test_scaled,Y_test) #accuracy
print("acc: ", minmax_score)

#confusion matrix
Y_pred_minmax = clf.predict(X_test_scaled)
print(metrics.confusion_matrix(Y_test,Y_pred_minmax))

plot_confusion_matrix(clf, X_test_scaled, Y_test) #wykres do tabelki wyżej
plt.show()

#classification_report
print(classification_report(Y_test, Y_pred_minmax, target_names=['0', '1']))
