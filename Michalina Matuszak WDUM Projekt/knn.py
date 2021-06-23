import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#metrics
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

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

#Standaryzacja danych
#StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#wyznaczanie optymalnej liczby k (sasiadow)
test_score_list = []
k_list = range(1, 100)

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, Y_train)
    test_score_list.append(knn.score(X_test_scaled, Y_test))
    
print("(k, score)")
k_list[np.argmax(test_score_list)], test_score_list[np.argmax(test_score_list)]

#trenowanie modelu
knn=KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train_scaled, Y_train)
acc_knn_sc = knn.score(X_test_scaled,Y_test)
print("acc: ", acc_knn_sc)

#confusion matrix
Y_pred5 = knn.predict(X_test_scaled)
print(metrics.confusion_matrix(Y_test,Y_pred5))

#wykres do tabelki wyzej
plot_confusion_matrix(knn, X_test_scaled, Y_test)
plt.show()

#classification raport
print(classification_report(Y_test, Y_pred5, target_names=['0', '1']))
