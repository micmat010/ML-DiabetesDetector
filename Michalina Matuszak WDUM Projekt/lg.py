import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

#trenowanie modelu
lg = LogisticRegression(C=100, max_iter=1000)
lg.fit(X_train_scaled,Y_train)
acc_lg_sc= lg.score(X_test_scaled, Y_test)
print("acc: ", acc_lg_sc)

#confusion matrix
Y_pred4 = lg.predict(X_test_scaled)
print(metrics.confusion_matrix(Y_test,Y_pred4))

#wykres do tabelki wyzej
plot_confusion_matrix(lg, X_test_scaled, Y_test)
plt.show()

#classification raport
print(classification_report(Y_test, Y_pred4, target_names=['0', '1']))

#ocena znaczenia poszczegolnych cech
from matplotlib import pyplot
importance = lg.coef_[0]

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
