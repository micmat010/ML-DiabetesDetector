Zawartość archiwum:
1. diabetes.csv - zbiór danych
2. svm.py - plik z modelem Support Vector Machines
3. lg.py - plik z modelem Logistic Regression
4. knn.py - plik z modelem K Nearest Neighbors
5. Michalina_Matuszak_WDUM_Projekt.ipynb - plik Jupyter Notebook
6. read_me.md - bieżący plik

Każdy z plików podzielony jest na sekcje oddzielone komentarzami. Do uruchomienia plików projektu potrzebna jest instalacja odpowiednich modułów: pandas, numpy, sklearn, matplotlib.

Raport ten jest bezpośrednio eksportowany z Jupyter Notebook, zawiera kod użyty w projekcie oraz jego opis w formatowaniu Markdown. Zawartość notebooka:

# Wstęp do uczenia maszynowego - projekt 
## Michalina Matuszak UTP Bydgoszcz

# Diabetes Data Set
####  Informacje niezbędne do zrozumienia projektu:

Zbiór danych o diabetykach zawierający spis wyników badań krwi oraz wywiadu medycznego od pacjentów bez oraz z zarejestrowaną cukrzycą. Obszarem badań jest cukrzyca typu 2, nazywana także cukrzycą insulinoniezależną. Cukrzyca typu 2 należy do grupy chorób metabolicznych i charakteryzuje się występowaniem wysokiego poziomy glukozy we krwi oraz opornością na insulinę i względnym jej niedoborem. Za dwie podstawowe przyczyny cukrzycy typu 2 uważa się wspomnianą już insulinooporność oraz upośledzenie wydzielania insuliny. Chorobę dagnozuje się wykonując regularne pomiary poziomu glukozy we krwi, ponieważ wartość ta bezpośdrednio wynika z udziału insuliny. Czynnikami pomocnymi przy diagnozie są BMI, wiek, dieta, tryb życia, ciśnienie tętnicze i inne.
W zbiorze danych cechy odpowiadają czynnikom badanym podczas diagnozy choroby. 

#### Celem projektu jest: 
1. Analiza znaczenia danej cechy przy diagnozie choroby oraz analiza wzajemnych korelacji.
2. Stworzenie modelu klasyfikującego badaną osobę na podstawie dostępnych czynników (0 - niesklasyfikowano choroby, 1- cukrzyca).
3. Ocena jakości poszczególnych modeli w zależności od użytego algorytmu.

#### Informacje o cechach zbioru:
0. Pregnancies - liczba przebytych ciąż
1. Glucose - poziom glukozy we krwi
2. BloodPressure - ciśnienie tętnicze krwi
3. SkinThickness - grubość skóry
4. Insulin - poziom insuliny we krwi
5. BMI - wskaźnik masy ciała
6. DiabetesPedigreeFunction - funkcja określająca występowanie określonego genu u danej osoby
7. Age - wiek
8. Outcome - wynik (0 - brak cukrzycy, 1 - cukrzyca)



```python
import numpy as np
import pandas as pd
import matplotlib as plt
%matplotlib inline
import seaborn as sns
```


```python
df = pd.read_csv('C:/Users/MSI-PC/WDUM/diabetes/diabetes.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    

# Eksplorazyjna analiza danych

**Po wyświetlenie 5 pierwszych wierszy ze zbioru widać, że występują w nim brakujące wartości** 


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Wymiar zbioru danych: (liczba wierszy, liczba cech)**


```python
df.shape
```




    (768, 9)



**Liczba poszczególnych wyników: (jak widać mamy więcej informacji o osobach bez cukrzycy)**


```python
# 0- brak cukrzycy
# 1- cukrzyca
df.Outcome.value_counts()
```




    0    500
    1    268
    Name: Outcome, dtype: int64



**Ponieważ prawie połowa danych zawiera *missing values*, najrozsądniejszą opcją jest zastąpienie wartosci zerowych na medianę z danej kolumny:**


```python
#Wstawianie NaN zamiast zerowych wartosci:
df.Insulin.replace(0, np.nan, inplace=True)
df.SkinThickness.replace(0, np.nan, inplace=True)
df.BMI.replace(0, np.nan, inplace=True)
```


```python
#Zaokrąglenie float do int, aby zachować spójność typów zmiennych
df = df.fillna(df.median()).round(0).astype(int) 
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>125</td>
      <td>34</td>
      <td>1</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>125</td>
      <td>27</td>
      <td>0</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>29</td>
      <td>125</td>
      <td>23</td>
      <td>1</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43</td>
      <td>2</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Wybrane statystyki zbioru:**


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>29.108073</td>
      <td>140.671875</td>
      <td>32.447917</td>
      <td>0.373698</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>8.791221</td>
      <td>86.383060</td>
      <td>6.868092</td>
      <td>0.510322</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>25.000000</td>
      <td>121.500000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>29.000000</td>
      <td>125.000000</td>
      <td>32.000000</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>37.000000</td>
      <td>1.000000</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.000000</td>
      <td>2.000000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Korelacja między cechami:**


```python
corr = df.corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pregnancies</th>
      <td>1.000000</td>
      <td>0.129459</td>
      <td>0.141282</td>
      <td>0.081770</td>
      <td>0.025047</td>
      <td>0.024186</td>
      <td>-0.023148</td>
      <td>0.544341</td>
      <td>0.221898</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>0.129459</td>
      <td>1.000000</td>
      <td>0.152590</td>
      <td>0.182037</td>
      <td>0.409283</td>
      <td>0.217741</td>
      <td>0.112770</td>
      <td>0.263514</td>
      <td>0.466581</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>0.141282</td>
      <td>0.152590</td>
      <td>1.000000</td>
      <td>0.124770</td>
      <td>0.059146</td>
      <td>0.184299</td>
      <td>0.007224</td>
      <td>0.239528</td>
      <td>0.065068</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>0.081770</td>
      <td>0.182037</td>
      <td>0.124770</td>
      <td>1.000000</td>
      <td>0.155610</td>
      <td>0.544493</td>
      <td>0.082819</td>
      <td>0.126107</td>
      <td>0.214873</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>0.025047</td>
      <td>0.409283</td>
      <td>0.059146</td>
      <td>0.155610</td>
      <td>1.000000</td>
      <td>0.180013</td>
      <td>0.128747</td>
      <td>0.097101</td>
      <td>0.203790</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>0.024186</td>
      <td>0.217741</td>
      <td>0.184299</td>
      <td>0.544493</td>
      <td>0.180013</td>
      <td>1.000000</td>
      <td>0.120317</td>
      <td>0.027734</td>
      <td>0.310830</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>-0.023148</td>
      <td>0.112770</td>
      <td>0.007224</td>
      <td>0.082819</td>
      <td>0.128747</td>
      <td>0.120317</td>
      <td>1.000000</td>
      <td>0.041030</td>
      <td>0.159888</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.544341</td>
      <td>0.263514</td>
      <td>0.239528</td>
      <td>0.126107</td>
      <td>0.097101</td>
      <td>0.027734</td>
      <td>0.041030</td>
      <td>1.000000</td>
      <td>0.238356</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>0.221898</td>
      <td>0.466581</td>
      <td>0.065068</td>
      <td>0.214873</td>
      <td>0.203790</td>
      <td>0.310830</td>
      <td>0.159888</td>
      <td>0.238356</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Wnioski:


```python
1. Wynik jest pozytywnie powiązany z poziomem glukozy we krwi
2. Wiek i liczba ciąż ma pozytywną korelację
3. BMI i grubość skóry mają pozytywną korelację
```


      File "<ipython-input-1-a524d99a6052>", line 1
        1. Wynik jest pozytywnie powiązany z poziomem glukozy we krwi
           ^
    SyntaxError: invalid syntax
    


**Aby lepiej zwizualizować wybrane wartości korelacji posłuży heatmap:**


```python
sns.heatmap(corr)
```




    <AxesSubplot:>




    
![png](output_25_1.png)
    


**Na poniższym wykresie widać współzależności danych cech z dodatkowym uwzględnieniem wartości wynikowej:**


```python
sns.pairplot(df, diag_kind='kde', hue="Outcome")
```




    <seaborn.axisgrid.PairGrid at 0x2ba66e98>




    
![png](output_27_1.png)
    


# Uczenie z nadzorem (ang. supervised learning) - klasyfikacja

https://scikit-learn.org/stable/supervised_learning.html

## Przygotowanie modelu:

**Podział danych na zbiór treningowy (70%) i testowy (30%):**


```python
from sklearn.model_selection import train_test_split
```


```python
#cechy jako zbiór X
X = df.drop('Outcome', axis=1)

#wynik jako Y
Y = df['Outcome']
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
```

## Algorytm Support Vector Machines SVM

https://scikit-learn.org/stable/modules/svm.html#svm

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC


```python
#import modułów
from sklearn import svm
from sklearn.svm import SVC
```

### Normalizacja danych:

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

**Przekształcamy cechy skalując każdą kolumnę do określonego zakresu. Estymator skaluje i tłumaczy każdą cechę indywidualnie tak, aby znajdowała się w podanym zakresie na zbiorze uczącym, np. od zera do jednego.**

```X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))```

```X_scaled = X_std * (max - min) + min```


```python
#MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

### Wyniki po normalizacji, ocena modelu:


```python
#import modułów
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
```

#### Wyznaczanie najbardziej optymalnego parametru C (Support Vector Classification):


```python
svm_score_list=[]
n_list = range(1,10)

for n in n_list:
    clf = svm.SVC( C=n, kernel="rbf")
    clf.fit(X_train_scaled, Y_train)
    svm_score_list.append(clf.score(X_test_scaled, Y_test))
   
print("(C, score)")
n_list[np.argmax(svm_score_list)], svm_score_list[np.argmax(svm_score_list)]
```

    (C, score)
    




    (1, 0.7792207792207793)



#### Wykres zależności parametru *C* od *score*:


```python
sns.lineplot(n_list, svm_score_list)
```

    c:\users\msi-pc\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:>




    
![png](output_49_2.png)
    


#### Accuracy:


```python
# Ocena modelu, który przyjmuje znormalizowane wartości
clf = svm.SVC(C = 1, kernel='rbf') # C-Support Vector Classification
clf.fit(X_train_scaled,Y_train)
minmax_score = clf.score(X_test_scaled,Y_test) #accuracy
print("MinMax score: ")
print("acc: ", minmax_score) 
```

    MinMax score: 
    acc:  0.7792207792207793
    

#### Confusion matrix:


```python
Y_pred_minmax = clf.predict(X_test_scaled)
print(metrics.confusion_matrix(Y_test,Y_pred_minmax)) #confusion matrix
```

    [[133  13]
     [ 38  47]]
    


```python
plot_confusion_matrix(clf, X_test_scaled, Y_test) #wykres do tabelki wyżej
plt.show()
```


    
![png](output_54_0.png)
    


#### Classification report:


```python
print(classification_report(Y_test, Y_pred_minmax, target_names=['0', '1']))
```

                  precision    recall  f1-score   support
    
               0       0.78      0.91      0.84       146
               1       0.78      0.55      0.65        85
    
        accuracy                           0.78       231
       macro avg       0.78      0.73      0.74       231
    weighted avg       0.78      0.78      0.77       231
    
    

## Wniosek:

**Ocena jakości modelu: średnia.**
**Model klasyfikujący chorobę u badanych powinien mieć większą czułość, niż precyzję (podawać więcej false positivów, niż false negativów), ponieważ diagnozując chorobę lepiej jest wykryć nieistniejącą chorobę (i taki wynik traktować jako klasyfikację do grupy podwyższonego ryzyka lub przedwczesną profilaktykę), niż nie wykryć niczego pomimo tego, że pacjent w rzeczywistości jest już chory.**

# Logistic Regression

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


```python
from sklearn.linear_model import LogisticRegression
```

### Standaryzacja danych:

**Standaryzujemy dane, usuwając średnią i skalując do wariancji jednostkowej.  Standardowy wynik próbki x oblicza się jako: 
```z = (x - u) / s ``` 
gdzie:
`u` jest średnią z próbek uczących, 
`s` jest odchyleniem standardowym próbek uczących**


```python
#StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

#### Accuracy:


```python
lg = LogisticRegression(C=100, max_iter=1000)
lg.fit(X_train_scaled,Y_train)
acc_lg_sc= lg.score(X_test_scaled, Y_test)
print("acc: ", acc_lg_sc)
```

    acc:  0.7748917748917749
    

#### Confusion matrix:


```python
Y_pred4 = lg.predict(X_test_scaled)
print(metrics.confusion_matrix(Y_test,Y_pred4))
```

    [[131  15]
     [ 37  48]]
    


```python
plot_confusion_matrix(lg, X_test_scaled, Y_test)
plt.show()
```


    
![png](output_69_0.png)
    


#### Classification report:


```python
print(classification_report(Y_test, Y_pred4, target_names=['0', '1']))
```

                  precision    recall  f1-score   support
    
               0       0.78      0.90      0.83       146
               1       0.76      0.56      0.65        85
    
        accuracy                           0.77       231
       macro avg       0.77      0.73      0.74       231
    weighted avg       0.77      0.77      0.77       231
    
    

#### Ocena znaczenia poszczególnych cech:


```python
from matplotlib import pyplot
importance = lg.coef_[0]

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

    Feature: 0, Score: 0.32586
    Feature: 1, Score: 1.05017
    Feature: 2, Score: -0.29311
    Feature: 3, Score: -0.06959
    Feature: 4, Score: -0.05448
    Feature: 5, Score: 0.74772
    Feature: 6, Score: 0.22185
    Feature: 7, Score: 0.25386
    


    
![png](output_73_1.png)
    



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype
    ---  ------                    --------------  -----
     0   Pregnancies               768 non-null    int32
     1   Glucose                   768 non-null    int32
     2   BloodPressure             768 non-null    int32
     3   SkinThickness             768 non-null    int32
     4   Insulin                   768 non-null    int32
     5   BMI                       768 non-null    int32
     6   DiabetesPedigreeFunction  768 non-null    int32
     7   Age                       768 non-null    int32
     8   Outcome                   768 non-null    int32
    dtypes: int32(9)
    memory usage: 27.1 KB
    

## Wniosek:

**Model używający regresji logistycznej jest w niewielkim stopniu bardziej trafny oraz posiada większą czułość niż model SVM.  Dodatkowo możemy przeanalizować znaczenie poszczególnych cech oraz ich wpływ na klasyfikację choroby. Jak widać na wykresie największy wpływ ma poziom glukozy (cecha 1) oraz BMI (cecha 5), najmniejsze znaczenie ma ciśnienie krwi (cecha 2).**

# Algorytm K-najbliższych sąsiadów

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


```python
from sklearn.neighbors import KNeighborsClassifier
```

#### Wyzaczanie najbardziej optymalnej liczby *k*:


```python
test_score_list = []
k_list = range(1, 100)

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, Y_train)
    test_score_list.append(knn.score(X_test_scaled, Y_test))
    
print("(k, score)")
k_list[np.argmax(test_score_list)], test_score_list[np.argmax(test_score_list)]
```

    (k, score)
    




    (23, 0.7965367965367965)



#### Wykres zależności liczby *k* od *score*:


```python
sns.lineplot(k_list, test_score_list)
```

    c:\users\msi-pc\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:>




    
![png](output_83_2.png)
    


#### Accuracy:


```python
knn=KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train_scaled, Y_train)
acc_knn_sc = knn.score(X_test_scaled,Y_test)
print("acc: ", acc_knn_sc)
```

    acc:  0.7965367965367965
    

#### Confusion matrix:


```python
Y_pred5 = knn.predict(X_test_scaled)
print(metrics.confusion_matrix(Y_test,Y_pred5))
```

    [[135  11]
     [ 36  49]]
    


```python
plot_confusion_matrix(knn, X_test_scaled, Y_test)
plt.show()
```


    
![png](output_88_0.png)
    


#### Classification report:


```python
print(classification_report(Y_test, Y_pred5, target_names=['0', '1']))
```

                  precision    recall  f1-score   support
    
               0       0.79      0.92      0.85       146
               1       0.82      0.58      0.68        85
    
        accuracy                           0.80       231
       macro avg       0.80      0.75      0.76       231
    weighted avg       0.80      0.80      0.79       231
    
    

## Wniosek:

**Model wykorzystujacy algorytm K-najbliższych sąsiadów jest najbardziej trafny, jednakże również jak w przypadku modelu SVM posiada niską czułość.**
