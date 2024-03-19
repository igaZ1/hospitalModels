from collections import Counter
from random import random, randint

import np as np
import pandas as pandas
import matplotlib.pyplot as plt
import sklearn as sklearn
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from modelChem import predictChem3, predictChem10, predictChem30
from models import make_models, plot_departmentTryb, plot_department, plot_departmentsAll

from modelSOR import predictSOR3, predictSOR10, predictSOR30
from modelHosp import predict10, predict3, predict30



"""
main.py
obliczenia do modeli i testy funkcji
"""





dsKardio16 = pandas.read_table('kardio/KARDIO2016.csv', sep=',')
dsSOR16A = pandas.read_table('all/SOR2016A.csv', sep=',')
dsSOR16B = pandas.read_table('all/SOR2016B.csv', sep=',')
dsSOR16 = pandas.concat([dsSOR16A, dsSOR16B])

dsKardio17 = pandas.read_table('kardio/KARDIO2017.csv', sep=',')
dsSOR17A = pandas.read_table('all/SOR2017A.csv', sep=',')
dsSOR17B = pandas.read_table('all/SOR2017B.csv', sep=',')
dsSOR17 = pandas.concat([dsSOR17A, dsSOR17B])

dsKardio18 = pandas.read_table('kardio/KARDIO2018.csv', sep=',')
dsSOR18A = pandas.read_table('all/SOR2018A.csv', sep=',')
dsSOR18B = pandas.read_table('all/SOR2018B.csv', sep=',')
dsSOR18 = pandas.concat([dsSOR18A, dsSOR18B])

dsKardio19 = pandas.read_table('kardio/KARDIO2019.csv', sep=',')
dsSOR19A = pandas.read_table('all/SOR2019A.csv', sep=',')
dsSOR19B = pandas.read_table('all/SOR2019B.csv', sep=',')
dsSOR19 = pandas.concat([dsSOR19A, dsSOR19B])

dsKardio20 = pandas.read_table('kardio/KARDIO2020.csv', sep=',')
dsSOR20A = pandas.read_table('all/SOR2020A.csv', sep=',')
dsSOR20B = pandas.read_table('all/SOR2020B.csv', sep=',')
dsSOR20 = pandas.concat([dsSOR20A, dsSOR20B])

dsKardio21 = pandas.read_table('kardio/KARDIO2021.csv', sep=',')
dsSOR21A = pandas.read_table('all/SOR2021A.csv', sep=',')
dsSOR21B = pandas.read_table('all/SOR2021B.csv', sep=',')
dsSOR21 = pandas.concat([dsSOR21A, dsSOR21B])

dsKardio22 = pandas.read_table('kardio/KARDIO2022.csv', sep=',')
dsSOR22A = pandas.read_table('all/SOR2022A.csv', sep=',')
dsSOR22B = pandas.read_table('all/SOR2022B.csv', sep=',')
dsSOR22 = pandas.concat([dsSOR22A, dsSOR22B])

dsSOR22['Data przyjęcia do szpitala'] = pandas.to_datetime(dsSOR22['Data przyjęcia do szpitala'], errors='coerce')
dsSOR21['Data przyjęcia do szpitala'] = pandas.to_datetime(dsSOR21['Data przyjęcia do szpitala'], errors='coerce')
# dsSOR21['Data przyjęcia do szpitala'] = pandas.to_datetime(dsSOR22['Data przyjęcia do szpitala'],  format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')


allSOR = pandas.concat([dsSOR22, dsSOR21, dsSOR20, dsSOR19, dsSOR18, dsSOR17, dsSOR16])
allKARDIO = pandas.concat([dsKardio22, dsKardio21, dsKardio20, dsKardio19, dsKardio18, dsKardio17, dsKardio16])
# allSOR = allSOR.drop('Liczba dni pobytu w szpitalu', axis=1)
# print(dsKardio16.head(10))
# print(dsSOR16A.head(10))
# print(dsSOR16A.describe())
# print(dsSOR16B.describe())

import pandas as pd


# #Czy allSOR[Kardio] = allKARDIO?
# # Wycięcie oddziału z allSOR
# oddzial = 'Kardiologia'
# allSOR_cut = allSOR[allSOR['Oddział pobyt główny'] == oddzial]
# if allSOR_cut.equals(allKARDIO):
#     print('Zestaw danych jest taki sam.')
# else:
#     print('Zestaw danych jest inny.')
#
# print('Liczba rekordów w allSOR:', allSOR_cut.shape[0])
# print('Liczba rekordów w allSOR:', allSOR.shape[0])
# print('Liczba rekordów w allKARDIO:', allKARDIO.shape[0])


print('---edycja danych---')
# Ustawienie domyślnego rozmiaru czcionki dla wszystkich wykresów
# plt.rcParams.update({'font.size': 30})


print(allSOR)
print(allSOR.describe())
# Usuniecie niepotrzebnych kolumn
allSOR = allSOR.drop('Liczba dni pobytu w szpitalu', axis=1)
# allSOR = allSOR.drop('Liczba dni pobytu w szpitalu', axis=1)

# print(dsSOR16B.head(10))
pandas.set_option('display.max_columns', 10)
pandas.options.display.max_columns = 10
print(allSOR.head(20))  # pierwsze 20 wierszy
print(allSOR['Tryb przyjęcia'].value_counts())  # pierwsze 20 wierszy

# print(dsSOR16.columns[0])
# dsSOR16['Tryb przyjęcia']

num_records = allSOR.shape[0]
print("Liczba rekordów:", num_records)

print('---Histogram trybu przyjęć---')
# Konwertowanie kolumny 'date' na typ daty i ustawienie jej jako indeks
allSOR['Data przyjęcia do szpitala'] = pandas.to_datetime(allSOR['Data przyjęcia do szpitala'], errors='coerce')
allSOR = allSOR.loc[
    (allSOR['Data przyjęcia do szpitala'] >= '2016-01-01') & (allSOR['Data przyjęcia do szpitala'] <= '2022-12-31')]
# allSOR.set_index('Data przyjęcia do szpitala', inplace=True)

# Tworzenie słownika mapującego oryginalne nazwy na nowe nazwy
column_names = {
    'Przyj. w trybie nagłym - inne przypadki': 'Przyj. w trybie nagłym',
    'Przyj. w trybie nagłym w wyniku przekazania przez zespół ratownictwa medycznego': 'Przyj. w trybie nagłym',
    'Przyj. w trybie nagłym bez skierowania': 'Przyj. w trybie nagłym',
    'Przyj. w trybie nagłym ze skier. z pomocy doraźnej': 'Przyj. w trybie nagłym',  # ??
    # 'bez skierowania inne': 'Przyj. w trybie nagłym',
    'Przyj. w trybie nagłym ze skier. innym niż pom. doraźnej': 'Przyj. w trybie nagłym',
    'Bez skier. - zagrożenie życia': 'Przyj. w trybie nagłym',

    'Przyj. planowe na podstawie skierowania': 'Przyj. planowe',
    'Przeniesienie z innego szpitala': 'Przyj. planowe',
    'Przyj. planowe osoby poza kolejnością, zgodnie z uprawnieniami przysługującymi jej na podstawie ustawy': 'Przyj. planowe',
    'Przyjęcie planowe': 'Przyj. planowe',
    'ze skierowaniem z zakładu opieki stacjonarnej': 'Przyj. planowe',
    'Przyjęcie przymusowe': 'Przyj. planowe',
    'Przyjęcie osoby podlegającej obowiązkowemu leczeniu': 'Przyj. planowe',

    'Przyj. noworodka w wyniku porodu w tym szpitalu': 'Przyj. noworodka',
    'Przyj. na podstawie karty diagnostyki i leczenia onkologicznego': 'Leczenie onkologiczne'
}

# Pobieranie danych
howMany = allSOR['Tryb przyjęcia'].value_counts()
howMany = howMany.drop('bez skierowania inne', axis=0)

# Zmiana nazw na podstawie słownika
howMany.rename(index=column_names, inplace=True)

# Połączenie kolumn o tych samych nazwach
howMany = howMany.groupby(howMany.index).sum()

# Sortowanie kolumn w kolejności malejącej
howMany = howMany.sort_values(ascending=False)

print(howMany)

# Tworzenie wykresu histogramu
plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
ax = howMany.plot(kind='bar')  # Wygenerowanie histogramu

plt.xlabel('Tryb przyjęcia', fontsize=20)  # Etykieta osi X
plt.ylabel('Liczba wystąpień', fontsize=20)  # Etykieta osi Y
plt.title('\n Tryb przyjęcia', fontsize=30)  # Tytuł wykresu

# Zmiana orientacji etykiet na osi X
ax.tick_params(axis='x', rotation=0, labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# # Zmiana rozmiaru tekstu dla etykiet pod słupkami
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

plt.show()  # Wyświetlenie wykresu

print(allSOR.columns)
patientsInDay = allSOR.groupby(
    pandas.Grouper(key='Data przyjęcia do szpitala', freq='M')).size()  # ile w danym miesiącu
patientsInDay = allSOR.groupby(pandas.Grouper(key='Data przyjęcia do szpitala', freq='D')).size()  # ile w danym dniu
#


# ///////////
print(patientsInDay)

# wykres szeregu czasowego
plt.plot(patientsInDay.index, patientsInDay.values)
plt.title('Liczba pacjentów w szpitalu danego dnia', fontsize=30)
plt.xlabel('Data', fontsize=20)
plt.ylabel('Liczba pacjentow', fontsize=20)

# Zmiana orientacji etykiet na osi X
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)

plt.show()
#
# print('---Kardio---')
#
# print(allKARDIO)
# print(allKARDIO.columns)
# print(allKARDIO['Data przyjęcia'])
# allKARDIO['Data przyjęcia'] = pandas.to_datetime(allKARDIO['Data przyjęcia'], format="%Y-%m-%d",  errors='coerce')
# # allKARDIO = allKARDIO.loc[(allKARDIO['Data przyjęcia'] >= '2016-01-31') & (allSOR['Data przyjęcia'] <= '2022-12-31')]
#
# patientsKardio = allKARDIO.groupby(pandas.Grouper(key='Data przyjęcia', freq='D')).size()  # ile w danym dniu
#
#
# # wykres szeregu czasowego
# plt.plot(patientsKardio.index, patientsKardio.values, label='Kardio')
# plt.plot(patientsInDay.index, patientsInDay.values, label='SOR')
# plt.title('Liczba pacjentów')
# plt.legend()
# plt.xlabel('Data')
# plt.ylabel('Liczba pacjentow')
# plt.show()


print('--------------------------------------')

print('---Histogram wieku---')
# Tworzenie DataFrame z danymi

print(allSOR.head(20))  # pierwsze 20 wierszy

import pandas as pd

# Tworzenie DataFrame z danymi
df = allSOR

# Usunięcie końcówki '.0' z kolumny 'Data ur.'
df['Data ur.'] = df['Data ur.'].astype(str).str.replace('.0', '')

# Konwertowanie kolumny 'Data ur.' na format daty
df['Data ur.'] = pd.to_datetime(df['Data ur.'], format='%Y')

# Konwertowanie kolumny 'Data przyjęcia do szpitala' na format daty i czasu
df['Data przyjęcia do szpitala'] = pd.to_datetime(df['Data przyjęcia do szpitala'], format='%Y-%m-%d %H:%M:%S')

# Obliczanie wieku pacjentów w dniach
df['Wiek'] = (df['Data przyjęcia do szpitala'] - df['Data ur.']).dt.days

# Przeliczenie wieku na lata
df['Wiek'] = df['Wiek'] / 365

# Analiza grup wiekowych
bins = [0, 18, 30, 40, 50, 60, 70, 120]  # Przykładowe przedziały wiekowe
labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71+']  # Etykiety grup wiekowych
df['Grupa wiekowa'] = pd.cut(df['Wiek'], bins=bins, labels=labels, right=False)

# Wyświetlanie wyników
print(df)

import matplotlib.pyplot as plt

# Filtrowanie danych, pomijając wpisy z płcią 'X'
filtered_df = df[df['Płeć'] != 'X']

# Grupowanie pacjentów według grup wiekowych i płci oraz obliczanie liczności w każdej grupie
grouped = filtered_df.groupby(['Grupa wiekowa', 'Płeć']).size().unstack()

# Generowanie histogramu
grouped.plot(kind='bar', stacked=True)

# Konfiguracja wykresu
plt.xlabel('Grupa wiekowa', fontsize=20)
plt.ylabel('Liczba pacjentów', fontsize=20)
plt.xticks(rotation=0)
plt.title('Histogram grupy wiekowej i płci', fontsize=30)

# Zmiana orientacji etykiet na osi X
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
# Wyświetlenie wykresu
plt.show()

# Sprawdzanie liczby wpisów o płci 'X'
# count_x = df[df['Płeć'] == 'X'].shape[0]
#
# print("Liczba wpisów o płci 'X':", count_x)


print('---Histogram Oddziału---')

import re

allSOR['Tryb przyjęcia'] = allSOR['Tryb przyjęcia'].replace(column_names)
pattern = re.compile(r'bez skierowania inne', flags=re.IGNORECASE)
mask = allSOR['Tryb przyjęcia'].str.contains(pattern)
allSOR = allSOR[~mask]

howMany2 = allSOR['Tryb przyjęcia'].value_counts()
print(howMany2)

column_names_odzial = {
    'Chir. Urazowo - Ortopedyczna': 'Chirurgia Urazowo - Ortopedyczna',
    'Angiologia': 'Choroby Wewnętrzne',
}

allSOR.loc[:, 'Oddział pobyt główny'] = allSOR['Oddział pobyt główny'].replace(column_names_odzial)

howMany2 = allSOR['Oddział pobyt główny'].value_counts()
print(howMany2)

# print(allSOR)

# Grupowanie danych po kolumnach 'Oddział pobyt główny' i 'Tryb przyjęcia' oraz zliczanie liczby pacjentów
grouped_data = allSOR.groupby(['Oddział pobyt główny', 'Tryb przyjęcia']).size().unstack()

# Posortowanie kolumn według sumy wartości w każdym wierszu
grouped_data = grouped_data[grouped_data.sum().sort_values().index]

# Wygenerowanie histogramu z zróżnicowaniem na podstawie 'Tryb przyjęcia'
grouped_data.plot(kind='bar', stacked=True)
plt.title('Oddział pobytu głównego', fontsize=30)
plt.xlabel('Oddział', fontsize=20)
plt.ylabel('Liczba pacjentów', fontsize=20)
# Zmiana etykiet na osi X

plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.legend(title='Tryb przyjęcia')

# Zmiana etykiet na osi X w pionie
plt.xticks(rotation=90, fontsize=16, ha='center')  # ha='center' ustawia etykiety na środku

# plt.show()

print('---ile pacjentów na oddziale w czasie---')
from matplotlib.dates import YearLocator, DateFormatter

# # # ///////////////////
# department = 'Chirurgia Onkologiczna'  # Zastąp 'SOR' odpowiednią nazwą wydziału
# plot_departmentTryb(department, allSOR, 'D')
#
# #
# # # /////////////////////////////////
# department = 'SOR'  # Zastąp 'SOR' odpowiednią nazwą wydziału
# # plot_department(department, allSOR, 'D')
# plot_departmentsAll(allSOR, 'D')


print('---Podział danych na PRZED i PO Covid')
#
# #Badamy poszczególne oddziały
# allSOR = allSOR.loc[allSOR['Oddział pobyt główny'] == 'Okulistyka']


# Konwertowanie kolumny 'Data przyjęcia do szpitala' na typ daty
allSOR['Data przyjęcia do szpitala'] = pd.to_datetime(allSOR['Data przyjęcia do szpitala'])

# department = 'SOR'  # Zastąp 'SOR' odpowiednią nazwą wydziału dddddddddddddddd
# allSOR = allSOR[allSOR['Oddział pobyt główny'] == department]


print(allSOR.loc[allSOR['Data przyjęcia do szpitala'] == pd.to_datetime('2022-01-02 08:27:00')])

# Filtrowanie danych przed 01.01.2020 / Usuwam COVID19
allPRZED = allSOR.loc[allSOR['Data przyjęcia do szpitala'] < pd.to_datetime('2020-01-01')]
print('allPRZED')
print(allPRZED)
print(allPRZED.describe())

# Filtrowanie danych po 01.01.2022
allPO = allSOR.loc[allSOR['Data przyjęcia do szpitala'] >= pd.to_datetime('2022-01-01')]
print('allPO')
print(allPO.head(10))
print(allPO.describe())

# /////////////////////////////////////////////


# Grupowanie danych według daty i obliczanie sumy pacjentów w danym dniu
editPRZED = allPRZED.groupby(allPRZED['Data przyjęcia do szpitala'].dt.date).size().reset_index(name='liczba pacjentów')

# Konwertowanie kolumny 'Data przyjęcia do szpitala' na typ daty
editPRZED['Data przyjęcia do szpitala'] = pd.to_datetime(editPRZED['Data przyjęcia do szpitala'])

# Zmiana nazwy kolumny na 'data'
editPRZED.rename(columns={'Data przyjęcia do szpitala': 'data'}, inplace=True)

# Wyświetlenie danych po dodaniu kolumny
print(editPRZED.head())

# Wyświetlenie nowego zestawu danych
print(editPRZED)

# Grupowanie danych według daty i obliczanie sumy pacjentów w danym dniu
editPO = allPO.groupby(allPO['Data przyjęcia do szpitala'].dt.date).size().reset_index(name='liczba pacjentów')

# Konwertowanie kolumny 'Data przyjęcia do szpitala' na typ daty
editPO['Data przyjęcia do szpitala'] = pd.to_datetime(editPO['Data przyjęcia do szpitala'])

# Zmiana nazwy kolumny na 'data'
editPO.rename(columns={'Data przyjęcia do szpitala': 'data'}, inplace=True)

# # Wyświetlenie danych po dodaniu kolumny
# print(editPRZED.head())

# Wyświetlenie nowego zestawu danych
print('--editPO--')
print(editPO)
print('--editPRZED--')
print(editPRZED)

# dane tylko z jednego departamentu

# Grupowanie danych według daty i obliczanie sumy pacjentów w danym dniu
editAll = allSOR.groupby(allSOR['Data przyjęcia do szpitala'].dt.date).size().reset_index(name='liczba pacjentów')

print("editAll")
print(editAll)

# Konwertowanie kolumny 'Data przyjęcia do szpitala' na typ daty
editAll['Data przyjęcia do szpitala'] = pd.to_datetime(editAll['Data przyjęcia do szpitala'])

# Zmiana nazwy kolumny na 'data'
editAll.rename(columns={'Data przyjęcia do szpitala': 'data'}, inplace=True)

print('--editAll--')
# print(editAll)

print('----Regresja liniowa----')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor

# make_modelsOld(editPO, 30)
# make_models(editAll, 60)
# make_models(editPRZED,  30)
# make_models(editPRZED, 60)

print('/m2/')

#
editPRZED2 = editPRZED
editPO2 = editPO

print('editPO')
print(editPO[:10])
print('editPO2.iloc[:8]2')
print(editPO2.iloc[:7 + 3])

print('departament')
# print(department)

print('editPO')
print(editPO)

editPRZED = editPRZED
train_df, test_df = train_test_split(editPRZED, test_size=0.1, shuffle=False)
train_df = editPRZED
test_df = editPO
print('train_df')
print(train_df.head())
print('test_df')
print(test_df.head())

print("Wynik :")


# Test obliczeń poniżej
print("Test modelu pzrewidującego 30 dni")

tabKtoryModelTegoDniaUz30 = pd.DataFrame(
    columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
             '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'])

# tabKtoryModelTegoDniaUz10 = pd.DataFrame(
#     columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

# tabKtoryModelTegoDniaUz3 = pd.DataFrame(
#     columns=['1', '2', '3'])

# print()



# train_df2 = test_df.iloc[:29]
# tabKtoryModelTegoDniaUz = make_models2(train_df, train_df2, test_df.iloc[30:59], 1, tabKtoryModelTegoDniaUz3)


# tabela zbierająca dane na temat typu modelu w danym dniu
#
# iteracja od i=30 do i=320 z krokiem 4
for i in range(40, 220, 50):
#
    # ramka danych do trenowania modelu na 30 dni przed okresem, który chcemy przewidzieć
    train_df2 = test_df.iloc[(i - 40):(i - 1)]
#
    # zebranie danych i trenowanie modelu
    # predict3(train_df2, test_df.iloc[i:i + 4])
    # predictSOR3(train_df2, test_df.iloc[i:i + 4])
    # predictSOR10(train_df2, test_df.iloc[i:i + 11])
    # tabKtoryModelTegoDniaUz30 = make_models2(train_df, train_df2, test_df.iloc[i:i + 31], 1, tabKtoryModelTegoDniaUz30)
    # predictSOR30(train_df2, test_df.iloc[i:i + 31])
    # tabKtoryModelTegoDniaUz3 = make_models2(train_df, train_df2, test_df.iloc[i:i + 4], 1, tabKtoryModelTegoDniaUz3)
    # tabKtoryModelTegoDniaUz10 = make_models2(train_df, train_df2, test_df.iloc[i:i + 11], 1, tabKtoryModelTegoDniaUz10)
    tabKtoryModelTegoDniaUz30 = make_models(editPRZED, train_df2, test_df.iloc[i:i + 31], 1, tabKtoryModelTegoDniaUz30)

    # predictChem3(train_df2, test_df.iloc[i:i + 4])
    # predictChem10(train_df2, test_df.iloc[i:i + 11])
    # predictChem30(train_df2, test_df.iloc[i:i + 31])

    predict3(train_df, test_df.iloc[i:i + 4])

    print("i")
    print(i)
#     # print("tabKtoryModelTegoDniaUz3")
#     # print(tabKtoryModelTegoDniaUz3)
# #
#
print("Koniec")
print("tabKtoryModelTegoDniaUz")
print(tabKtoryModelTegoDniaUz30)



# Tworzenie ramki danych (DataFrame) z danymi
tabModelWykres = pd.DataFrame(tabKtoryModelTegoDniaUz30)
# Przetwarzanie danych
model_counts = {day: Counter(tabModelWykres[day]) for day in tabModelWykres.columns}

# Tworzenie wykresu punktowego
fig, ax = plt.subplots(figsize=(10, 6))

models = tabModelWykres.stack().unique()

for model in models:
    counts = [model_count[model] for model_count in model_counts.values()]
    ax.plot(list(model_counts.keys()), counts, marker='o', label=model)

ax.set_xlabel('Dzień')
ax.set_ylabel('Liczba wystąpień')
ax.set_title('Liczba wystąpień poszczególnych modeli dla każdego dnia')
ax.legend()

plt.show()

print("Test dokładności modeli")

#
# tabDokladnosci30 = pd.DataFrame(
#     columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
#              '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'])
# #
# tabDokladnosciOs30 = pd.DataFrame(
#     columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
#              '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'])
#
# tabDokladnosci10 = pd.DataFrame(
#     columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
#
# tabDokladnosciOs10 = pd.DataFrame(
#     columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

tabDokladnosci3 = pd.DataFrame(
    columns=['1', '2', '3'])
tabDokladnosciOs3 = pd.DataFrame(
    columns=['1', '2', '3'])


# for i in range(30, 220, 4):
for i in range(30, 330, 4):
    # ramka danych do trenowania modelu na 30 dni przed okresem, który chcemy przewidzieć
    train_df2 = test_df.iloc[(i - 40):(i - 1)]

    # tabP = predictSOR3(train_df2, test_df.iloc[i:i + 4])
    # tabP = predictSOR10(train_df2, test_df.iloc[i:i + 11])
    # tabP = predictSOR30(train_df2, test_df.iloc[i:i + 31])
    # tabP = predict3(train_df, test_df.iloc[i:i + 4])
    # tabDokladnosci3.loc[len(tabDokladnosci3)] = tabP['procBledu'].tolist()
    # tabDokladnosci10.loc[len(tabDokladnosci10)] = tabP['procBledu'].tolist()
    # tabDokladnosci3.loc[len(tabDokladnosci3)] = tabP['procBledu'].tolist()

    # tabDokladnosciOs3.loc[len(tabDokladnosciOs3)] = tabP['blad'].tolist()
    # tabDokladnosciOs3.loc[len(tabDokladnosciOs3)] = tabP['blad'].tolist()


    print("i")
    print(i)
    # print(tabDokladnosci3)
    # print("tabKtoryModelTegoDniaUz3")
    # print(tabKtoryModelTegoDniaUz3)

df = pd.DataFrame(tabDokladnosci3)

print('błąd procentowy')
# Stworzenie nowej tabeli 'średnia' dla każdego dnia
result_table1 = pd.DataFrame({
    'średnia': df.mean(),
    'odchylenie_standardowe': df.std()
})
print(result_table1)
result_table1.to_csv('plikProc.txt', sep='\t', index_label='dzien')



print('błąd w liczbie os')
df = pd.DataFrame(tabDokladnosciOs3)

# Stworzenie nowej tabeli 'średnia' dla każdego dnia
result_table = pd.DataFrame({
    'średnia': df.mean(),
    'odchylenie_standardowe': df.std()
})
# Wyświetlenie nowej tabeli
print(result_table)
# #

result_table.to_csv('plikOS.txt', sep='\t', index_label='dzien')



# ///////////////////////////////////
# print('Szereg czasowy - analiza')
#
# # wczytanie danych
#
# # zmiana formatu daty
# allPRZED.loc[:, 'Data przyjęcia do szpitala'] = pd.to_datetime(allPRZED['Data przyjęcia do szpitala'])
#
# # grupowanie danych według daty i obliczanie sumy pacjentów w danym dniu
# patientsInDay = allPRZED.groupby(pd.Grouper(key='Data przyjęcia do szpitala', freq='M')).size()
#
# # wykres szeregu czasowego
# plt.figure(figsize=(12, 6))
# plt.plot(patientsInDay.index, patientsInDay.values)
# plt.title('Liczba pacjentów w szpitalu')
# plt.xlabel('Data')
# plt.ylabel('Liczba pacjentów')
# plt.show()
#
# # obliczenie transformaty Fouriera
# fft = np.fft.fft(patientsInDay)
# freqs = np.fft.fftfreq(len(patientsInDay))
#
# # wykres widma amplitudowego
# plt.figure(figsize=(12, 6))
# plt.plot(np.abs(freqs), np.abs(fft))
# plt.title('Widmo amplitudowe pacjentów w szpitalu')
# plt.xlabel('Częstotliwość')
# plt.ylabel('Amplituda')
# plt.xlim(0, 0.1)
# plt.show()
#
# from statsmodels.graphics.tsaplots import plot_acf
#
# # Przekształcenie danych na szereg czasowy
# data = pd.Series(patientsInDay.values, index=patientsInDay.index)
#
# # Wykres funkcji autokorelacji
# plot_acf(data)
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.title('Autocorrelation Plot')
# plt.show()
#
# # Przekształcenie danych na szereg czasowy
# data = np.array(patientsInDay.values)
#
# # Obliczenie FFT
# fft = np.fft.fft(data)
# freq = np.fft.fftfreq(len(data))
#
# # Wykres widma amplitudowego
# plt.plot(freq, np.abs(fft))
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.title('Amplitude Spectrum')
# plt.show()





