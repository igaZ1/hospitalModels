from pip._internal.utils.misc import tabulate
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, classification_report, \
    r2_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates

"""
modelSOR.py

modele predykcyjne dla oddziału SOR przewidujące liczbę pacjentów na 3, 10 i 30 dni.

"""


def predictSOR3(train_df2, test_df):
    """
    About method predictSOR3: metoda pzrewidująca liczbę pacjentów dla oddziału Chemioterapia na 3 dni
    :param train_df2: zestaw danych treningowych
    :param test_df: zestaw dat, dla których model ma przewidzieć liczbę pacjentów
    :return: pred - tabela wyników
    """
    # Przygotowanie danych
    test_data = test_df[['data', 'liczba pacjentów']].copy()
    test_data['data'] = pd.to_datetime(test_data['data'])
    test_data = test_data.set_index('data')
    test_data = test_data.sort_index()

    train_df2 = train_df2.dropna()
    train_data = train_df2[['data', 'liczba pacjentów']].copy()
    train_data['data'] = pd.to_datetime(train_data['data'])
    train_data = train_data.set_index('data')
    train_data = train_data.sort_index()

    window_size = 1
    # Tworzenie okienkowania
    for i in range(1, window_size + 1):
        train_data[f'lag_{i}'] = train_data['liczba pacjentów'].shift(i)
        test_data[f'lag_{i}'] = test_data['liczba pacjentów'].shift(i)

    # Usuwanie wierszy z brakującymi danymi
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    X_train = train_data.drop('liczba pacjentów', axis=1)
    y_train = train_data['liczba pacjentów']

    X_test = test_data.drop('liczba pacjentów', axis=1)
    y_test = test_data['liczba pacjentów']

    # print(X_train)
    # print(y_train)

    # Trenowanie modeli
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    dt_model = DecisionTreeRegressor()
    # Używanie GridSearchCV do optymalizacji hiperparametrów dla drzewa decyzyjnego
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 10], 'min_samples_split': [2, 3, 4, 5, 10, 20, 30, 40]}
    grid_search = GridSearchCV(dt_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    dt_model = grid_search.best_estimator_
    dt_model.fit(X_train, y_train)

    # nn_model = MLPRegressor(max_iter=1000, random_state=0, hidden_layer_sizes=(100,), activation='tanh', alpha=0.0003,
    #                         solver='lbfgs')
    nn_model = MLPRegressor(max_iter=2000, random_state=0, hidden_layer_sizes=(200,), activation='logistic',
                            alpha=0.0003,
                            solver='lbfgs')

    nn_model.fit(X_train, y_train)

    print("Pojedyńcze daty")
    # # Przygotowanie danych

    # predictions = pd.DataFrame(columns=['data', 'predicted', 'predictedRound', 'modelType', 'real', 'błąd','accuracyPercent'])

    # pred = pd.DataFrame(columns=['data', 'przewidzianaLiczbaPacjentow'])

    pred = pd.DataFrame(columns=['data', 'przewidzianaLiczbaPacjentow', 'blad', 'procBledu'])

    i = 0
    # Iteracja po każdym wierszu w X_test
    for index, row in X_test.iterrows():
        i = i + 1

        # trenowanie modelu

        # Przekształcenie wiersza danych w tablicę 2D
        X_test_one_day = row.values.reshape(1, -1)

        # Przewidywanie wartości dla jednego dnia
        if i in {1, 2, 3}:
            lr_model.fit(X_train, y_train)
            prediction = np.round(lr_model.predict(X_test_one_day))
        # elif i in {2, 3}:
        #     dt_model.fit(X_train, y_train)
        #     prediction = np.round(dt_model.predict(X_test_one_day))
        else:
            nn_model.fit(X_train, y_train)
            prediction = np.round(nn_model.predict(X_test_one_day))

        # Liczenie błędu
        error = (y_test.loc[index] - prediction)
        if y_test.loc[index] != 0:
            percent_error = (error / y_test.loc[index]) * 100
        else:
            percent_error = 0
        # Unikamy dzielenia przez zero

        # Dodanie do ramki danych
        pred.loc[len(pred)] = [index, prediction, error, percent_error]

    import matplotlib.dates as mdates

    # Utworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    # Dodanie danych do wykresu
    ax.plot(y_test.index, y_test.values, label='Real', linewidth=2)
    ax.plot(y_test.index, pred['przewidzianaLiczbaPacjentow'], label='Predicted', linewidth=2)
    ax.set_ylim(0)

    # Ustawienie formatu daty na osi X (bez godzin)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.set_title('SOR Porównanie modelu', fontsize=20)
    ax.set_xlabel('Data', fontsize=20)
    ax.set_ylabel('Liczba pacjentów', fontsize=20)
    ax.legend(fontsize=16)
    plt.show()

    print("pred")
    print(pred)

    return pred


def predictSOR10(train_df2, test_df):
    """
    About method predictSOR10: metoda ppzrewidująca liczbę pacjentów dla oddziału SOR na 10 dni
    :param train_df2: zestaw danych treningowych
    :param test_df: zestaw dat, dla których model ma przewidzieć liczbę pacjentów
    :return: pred - tabela wyników
    """
    # Przygotowanie danych
    test_data = test_df[['data', 'liczba pacjentów']].copy()
    test_data['data'] = pd.to_datetime(test_data['data'])
    test_data = test_data.set_index('data')
    test_data = test_data.sort_index()

    train_df2 = train_df2.dropna()
    train_data = train_df2[['data', 'liczba pacjentów']].copy()
    train_data['data'] = pd.to_datetime(train_data['data'])
    train_data = train_data.set_index('data')
    train_data = train_data.sort_index()

    # # Przygotowanie danych

    window_size = 1
    # Tworzenie okienkowania
    for i in range(1, window_size + 1):
        test_data[f'lag_{i}'] = test_data['liczba pacjentów'].shift(i)
        train_data[f'lag_{i}'] = train_data['liczba pacjentów'].shift(i)

    # Usuwanie wierszy z brakującymi danymi
    test_data = test_data.dropna()
    train_data = train_data.dropna()

    X_test = test_data.drop('liczba pacjentów', axis=1)
    y_test = test_data['liczba pacjentów']

    X_train = train_data.drop('liczba pacjentów', axis=1)
    y_train = train_data['liczba pacjentów']

    print("X_train")
    print(X_train)
    # print(X_train)
    # print(y_train)

    # Trenowanie modeli
    lr_model = LinearRegression()

    dt_model = DecisionTreeRegressor()
    # Używanie GridSearchCV do optymalizacji hiperparametrów dla drzewa decyzyjnego
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 10], 'min_samples_split': [2, 3, 4, 5, 10, 20, 30, 40]}
    grid_search = GridSearchCV(dt_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    dt_model = grid_search.best_estimator_

    # nn_model = MLPRegressor(max_iter=1000, random_state=0, hidden_layer_sizes=(100,), activation='tanh', alpha=0.0003,
    #                         solver='lbfgs')
    nn_model = MLPRegressor(max_iter=2000, random_state=0, hidden_layer_sizes=(200,), activation='logistic',
                            alpha=0.0003,
                            solver='lbfgs')

    print("Pojedyńcze daty")
    # predictions = pd.DataFrame(columns=['data', 'predicted', 'predictedRound', 'modelType', 'real', 'błąd','accuracyPercent'])

    # pred = pd.DataFrame(columns=['data', 'przewidzianaLiczbaPacjentow'])
    pred = pd.DataFrame(columns=['data', 'przewidzianaLiczbaPacjentow', 'blad', 'procBledu'])

    i = 0
    # Iteracja po każdym wierszu w X_test
    for index, row in X_test.iterrows():
        i = i + 1

        # Przekształcenie wiersza danych w tablicę 2D
        X_test_one_day = row.values.reshape(1, -1)

        # Przewidywanie wartości dla jednego dnia
        if i in {2}:
            nn_model.fit(X_train, y_train)
            prediction = np.round(nn_model.predict(X_test_one_day))  # sieci
        else:  # 1, 3, 4, 5, 6, 7, 8, 9, 10
            lr_model.fit(X_train, y_train)
            prediction = np.round(lr_model.predict(X_test_one_day))  # regresja

        # elif i in {:
        #     dt_model.fit(X_train, y_train)
        #     prediction = np.round(dt_model.predict(X_test_one_day))

        # Liczenie błędu
        error = (y_test.loc[index] - prediction)
        if y_test.loc[index] != 0:
            percent_error = (error / y_test.loc[index]) * 100
        else:
            percent_error = 0
        # Unikamy dzielenia przez zero

        # Dodanie do ramki danych
        pred.loc[len(pred)] = [index, prediction, error, percent_error]
        # pred.loc[len(pred)] = [index, prediction]

    print("pred")
    print(pred)
    #
    # Utworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    # Dodanie danych do wykresu
    ax.plot(y_test.index, y_test.values, label='Real', linewidth=2)
    ax.plot(y_test.index, pred['przewidzianaLiczbaPacjentow'], label='Predicted', linewidth=2)
    ax.set_title('SOR Porównanie modelu', fontsize=20)
    ax.set_xlabel('Data', fontsize=20)
    ax.set_ylim(0)
    ax.set_ylabel('Liczba pacjentów', fontsize=20)
    ax.legend(fontsize=16)
    plt.show()

    return pred


def predictSOR30(train_df2, test_df):
    """
    About method predictSOR30: metoda ppzrewidująca liczbę pacjentów dla oddziału SOR na 30 dni
    :param train_df2: zestaw danych treningowych
    :param test_df: zestaw dat, dla których model ma przewidzieć liczbę pacjentów
    :return: pred - tabela wyników
    """
    # Przygotowanie danych
    test_data = test_df[['data', 'liczba pacjentów']].copy()
    test_data['data'] = pd.to_datetime(test_data['data'])
    test_data = test_data.set_index('data')
    test_data = test_data.sort_index()

    train_df2 = train_df2.dropna()
    train_data = train_df2[['data', 'liczba pacjentów']].copy()
    train_data['data'] = pd.to_datetime(train_data['data'])
    train_data = train_data.set_index('data')
    train_data = train_data.sort_index()

    window_size = 1
    # Tworzenie okienkowania
    for i in range(1, window_size + 1):
        train_data[f'lag_{i}'] = train_data['liczba pacjentów'].shift(i)
        test_data[f'lag_{i}'] = test_data['liczba pacjentów'].shift(i)

    # Usuwanie wierszy z brakującymi danymi
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    X_train = train_data.drop('liczba pacjentów', axis=1)
    y_train = train_data['liczba pacjentów']

    X_test = test_data.drop('liczba pacjentów', axis=1)
    y_test = test_data['liczba pacjentów']

    print(X_train)
    print(y_train)

    # Trenowanie modeli
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    dt_model = DecisionTreeRegressor()
    # Używanie GridSearchCV do optymalizacji hiperparametrów dla drzewa decyzyjnego
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 10], 'min_samples_split': [2, 3, 4, 5, 10, 20, 30, 40]}
    grid_search = GridSearchCV(dt_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    dt_model = grid_search.best_estimator_
    dt_model.fit(X_train, y_train)

    # nn_model = MLPRegressor(max_iter=1000, random_state=0, hidden_layer_sizes=(100,), activation='tanh', alpha=0.0003,
    #                         solver='lbfgs')
    nn_model = MLPRegressor(max_iter=2000, random_state=0, hidden_layer_sizes=(200,), activation='logistic',
                            alpha=0.0003,
                            solver='lbfgs')

    nn_model.fit(X_train, y_train)

    print("Pojedyńcze daty")
    # # Przygotowanie danych

    # predictions = pd.DataFrame(columns=['data', 'predicted', 'predictedRound', 'modelType', 'real', 'błąd','accuracyPercent'])

    # Tworzenie okienkowania

    # pred = pd.DataFrame(columns=['data', 'przewidzianaLiczbaPacjentow'])

    pred = pd.DataFrame(columns=['data', 'przewidzianaLiczbaPacjentow', 'blad', 'procBledu'])

    i = 0
    # Iteracja po każdym wierszu w X_test
    for index, row in X_test.iterrows():
        i = i + 1

        # Przekształcenie wiersza danych w tablicę 2D
        X_test_one_day = row.values.reshape(1, -1)

        # Przewidywanie wartości dla jednego dnia
        if i in {4, 8, 12, 20}:  # sieci
            nn_model.fit(X_train, y_train)
            prediction = np.round(nn_model.predict(X_test_one_day))
        elif i in {7, 15, 17, 19}:
            dt_model.fit(X_train, y_train)
            prediction = np.round(dt_model.predict(X_test_one_day))
        else:  # regresja
            lr_model.fit(X_train, y_train)
            prediction = np.round(lr_model.predict(X_test_one_day))

        # Liczenie błędu
        error = (y_test.loc[index] - prediction)

        percent_error = (error / y_test.loc[index]) * 100

        # Dodanie do ramki danych
        pred.loc[len(pred)] = [index, prediction, error, percent_error]

    print("pred")
    print(pred)

    # Utworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    # Dodanie danych do wykresu
    ax.plot(y_test.index, y_test.values, label='Real', linewidth=2)
    ax.plot(y_test.index, pred['przewidzianaLiczbaPacjentow'], label='Predicted', linewidth=2)
    ax.set_title('SOR Porównanie modelu', fontsize=20)
    ax.set_xlabel('Data', fontsize=20)
    ax.set_ylabel('Liczba pacjentów', fontsize=20)
    ax.legend(fontsize=16)
    plt.show()

    return pred
