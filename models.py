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
models.py

funkcje służace do analizy danych i obliczeń, oraz wstępna wersja modelu

"""


def make_models(train_df, train_df2, test_df, window_size, tabKtoryModelTegoDniaUz):
    # Przygotowanie danych
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    train_data = train_df[['data', 'liczba pacjentów']].copy()
    train_data['data'] = pd.to_datetime(train_data['data'])
    train_data = train_data.set_index('data')
    train_data = train_data.sort_index()

    test_data = test_df[['data', 'liczba pacjentów']].copy()
    test_data['data'] = pd.to_datetime(test_data['data'])
    test_data = test_data.set_index('data')
    test_data = test_data.sort_index()

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

    initial_rows = X_test.shape[0]
    X_test = X_test.dropna()
    # X_test = X_test.fillna(0)
    y_test = y_test.loc[X_test.index]
    remaining_rows = X_test.shape[0]
    removed_rows = initial_rows - remaining_rows
    # print(f'Liczba usuniętych wierszy: {removed_rows}')
    # # print(X_train.isna().any())
    # # print(X_test.isna().any())

    # Trenowanie modeli
    lr_model = LinearRegression()
    # lr_model.fit(X_train, y_train)

    dt_model = DecisionTreeRegressor()

    # Używanie GridSearchCV do optymalizacji hiperparametrów dla drzewa decyzyjnego
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 10], 'min_samples_split': [2, 3, 4, 5, 10, 20, 30, 40]}
    grid_search = GridSearchCV(dt_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f'Best parameters: {grid_search.best_params_}')

    dt_model = grid_search.best_estimator_
    # dt_model.fit(X_train, y_train)

    # nn_model = MLPRegressor(max_iter=1000, random_state=0, hidden_layer_sizes=(100,), activation='tanh', alpha=0.0003,
    #                         solver='lbfgs')
    nn_model = MLPRegressor(max_iter=2000, random_state=0, hidden_layer_sizes=(200,), activation='logistic',
                            alpha=0.0003,
                            solver='lbfgs')

    # param_grid = {
    #     'max_iter': [1000, 2000, 900, 1100],
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    #     'activation': ['logistic', 'tanh', 'relu'],
    #     'solver': ['lbfgs'],
    #     'alpha': [0.0001, 0.0002, 0.0003, 0.0004]
    # }
    #
    # grid_search_nn = GridSearchCV(nn_model, param_grid, cv=5, n_jobs=-1)
    # grid_search_nn.fit(X_train, y_train)
    # nn_model = grid_search_nn.best_estimator_
    # print(f'Best parameters for Neural Network: {grid_search_nn.best_params_}')
    # nn_model_best_estimator_ = grid_search_nn.best_estimator_

    # nn_model.fit(X_train, y_train)

    print("X_test")
    print(X_test)

    # # Zastosowanie modelu
    # lr_predictions = lr_model.predict(X_test)
    # dt_predictions = dt_model.predict(X_test)
    # nn_predictions = nn_model.predict(X_test)
    #
    # lr_mse = mean_squared_error(y_test, lr_predictions)
    # dt_mse = mean_squared_error(y_test, dt_predictions)
    # nn_mse = mean_squared_error(y_test, nn_predictions)
    #
    # print(f'Linear Regression MSE: {lr_mse}')
    # # accuracy = accuracy_score(y_test, lr_predictions)
    # # print("Accuracy:", accuracy)
    # print(f'Decision Tree MSE: {dt_mse}')
    # print(f'Neural Network MSE: {nn_mse}')
    #
    # # Ocena modeli za pomocą kroswalidacji
    # models = []
    # models.append(('Linear Regression', LinearRegression()))
    # models.append(('Decision Tree', DecisionTreeRegressor()))
    # models.append(('Neural Network', MLPRegressor(random_state=0)))
    #
    # print("y_test")
    # print(y_test)
    #
    # # Dokładność modeli
    # lr_accuracy = lr_model.score(X_test, y_test)
    # dt_accuracy = dt_model.score(X_test, y_test)
    # nn_accuracy = nn_model.score(X_test, y_test)
    #
    # # Przygotowanie danych
    # test_data = test_data.reset_index()
    # test_data = test_data.reindex(range(len(X_test)))
    #
    # test_data['predicted_lr'] = lr_predictions
    # test_data['predicted_dt'] = dt_predictions
    # test_data['predicted_nn'] = nn_predictions
    #
    # # Obliczanie oceny dla każdego dnia
    # test_data['lr'] = (test_data['predicted_lr'])
    # test_data['dt'] = (test_data['predicted_dt'])
    # test_data['nn'] = (test_data['predicted_nn'])
    # test_data['real'] = y_test.values
    #
    # # Wyświetlanie oceny dla każdego dnia
    # print("Ocena dla każdego dnia:")
    # print(test_data[['data', 'lr', 'dt', 'nn', 'real']])
    #
    # # Obliczanie R^2 dla każdego modelu
    # r2_lr = r2_score(test_data['real'], test_data['lr'])
    # r2_dt = r2_score(test_data['real'], test_data['dt'])
    # r2_nn = r2_score(test_data['real'], test_data['nn'])
    #
    # print("R^2 dla mse_lr:", r2_lr)
    # print("R^2 dla mse_dt:", r2_dt)
    # print("R^2 dla mse_nn:", r2_nn)
    #
    # # Obliczanie oceny w procentach dla każdego dnia
    # test_data['lr_percent'] = (test_data['predicted_lr'] / test_data['real']) * 100
    # test_data['dt_percent'] = (test_data['predicted_dt'] / test_data['real']) * 100
    # test_data['nn_percent'] = (test_data['predicted_nn'] / test_data['real']) * 100
    #
    # # Wyświetlanie oceny w procentach dla każdego dnia
    # print("Ocena w procentach dla każdego dnia:")
    # print(test_data[['data', 'lr_percent', 'dt_percent', 'nn_percent', 'real']])
    #
    #
    #
    #
    # results = []
    # names = []
    #
    # for name, model in models:
    #     kfold = KFold(n_splits=10, shuffle=True)
    #     cv_results = cross_val_score(model, X_train, y_train, cv=kfold,
    #                                  scoring='neg_mean_squared_error')
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)
    #
    # report_nn = classification_report(y_test.values, nn_predictions)
    # print("report_nn")
    # print(report_nn)
    #
    # print("prównanie")
    # # Wykres porównujący modele
    # plt.clf()
    # plt.boxplot(results)
    # plt.xticks(range(1, len(names) + 1), names)
    # plt.xlabel('Nazwa modelu')
    # plt.ylabel('Wartość MSE')
    # plt.title('Porównanie modeli')
    # plt.show()
    #
    # # Wykres porównujący przewidywane dane do rzeczywistych danych
    # plt.plot(y_test.index, y_test.values, label='Real')
    # plt.plot(y_test.index, lr_predictions, label='Linear Regression')
    # plt.plot(y_test.index, dt_predictions, label='Decision Tree')
    # plt.plot(y_test.index, nn_predictions, label='Neural Network')
    # plt.xlabel('Data', fontsize=20)
    # plt.ylabel('Liczba pacjentów', fontsize=20)
    # plt.title('Porównanie modeli predykcyjnych', fontsize=20)
    # plt.legend()
    # plt.legend(fontsize=20)
    # plt.show()

    # ///////////////////////////////////////
    # ///////////////////////////////////////
    # ///////////////////////////////////////
    # ///////////////////////////////////////
    # ///////////////////////////////////////

    print("Pojedyńcze daty")
    # # Przygotowanie danych

    # predictions = PrettyTable()
    # predictions.field_names = ['data', 'predicted', 'przewidziana liczba', 'prawdziwa liczba', 'błąd[os]',
    #                                  'dokładnosc[%]', 'typ modelu']

    predictions = pd.DataFrame(
        columns=['data', 'predicted', ' przewidziano', 'realna', 'błąd[os]', 'dokładność[%]', 'modelType'])

    train_df2 = train_df2.dropna()
    # train_df2 = train_df2.fillna(0)
    train_data = train_df2[['data', 'liczba pacjentów']].copy()
    train_data['data'] = pd.to_datetime(train_data['data'])
    train_data = train_data.set_index('data')
    train_data = train_data.sort_index()
    # Tworzenie okienkowania
    for i in range(1, window_size + 1):
        train_data[f'lag_{i}'] = train_data['liczba pacjentów'].shift(i)
    # Usuwanie wierszy z brakującymi danymi
    train_data = train_data.dropna()
    train_data = train_data.fillna(0)
    X_train = train_data.drop('liczba pacjentów', axis=1)
    y_train = train_data['liczba pacjentów']

    print("X_train")
    print(X_train)

    # Iteracja po każdym wierszu
    for index, row in X_test.iterrows():

        lr_model.fit(X_train, y_train)
        dt_model.fit(X_train, y_train)
        nn_model.fit(X_train, y_train)

        # Przekształcenie wiersza danych w tablicę 2D
        X_test_one_day = row.values.reshape(1, -1)

        # Przewidywanie wartości dla jednego dnia
        lr_prediction = lr_model.predict(X_test_one_day)
        dt_prediction = dt_model.predict(X_test_one_day)
        nn_prediction = nn_model.predict(X_test_one_day)

        # Obliczanie dokładności w procentach
        lr_percent = (lr_prediction / y_test.loc[index]) * 100
        dt_percent = (dt_prediction / y_test.loc[index]) * 100
        nn_percent = (nn_prediction / y_test.loc[index]) * 100

        # Obliczenie błędu średniokwadratowego dla jednego dnia
        lr_mse = mean_squared_error([y_test.loc[index]], [lr_prediction])
        dt_mse = mean_squared_error([y_test.loc[index]], [dt_prediction])
        nn_mse = mean_squared_error([y_test.loc[index]], [nn_prediction])

        # Wyświetlenie wyników
        print("----------")
        print(f"Prawdziwa wartość: {y_test.loc[index]:.2f}")
        print(f"Przewidywana liczba osób dla dnia {index.date()}:")
        print(f"Regresja liniowa: {lr_prediction[0]:.2f}")
        print(f"Drzewo decyzyjne: {dt_prediction[0]:.2f}")
        print(f"Sztuczna sieć neuronowa: {nn_prediction[0]:.2f}")
        print(f"---Błąd średniokwadratowy dla dnia {index.date()}:")
        print(f"Regresja liniowa: {lr_mse:.2f}")
        print(f"Drzewo decyzyjne: {dt_mse:.2f}")
        print(f"Sztuczna sieć neuronowa: {nn_mse:.2f}")
        print(f"---Ocena w procentach dla dnia {index.date()}:")
        print(f"Regresja liniowa: {lr_percent[0]:.2f} %")
        print(f"Drzewo decyzyjne: {dt_percent[0]:.2f} %")
        print(f"Sztuczna sieć neuronowa: {nn_percent[0]:.2f} %")

        # Wybór wartości, która jest najbliższa 100%
        lr_p = abs(100 - lr_percent)
        dt_p = abs(100 - dt_percent)
        nn_p = abs(100 - nn_percent)

        # Znalezienie indeksu wartości z najmniejszą różnicą
        min_index = np.argmin([lr_p, dt_p, nn_p])
        print("min_index")
        print(min_index)

        # Wybór wartości z najmniejszą różnicą
        if min_index == 0:
            closest_value = lr_prediction
            modelType = "linear regresion"
            accuracyPercent = f"{lr_percent}%"
        elif min_index == 1:
            closest_value = dt_prediction
            modelType = "decision tree"
            accuracyPercent = f"{dt_percent}%"
        else:
            closest_value = nn_prediction
            modelType = "neural network"
            accuracyPercent = f"{nn_percent}%"

        r = np.round(closest_value)
        diff = r - y_test.loc[index]
        predictions.loc[len(predictions)] = [index, closest_value, r, y_test.loc[index], diff, accuracyPercent,
                                             modelType]

    # # Używanie funkcji tabulate do wygenerowania tabeli
    # table = tabulate(predictions.drop('predicted', axis=1), headers='keys', tablefmt='pretty', showindex=False)

    print("Metoda porównująca wynik do realnych danych")
    print(predictions.drop('predicted', axis=1))
    # print(predictions['predicted'])
    print(predictions['modelType'].value_counts())

    # Utworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    # Dodanie danych do wykresu
    ax.plot(y_test.index, y_test.values, label='Real', linewidth=2)
    ax.plot(y_test.index, predictions['predicted'], label='Predicted', linewidth=2)
    ax.set_title('Model wybierający najbliższą wartość do realnej', fontsize=20)
    ax.set_xlabel('Data', fontsize=20)
    ax.set_ylabel('Liczba pacjentów', fontsize=20)
    ax.legend(fontsize=16)
    plt.show()

    model_type = predictions['modelType']

    # tabKtoryModelTegoDniaUz.loc[len(tabKtoryModelTegoDniaUz)] = [model_type[0], model_type[1], model_type[2]]
    #
    # tabKtoryModelTegoDniaUz.loc[len(tabKtoryModelTegoDniaUz)] = [model_type[0], model_type[1], model_type[2], model_type[3], model_type[4], model_type[5], model_type[6],
    #                                                              model_type[7], model_type[8], model_type[9]]

    tabKtoryModelTegoDniaUz.loc[len(tabKtoryModelTegoDniaUz)] = [model_type[0], model_type[1], model_type[2],
                                                                 model_type[3], model_type[4], model_type[5],
                                                                 model_type[6], model_type[7], model_type[8],
                                                                 model_type[9], model_type[10],
                                                                 model_type[11], model_type[12], model_type[13],
                                                                 model_type[14], model_type[15], model_type[16],
                                                                 model_type[17], model_type[18], model_type[19],
                                                                 model_type[20], model_type[21], model_type[22],
                                                                 model_type[23], model_type[24], model_type[25],
                                                                 model_type[26], model_type[27], model_type[28],
                                                                 model_type[29]]

    # Dodanie zwracanych prognoz
    return tabKtoryModelTegoDniaUz


def plot_departmentTryb(department, allSOR, dm):
    department_data = allSOR[allSOR['Oddział pobyt główny'] == department]
    patients_in_day = department_data.groupby(pd.Grouper(key='Data przyjęcia do szpitala', freq=dm)).size()

    unique_modes = department_data['Tryb przyjęcia'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_modes)))
    #
    plt.figure(figsize=(10, 6))

    for i, mode in enumerate(unique_modes):
        mode_data = department_data[department_data['Tryb przyjęcia'] == mode]
        mode_patients_in_day = mode_data.groupby(pd.Grouper(key='Data przyjęcia do szpitala', freq='D')).size()
        plt.plot(mode_patients_in_day.index, mode_patients_in_day.values, label=mode)

    plt.title(department, fontsize=24)
    plt.xlabel('Data', fontsize=24)
    plt.ylabel('Liczba pacjentów', fontsize=24)
    plt.title('Liczba pacjentów na oddziałach', fontsize=24)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', labelsize=16)  # Dostosuj rozmiar czcionki na osiach
    plt.show()


def plot_department(department, allSOR, dm):
    department_data = allSOR[allSOR['Oddział pobyt główny'] == department]
    patients_in_day = department_data.groupby(pd.Grouper(key='Data przyjęcia do szpitala', freq=dm)).size()

    plt.figure(figsize=(10, 6))

    plt.plot(patients_in_day.index, patients_in_day.values)

    plt.title(department)
    plt.xlabel('Data')
    plt.ylabel('Liczba pacjentów')
    # plt.show()


def plot_departmentsAll(allSOR, dm):
    unique_departments = allSOR['Oddział pobyt główny'].unique()
    num_departments = len(unique_departments)

    num_rows = (num_departments + 3) // 4  # Obliczenie liczby wierszy w subplotach

    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows), sharex=True, sharey=True)

    for i, department in enumerate(unique_departments):
        row = i // 4  # Numer wiersza dla danego oddziału
        col = i % 4  # Numer kolumny dla danego oddziału

        department_data = allSOR[allSOR['Oddział pobyt główny'] == department]
        patients_in_day = department_data.groupby(pd.Grouper(key='Data przyjęcia do szpitala', freq=dm)).size()

        axes[row, col].plot(patients_in_day.index, patients_in_day.values)
        axes[row, col].set_title(department, fontsize=16)
        axes[row, col].set_ylabel('Liczba pacjentów', fontsize=12)

        # if row == num_rows - 1 and col == 0:
        # axes[row, col].set_xlabel('Data')
        # axes[row, col].xaxis.set_major_locator(
        #     mdates.YearLocator(1))  # Wybierz co który rok mają być widoczne etykiety
        # axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format daty
    # else:
    #     axes[row, col].set_xtikclabels([])  # Usunięcie etykiet osi x dla subplotów na niższych wierszach


    # Ukrycie pustych subplotów
    for i in range(num_departments, num_rows * 4):
        row = i // 4
        col = i % 4
        fig.delaxes(axes[row, col])

    fig.suptitle('Liczba pacjentów dziennie', fontsize=24)

    plt.tight_layout()
    plt.show()
