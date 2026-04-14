import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def create_neural_network(use_shap=False):
    file = "datensatz/" + "spritzguss.csv"
    data = pd.read_csv((file))

    X = data.iloc[:, :-1] # X enthält immer alle Spalten des Datensatzes außer die letzte Spalte
    y = data.iloc[:, -1] # Y enthält immer die letzte Spalte des Datensatzes
    # -> Man muss nur sichergehen dass im gegebenen Datensatz das Qualitätsmaß in der letzten Spalte ist

    min_max_scaler = MinMaxScaler(feature_range=(0, 1) )
    X_scaled = min_max_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #Aufteilung des Datensatz in 80% Trainings- und 20% Testdaten

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16), # Die jeweilige Anzahl an genutzten Neuronen im jeweiligen Layer
        activation='relu',          # Die genutzte Aktivierungsfunktion
        solver='adam',              # Der Solver kümmert sich um optimierung der Gewichtungen
        alpha=0.0001,               # Stärke der L2 Regularisierung
        max_iter=500,               # Anzahl der Epochen
        random_state=42             # Der genutzte Seed
    )

    model.fit(X_train, y_train) # NN wird mit den Trainingsdaten trainiert

    return model, min_max_scaler