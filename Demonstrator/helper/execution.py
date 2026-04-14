import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nn.neural_network import create_neural_network

def return_neural_net():
    file = "datensatz/" + "spritzguss.csv"
    data = pd.read_csv((file))

    X = data.iloc[:, :-1] # X enthält immer alle Spalten des Datensatzes außer die letzte Spalte
    y = data.iloc[:, -1] # Y enthält immer die letzte Spalte des Datensatzes
    # -> Man muss nur sichergehen dass im gegebenen Datensatz das Qualitätsmaß in der letzten Spalte ist

    min_max_scaler = MinMaxScaler(feature_range=(0, 1) )
    X_scaled = min_max_scaler.fit_transform(X)

    model = create_neural_network(X_scaled, y, False)
    return model
