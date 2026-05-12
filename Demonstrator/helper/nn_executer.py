import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

def create_neural_network(scaler, hidden_layers=(64, 32, 16), acti_func='relu', solve_func='adam', max_iterations=500):
    file = "dataset/" + "spritzguss.csv"
    data = pd.read_csv((file))

    X = data.iloc[:, :-1] # X enthält immer alle Spalten des Datensatzes außer die letzte Spalte
    y = data.iloc[:, -1] # Y enthält immer die letzte Spalte des Datensatzes
    X = scaler.transform(X)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Aufteilung des Datensatz in 80% Trainings- und 20% Testdaten

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers, # Die jeweilige Anzahl an genutzten Neuronen im jeweiligen Layer
        activation=acti_func,          # Die genutzte Aktivierungsfunktion
        solver=solve_func,              # Der Solver kümmert sich um optimierung der Gewichtungen
        alpha=0.0001,               # Stärke der L2 Regularisierung
        max_iter=max_iterations,               # Anzahl der Epochen
        #random_state=42             # Der genutzte Seed
    )

    model.fit(X_train, y_train) # NN wird mit den Trainingsdaten trainiert

    y_pred = model.predict(X_test) # Das NN sagt von den Testdaten ausgehend die Qualität hervor
    mse = mean_squared_error(y_test, y_pred) #Mithilfe der Testdaten wird getestet wie sehr sich die tatsächlichen Zielvariablen von den vom NN vorhergesehenen unterscheiden
    perc = accuracy_percentage(y_test, y_pred) #Die Funktion gibt an wie viele der tatsächlichen Zielvariablen und vorhergesagten Zielvariablen identisch sind
    #if(use_shap):
    #    help.shap_explainer(model, X_test) # Siehe die dazugehörige Methode
    
    return model, mse, perc

def accuracy_percentage(y_test, y_pred): #Kalkuliert den Prozentsatz akkurater Vorhersagen
    y_test_np = list(y_test)
    n = len(y_test_np)
    accurate = 0
    for i in range(0, n-1):
        if y_test_np[i] == np.round(y_pred[i]):
            accurate = accurate + 1
    percent = (accurate / n) * 100
    return(percent)
