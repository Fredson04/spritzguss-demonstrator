import pandas as pd
import shap
from sklearn.model_selection import train_test_split
def shap_explainer2(model): # Gibt eine Erklärung der NN Gewichtung mithilfe von Shap aus
    X_train, X_test, y_train, y_test = get_train_test_split()
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    #shap_values = explainer.shap_values(X_test)
    shap.plots.waterfall(shap_values[0])

def shap_explainer(model, scaler): # Gibt eine Erklärung der NN Gewichtung mithilfe von Shap aus
    X_train, X_test, y_train, y_test = get_train_test_split(scaler)
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    #shap_values = explainer.shap_values(X_test)
    #shap.plots.waterfall(shap_values[0])
    shap.summary_plot(shap_values, X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
def get_train_test_split(scaler):
    file = "dataset/" + "spritzguss-new.csv"
    data = pd.read_csv((file))

    X = data.iloc[:, :-1] # X enthält immer alle Spalten des Datensatzes außer die letzte Spalte
    y = data.iloc[:, -1] # Y enthält immer die letzte Spalte des Datensatzes
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Aufteilung des Datensatz in 80% Trainings- und 20% Testdaten
    return X_train, X_test, y_train, y_test