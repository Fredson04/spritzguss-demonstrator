import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import shap
from matplotlib.animation import FuncAnimation
from IPython import display


def initialize_population(X, pop_size = 30): # Initialisiere ein Array X.size * pop_size gefüllt mit zufälligen Werten
    mean = X.mean(axis=0)
    particles = np.random.normal(loc=mean, size=(pop_size, X.shape[1])) #Die Zufälligen Werte sind normalverteilt mit der Mitte bei dem Durchschnitt von X
    min_max_scaler = MinMaxScaler(feature_range=(0, 1) ) 
    particles = min_max_scaler.fit_transform(particles) # Skaliert die Partikel auf eine Spanne zwischen 0 und 1
    return particles

def initialize_max_population(X, pop_size = 30): #Initialisiere ein Array X.size * pop_size gefüllt mit 1en, also dem Maximum bei min max Skalierung mit Feature Range 0, 1
    particles = np.ones((pop_size, X.shape[1]))
    return particles

def plot_scores(scores, title="Plot der besten Qualität", y="Zielvariabel", x="Anzahl der Iterationen"): # Plottet jeden Wert innerhalb von scores
    plt.plot(scores)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()
    plt.close()

def plot_animated_scores(scores, title="Plot der besten Qualität", y="Zielvariabel", x="Iteration"):
    plt.ion()
    fig, ax = plt.subplots()
    x = []
    y = []
    for i in range(len(scores)):
        x.append(i)
        y.append(scores[i])

        ax.clear()
        ax.plot(x, y)

        plt.draw()
        plt.pause(0.2)

    plt.ioff()
    plt.show()

def shap_explainer(model, X_test): # Gibt eine Erklärung der NN Gewichtung mithilfe von Shap aus
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    print(shap_values)
    shap.plots.waterfall(shap_values[0])