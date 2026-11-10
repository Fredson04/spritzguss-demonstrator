import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def shap_explainer2(model): # Gibt eine Erklärung der NN Gewichtung mithilfe von Shap aus
    X_train, X_test, y_train, y_test = get_train_test_split()
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    #shap_values = explainer.shap_values(X_test)
    shap.plots.waterfall(shap_values[0])

def shap_explainer2(model, scaler): #Alter code sicherheitshalber beibehalten
    X_train, X_test, y_train, y_test = get_train_test_split(scaler)
    background = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test)
    #shap_values = explainer.shap_values(X_test)
    #shap.plots.waterfall(shap_values[0])
    
    feature_names = [
        "Schmelztemperatur",
        "Werkzeugtemperatur", 
        "Schließkraft",
        "Gegendruck",
        "Einspritzdruck",
        "Schussvolumen"
    ]
    
    #shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)

    #Nur zum Beispiel
    plt.title("Einfluss der Prozessparameter auf die Vorhersage", fontsize=16)
    plt.xlabel("Einfluss auf die Vorhersage (SHAP-Wert)", fontsize=14)
    plt.ylabel("Prozessparameter", fontsize=14)
    plt.text(x=0.05, y=0.95, s="vvbbb", transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()
    
    
def shap_explainer(model, scaler):  # Gibt eine Erklärung der NN Gewichtung mithilfe von Shap aus 
    X_train, X_test, y_train, y_test = get_train_test_split(scaler)
    background = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test)

    feature_names = [
        "Schmelztemperatur",
        "Werkzeugtemperatur", 
        "Schließkraft",
        "Gegendruck",
        "Einspritzdruck",
        "Schussvolumen"
    ]

    fig, (ax_plot, ax_text) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [4, 1]}
    )

    plt.sca(ax_plot)
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                    feature_names=feature_names, show=False)

    ax_plot.set_title("Einfluss der Prozessparameter auf die Vorhersage",
                    fontsize=16, fontweight="bold")
    ax_plot.set_xlabel("Einfluss auf die Vorhersage (SHAP-Wert)", fontsize=12)
    ax_plot.set_ylabel("Prozessparameter", fontsize=12)

    ax_text.axis("off")

    box_style = dict(
        boxstyle="round,pad=0.5",
        facecolor="#f0f0f0",
        edgecolor="#cccccc",
        linewidth=1
    )

    # Unsichtbarer Hintergrund-Text für das Kästchen (voller Inhalt, transparent)
    ax_text.text(
        0.0, 0.95,
        "Was zeigt das Diagramm?\n"
        "Auf Basis vergangener Produktionsdaten sagt ein KI-Modell die\n"
        "Qualität der LED-Linsen für die aktuell gewählten Maschinen-\n"
        "einstellungen vorher. Das Diagramm zeigt, wie stark die einzelnen\n"
        "Prozessparameter die Vorhersage des KI-Modells beeinflussen. Je\n"
        "länger der Balken, desto größer der Einfluss.",
        transform=ax_text.transAxes,
        fontsize=11,
        verticalalignment="top",
        color="none",        # Text unsichtbar – nur das Kästchen zählt
        bbox=box_style
    )

    # Fette Überschrift
    ax_text.text(
        0.0, 0.95, "Was zeigt das Diagramm?",
        transform=ax_text.transAxes,
        fontsize=11, fontweight="bold",
        verticalalignment="top"
    )

    # Normaler Text darunter
    ax_text.text(
        0.0, 0.55,
        "Auf Basis vergangener Produktionsdaten sagt ein KI-Modell die\n"
        "Qualität der LED-Linsen für die aktuell gewählten Maschinen-\n"
        "einstellungen vorher. Das Diagramm zeigt, wie stark die einzelnen\n"
        "Prozessparameter die Vorhersage des KI-Modells beeinflussen. Je\n"
        "länger der Balken, desto größer der Einfluss.",
        transform=ax_text.transAxes,
        fontsize=11,
        verticalalignment="top"
    )
    plt.tight_layout()
    plt.show()
    
def get_train_test_split(scaler):
    file = "dataset/" + "spritzguss-new.csv"
    data = pd.read_csv((file))

    X = data.iloc[:, :-1] # X enthält immer alle Spalten des Datensatzes außer die letzte Spalte
    y = data.iloc[:, -1] # Y enthält immer die letzte Spalte des Datensatzes
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Aufteilung des Datensatz in 80% Trainings- und 20% Testdaten
    return X_train, X_test, y_train, y_test