import pickle

import customtkinter
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from algorithm import ga, sa
import algorithm.pso as pso
from helper.help_functions import create_scaler, plot_scores
from PIL import Image
from CTkToolTip import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

from helper.nn_executer import create_neural_network

# Konstanten
STEPS=1000 # Nur in dem Wertebereich der Schritte können den Slidern Werte zugeordnet werden, also muss die Schrittmenge hoch (oder nonexistent) sein damit die Qualität der optimierten Parameter erreicht werden kann
CELSIUS="°C"
SECONDS="s"
NEWTON="n"
NEWTONMETER="N\u22C5m"
BAR="Bar"
CM="cm"
CM3="cm\u00B3"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1300x850")
        self.title("Spritzguss Demonstrator")
        
        self.min_max_scaler = create_scaler()
        self.model = pickle.load(open("nn/neural-net.sav", 'rb'))
        self.scores = []
        
        def update_label1(value):
            self.amount1label.configure(text=(f"{value:.4f}", CELSIUS))
        def update_label2(value):
            self.amount2label.configure(text=(f"{value:.4f}", CELSIUS))
        def update_label3(value):
            self.amount3label.configure(text=(f"{value:.4f}", SECONDS))
        def update_label4(value):
            self.amount4label.configure(text=(f"{value:.4f}", SECONDS))
        def update_label5(value):
            self.amount5label.configure(text=(f"{value:.4f}", SECONDS))
        def update_label6(value):
            self.amount6label.configure(text=(f"{value:.4f}", NEWTON))
        def update_label7(value):
            self.amount7label.configure(text=(f"{value:.4f}", NEWTON))
        def update_label8(value):
            self.amount8label.configure(text=(f"{value:.4f}", NEWTONMETER))
        def update_label9(value):
            self.amount9label.configure(text=(f"{value:.4f}", NEWTONMETER))
        def update_label10(value):
            self.amount10label.configure(text=(f"{value:.4f}", BAR))
        def update_label11(value):
            self.amount11label.configure(text=(f"{value:.4f}", BAR))
        def update_label12(value):
            self.amount12label.configure(text=(f"{value:.4f}", CM))
        def update_label13(value):
            self.amount13label.configure(text=(f"{value:.4f}", CM3))
            
        # Tabs    
            
        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.pack(padx=0, pady=0)

        self.tabview.add("Optimierungsalgorithmus")
        self.tabview.add("Maschinensimulation")
        self.tabview.add("Neuronales Netz")
        self.tabview.set("Maschinensimulation")
        
        # Optimierungsalgorithmus - Tab
        
        #self.optAlgoFrame = customtkinter.CTkFrame(self.tabview.tab("Optimierungsalgorithmus"), width=200, height=200) 
        
        #fig = plot_scores(self.scores)
        #canvas = FigureCanvasTkAgg(fig, self.tabview.tab("Optimierungsalgorithmus"))  # A tk.DrawingArea.
        #canvas.draw()
        #canvas.get_tk_widget().grid(row=0, column=0)  
        
        # Neuronales Netz
        
        self.nnFrame = customtkinter.CTkFrame(self.tabview.tab("Neuronales Netz"), width=200, height=200)
        self.nnFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.layerLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die Größe und Anzahl der Versteckten Schichten an", fg_color="transparent", font=("Arial", 16, "bold") )
        self.layerLabel.grid(row=1, column=0, padx=20, pady=10)
        self.layers_text = customtkinter.CTkEntry(self.nnFrame, placeholder_text="64, 32, 16")
        self.layers_text.grid(row=1, column=1, padx=20)
        self.activLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die zu nutzende Aktivierungsfunktion an", fg_color="transparent", font=("Arial", 16, "bold") )
        self.activLabel.grid(row=2, column=0, padx=20, pady=10)
        self.activVar = customtkinter.StringVar(value="relu")
        self.activOption = customtkinter.CTkOptionMenu(self.nnFrame,values=["relu", "identity", "logistic", "tanh"],
                                         variable=self.activVar)
        self.activOption.grid(row=2, column=1, padx=20, pady=10)
        self.solverLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die zu nutzende Solver Funktion an", fg_color="transparent", font=("Arial", 16, "bold") )
        self.solverLabel.grid(row=3, column=0, padx=20, pady=10)
        self.solverVar = customtkinter.StringVar(value="adam")
        self.solverOption = customtkinter.CTkOptionMenu(self.nnFrame,values=["lbfgs", "sgd", "adam"],
                                         variable=self.solverVar)
        self.solverOption.grid(row=3, column=1, padx=20, pady=10)
        self.iterationsLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die maximale Anzahl an Iterationen an", fg_color="transparent", font=("Arial", 16, "bold") )
        self.iterationsLabel.grid(row=4, column=0, padx=20, pady=10)
        self.iterations_text = customtkinter.CTkEntry(self.nnFrame, placeholder_text="500")
        self.iterations_text.grid(row=4, column=1, padx=20)
        
        self.mseLabel = customtkinter.CTkLabel(self.nnFrame, text="Mean Squared Error des Neuronalen Netzes: ", fg_color="transparent", font=("Arial", 16, "bold") )
        self.mseLabel.grid(row=6, column=0, padx=20, pady=10)
        self.mseValue = customtkinter.CTkLabel(self.nnFrame, text="-", fg_color="transparent", font=("Arial", 16, "bold") )
        self.mseValue.grid(row=6, column=1, padx=20, pady=10)
        
        self.percLabel = customtkinter.CTkLabel(self.nnFrame, text="Akkuratheit des Neuronalen Netzes: ", fg_color="transparent", font=("Arial", 16, "bold") )
        self.percLabel.grid(row=7, column=0, padx=20, pady=10)
        self.percValue = customtkinter.CTkLabel(self.nnFrame, text="-", fg_color="transparent", font=("Arial", 16, "bold") )
        self.percValue.grid(row=7, column=1, padx=20, pady=10)
        
        def nn_button_pressed():
            print("test")
            print(self.activVar.get(), self.solverVar.get(), int(self.iterations_text.get()))
            self.model, mse, perc = create_neural_network(self.min_max_scaler, acti_func=self.activVar.get(), solve_func=self.solverVar.get(), max_iterations=int(self.iterations_text.get()))
            print(mse, perc)
            mse = f"{mse:.4f}"
            self.mseValue.configure(text=mse)
            perc = f"{perc:.4f}%"
            self.percValue.configure(text=perc)
        
        self.createNNButton = customtkinter.CTkButton(self.nnFrame, text="Erstelle Neuronales Netz", command=nn_button_pressed)
        self.createNNButton.grid(row=5, column=1, padx=20)
        
        # Maschinensimulation - Tab
        # Widgets:
        self.parameterLabel = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text="Prozessparameter", fg_color="transparent", font=("Arial", 16, "bold") )
        self.parameterLabel.grid(row=0, column=0, padx=20, pady=10)
        parameter_tooltip_string = "Die hier gelisteten Parameter stellen die Prozessparameter des Industrieprozess der Herstellung von Kunststofflinsen dar"
        self.parameter_tooltip = CTkToolTip(self.parameterLabel, message=parameter_tooltip_string)
        self.parameterAmountLabel = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text="Wert", fg_color="transparent", font=("Arial", 16, "bold") )
        self.parameterAmountLabel.grid(row=0, column=1, padx=20, pady=10)
        self.sliderLabel = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text="Anpassen der Prozessparameter", fg_color="transparent", font=("Arial", 16, "bold") )
        self.sliderLabel.grid(row=0, column=2, padx=20, pady=10)
        slider_label_tooltip_string = "Der Wertebereich jedes Prozessparameters basiert auf dem Wertebereich des jeweiligen Parameters im Kunststofflinsendatensatz. Hierbei weicht der minimal einstellbare Wert jedes Parameters um 50% zum minimalwert im Datensatz ab, und ebenso ist es bei dem maximal einstellbaren Wert."
        self.slider_label_tooltip = CTkToolTip(self.sliderLabel, message=slider_label_tooltip_string)
        self.slider1var = tk.DoubleVar(value=((155.032-81.747)/2)+81.747)
        self.slider1 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=81.747/1.5, to=155.032*1.5, variable=self.slider1var, command=update_label1)
        self.slider1.grid(row=1, column=2, padx=20, pady=10)
        self.slider1label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Temperatur des Polymer vor Injektion in den Guss:", fg_color="transparent")
        self.slider1label.grid(row=1, column=0, padx=20, pady=10)
        self.amount1label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider1var.get()):.4f}", CELSIUS), fg_color="transparent")
        self.amount1label.grid(row=1, column=1, padx=20, pady=10)
        self.slider2var = tk.DoubleVar(value=((82.159-78.409)/2)+78.409)
        self.slider2 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=78.409/1.5, to=82.159*1.5, variable=self.slider2var, command=update_label2)
        self.slider2.grid(row=2, column=2, padx=20, pady=10)
        self.slider2label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Temperatur des Gusses:", fg_color="transparent")
        self.slider2label.grid(row=2, column=0, padx=20, pady=10)
        self.amount2label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider2var.get()):.4f}", CELSIUS), fg_color="transparent")
        self.amount2label.grid(row=2, column=1, padx=20, pady=10)
        self.slider3var = tk.DoubleVar(value=((11.232-6.084)/2)+6.084)
        self.slider3 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=6.084/1.5, to=11.232*1.5, variable=self.slider3var, command=update_label3)
        self.slider3.grid(row=3, column=2, padx=20, pady=10)
        self.slider3label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Zeit um den Guss zu füllen:", fg_color="transparent")
        self.slider3label.grid(row=3, column=0, padx=20, pady=10)
        self.amount3label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider3var.get()):.4f}", SECONDS), fg_color="transparent")
        self.amount3label.grid(row=3, column=1, padx=20, pady=10)
        self.slider4var = tk.DoubleVar(value=((6.61-2.78)/2)+2.78)
        self.slider4 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=2.78/1.5, to=6.61*1.5, variable=self.slider4var, command=update_label4)
        self.slider4.grid(row=4, column=2, padx=20, pady=10)
        self.slider4label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Zeit um das Produkt zu plastizieren:", fg_color="transparent")
        self.slider4label.grid(row=4, column=0, padx=20, pady=10)
        self.amount4label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider4var.get()):.4f}", SECONDS), fg_color="transparent")
        self.amount4label.grid(row=4, column=1, padx=20, pady=10)
        self.slider5var = tk.DoubleVar(value=((75.79-74.78)/2)+74.78)
        self.slider5 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=74.78/1.5, to=75.79*1.5, variable=self.slider5var, command=update_label5)
        self.slider5.grid(row=5, column=2, padx=20, pady=10)
        self.slider5label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Zeit um den Prozess für ein Produkt zu beenden:", fg_color="transparent")
        self.slider5label.grid(row=5, column=0, padx=20, pady=10)
        self.amount5label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider5var.get()):.4f}", SECONDS), fg_color="transparent")
        self.amount5label.grid(row=5, column=1, padx=20, pady=10)
        self.slider6var = tk.DoubleVar(value=((930.6-876.7)/2)+876.7)
        self.slider6 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=876.7/1.5, to=930.6*1.5, variable=self.slider6var, command=update_label6)
        self.slider6.grid(row=6, column=2, padx=20, pady=10)
        self.slider6label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Schließmoment des Gusses:", fg_color="transparent")
        self.slider6label.grid(row=6, column=0, padx=20, pady=10)
        self.amount6label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider6var.get()):.4f}", NEWTON), fg_color="transparent")
        self.amount6label.grid(row=6, column=1, padx=20, pady=10)
        self.slider7var = tk.DoubleVar(value=((946.5-894.8)/2)+894.8)
        self.slider7 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=894.8/1.5, to=946.5*1.5, variable=self.slider7var, command=update_label7)
        self.slider7.grid(row=7, column=2, padx=20, pady=10)
        self.slider7label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Höchstwert des Schließmoment des Gusses:", fg_color="transparent")
        self.slider7label.grid(row=7, column=0, padx=20, pady=10)
        self.amount7label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider7var.get()):.4f}", NEWTON), fg_color="transparent")
        self.amount7label.grid(row=7, column=1, padx=20, pady=10)
        self.slider8var = tk.DoubleVar(value=((130.3-94.2)/2)+94.2)
        self.slider8 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=94.2/1.5, to=130.3*1.5, variable=self.slider8var, command=update_label8)
        self.slider8.grid(row=8, column=2, padx=20, pady=10)
        self.slider8label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Höchstwert des Drehmoments der Einspritzschnecke:", fg_color="transparent")
        self.slider8label.grid(row=8, column=0, padx=20, pady=10)
        self.amount8label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider8var.get()):.4f}", NEWTONMETER), fg_color="transparent")
        self.amount8label.grid(row=8, column=1, padx=20, pady=10)
        self.slider9var = tk.DoubleVar(value=((114.9-76.5)/2)+76.5)
        self.slider9 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=76.5/1.5, to=114.9*1.5, variable=self.slider9var, command=update_label9)
        self.slider9.grid(row=9, column=2, padx=20, pady=10)
        self.slider9label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Mittelwert des Drehmoments der Einspritzschnecke:", fg_color="transparent")
        self.slider9label.grid(row=9, column=0, padx=20, pady=10)
        self.amount9label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider9var.get()):.4f}", NEWTONMETER), fg_color="transparent")
        self.amount9label.grid(row=9, column=1, padx=20, pady=10)
        self.slider10var = tk.DoubleVar(value=((155.5-144.8)/2)+144.8)
        self.slider10 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=144.8/1.5, to=155.5*1.5, variable=self.slider10var, command=update_label10)
        self.slider10.grid(row=10, column=2, padx=20, pady=10)
        self.slider10label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Höchstwert des Widerstands der Einspritzschnecke:", fg_color="transparent")
        self.slider10label.grid(row=10, column=0, padx=20, pady=10)
        self.amount10label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider10var.get()):.4f}", BAR), fg_color="transparent")
        self.amount10label.grid(row=10, column=1, padx=20, pady=10)
        self.slider11var = tk.DoubleVar(value=((943-780.5)/2)+780.5)
        self.slider11 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=780.5/1.5, to=943*1.5, variable=self.slider11var, command=update_label11)
        self.slider11.grid(row=11, column=2, padx=20, pady=10)
        self.slider11label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Höchstwert des Drucks beim Spritzen:", fg_color="transparent")
        self.slider11label.grid(row=11, column=0, padx=20, pady=10)
        self.amount11label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider11var.get()):.4f}", BAR), fg_color="transparent")
        self.amount11label.grid(row=11, column=1, padx=20, pady=10)
        self.slider12var = tk.DoubleVar(value=((9.06-8.33)/2)+8.33)
        self.slider12 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=8.33/1.5, to=9.06*1.5, variable=self.slider12var, command=update_label12)
        self.slider12.grid(row=12, column=2, padx=20, pady=10)
        self.slider12label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Position der Einspritzschnecke am Ende des Prozesses:", fg_color="transparent")
        self.slider12label.grid(row=12, column=0, padx=20, pady=10)
        self.amount12label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider12var.get()):.4f}", CM), fg_color="transparent")
        self.amount12label.grid(row=12, column=1, padx=20, pady=10)
        self.slider13var = tk.DoubleVar(value=((19.23-18.51)/2)+18.51)
        self.slider13 = customtkinter.CTkSlider(self.tabview.tab("Maschinensimulation"), from_=18.51/1.5, to=19.23*1.5, variable=self.slider13var, command=update_label13)
        self.slider13.grid(row=13, column=2, padx=20, pady=10)
        self.slider13label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"Volumen des Spritzers:", fg_color="transparent")
        self.slider13label.grid(row=13, column=0, padx=20, pady=10)
        self.amount13label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=(f"{(self.slider13var.get()):.4f}", CM3), fg_color="transparent")
        self.amount13label.grid(row=13, column=1, padx=20, pady=10)
        
        # Production Widgets
        self.produce_label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text="-", fg_color="transparent")
        self.produce_label.grid(row=10, column=7, padx=20, pady=10)
        self.produce_button = customtkinter.CTkButton(self.tabview.tab("Maschinensimulation"), text="Produziere ", command=self.produce_func)
        self.produce_button.grid(row=11, column=7, padx=20, pady=10)
        self.algoOptionVar = customtkinter.StringVar(value="Partikelschwarmoptimierung")
        self.algoOption = customtkinter.CTkOptionMenu(self.tabview.tab("Maschinensimulation"), values=["Partikelschwarmoptimierung", "Genetischer Algorithmus", "Simulierte Abkühlung"],variable=self.algoOptionVar)
        self.algoOption.set("Partikelschwarmoptimierung")
        self.algoOption.grid(row=12, column=7, padx=20, pady=10)
        self.ai_button = customtkinter.CTkButton(self.tabview.tab("Maschinensimulation"), text="Generiere optimale Parameter ", command=self.use_algo, state="disabled") # Evtl mit combobox verschiedene Algorithmen anbieten
        self.ai_button.grid(row=13, column=7, padx=20, pady=10)
        ai_tooltip_string = ("{Die Parameter werden mit ", self.algoOptionVar.get(), " generiert. Dieser stochastische Optimierungsalgorithmus produziert nicht-deterministische Ergebnisse.}")
        self.ai_tooltip = CTkToolTip(self.ai_button, message=ai_tooltip_string)
        
        # Bild
        self.placeholder_pic = customtkinter.CTkImage(light_image=Image.open('graphics/placeholder.jpg'), dark_image=Image.open('graphics/placeholder.jpg'), size=(200,200)) # WidthxHeight
        self.ausschuss_pic = customtkinter.CTkImage(light_image=Image.open('graphics/ausschuss.jpg'), dark_image=Image.open('graphics/ausschuss.jpg'), size=(200,200)) # WidthxHeight
        self.inOrdnung_pic = customtkinter.CTkImage(light_image=Image.open('graphics/inOrdnung.jpg'), dark_image=Image.open('graphics/inOrdnung.jpg'), size=(200,200)) # WidthxHeight
        self.sollbereich_pic = customtkinter.CTkImage(light_image=Image.open('graphics/sollbereich.jpg'), dark_image=Image.open('graphics/sollbereich.jpg'), size=(200,200)) # WidthxHeight
        self.ineffizient_pic = customtkinter.CTkImage(light_image=Image.open('graphics/ineffizient.jpg'), dark_image=Image.open('graphics/ineffizient.jpg'), size=(200,200)) # WidthxHeight
        self.image_label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text="", image=self.placeholder_pic)
        self.image_label.place(relx=0.95, rely=0.05, anchor="ne")
        
    def produce_func(self):
        self.vars = []
        self.vars.append(self.slider1var.get()) 
        self.vars.append(self.slider2var.get()) 
        self.vars.append(self.slider3var.get()) 
        self.vars.append(self.slider4var.get()) 
        self.vars.append(self.slider5var.get()) 
        self.vars.append(self.slider6var.get()) 
        self.vars.append(self.slider7var.get()) 
        self.vars.append(self.slider8var.get()) 
        self.vars.append(self.slider9var.get()) 
        self.vars.append(self.slider10var.get()) 
        self.vars.append(self.slider11var.get()) 
        self.vars.append(self.slider12var.get()) 
        self.vars.append(self.slider13var.get()) 
        self.vars = np.array(self.vars).reshape(1, -1)
        self.vars = self.min_max_scaler.transform(self.vars)
        self.prediction = (self.model.predict(self.vars)).item()
        
        quality_cat = self.judge_quality()
        msg = f"Sie haben Qualität {self.prediction:.4f} erzielt! Wertung: {quality_cat}"
        self.produce_label.configure(text=msg)
        produce_tooltip_string = "Die Qualität wird durch ein Neuronales Netz berechnet. Dieses wurde mit einem Datensatz der die Herstellung von Kunststofflinsen abbildet trainiert."
        self.produce_tooltip = CTkToolTip(self.produce_label, message=produce_tooltip_string)
        self.ai_button.configure(state="normal")
    
    def use_algo(self):
        if(self.algoOptionVar=="Genetischer Algorithmus"):
            solution_std, fitness, self.scores, iterations = ga.ga(self.model, self.vars, 2, 30, 100, 0.2, "tournament", "blend")
        elif(self.algoOptionVar=="Simulierte Abkühlung"):
            solution_std, fitness, self.scores, iterations = sa.simulated_annealing(self.model, self.vars, 2, 100, 100, 0.1, False, "exponential")
        else:
            solution_std, fitness, self.scores, it = pso.pso(self.model, self.vars, 5.0, pop_size=30, iterations=100, w=0.6, c1=1, c2=2)
        solution = self.min_max_scaler.inverse_transform(solution_std)
        self.transformed_solution = solution.squeeze()
        self.Optparamterlabel0 = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text="Optimierte Parameter", fg_color="transparent", font=("Arial", 16, "bold"))
        self.Optparamterlabel0.grid(row=0, column=3, padx=20, pady=10)
        self.optParameterLabels = []
        for i, value in enumerate(self.transformed_solution):
            optParameterLabel = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text=f"{value:.4f}", fg_color="transparent")
            optParameterLabel.grid(row=i + 1, column=3, padx=20, pady=10)
            self.optParameterLabels.append(optParameterLabel)
        self.useOptimizedButton = customtkinter.CTkButton(self.tabview.tab("Maschinensimulation"), text="Übernehme generierte Werte", command=self.useOptimizedFunc)
        self.useOptimizedButton.grid(row=14, column=3, padx=20, pady=10)
        
    def useOptimizedFunc(self):
        self.slider1.set((self.transformed_solution)[0])
        self.amount1label.configure(text=(f"{(self.transformed_solution)[0]:.4f}", CELSIUS))
        self.slider2.set((self.transformed_solution)[1])
        self.amount2label.configure(text=(f"{(self.transformed_solution)[1]:.4f}", CELSIUS))
        self.slider3.set((self.transformed_solution)[2])
        self.amount3label.configure(text=(f"{(self.transformed_solution)[2]:.4f}", SECONDS))
        self.slider4.set((self.transformed_solution)[3])
        self.amount4label.configure(text=(f"{(self.transformed_solution)[3]:.4f}", SECONDS))
        self.slider5.set((self.transformed_solution)[4])
        self.amount5label.configure(text=(f"{(self.transformed_solution)[4]:.4f}", SECONDS))
        self.slider6.set((self.transformed_solution)[5])
        self.amount6label.configure(text=(f"{(self.transformed_solution)[5]:.4f}", NEWTON))
        self.slider7.set((self.transformed_solution)[6])
        self.amount7label.configure(text=(f"{(self.transformed_solution)[6]:.4f}", NEWTON))
        self.slider8.set((self.transformed_solution)[7])
        self.amount8label.configure(text=(f"{(self.transformed_solution)[7]:.4f}", NEWTONMETER))
        self.slider9.set((self.transformed_solution)[8])
        self.amount9label.configure(text=(f"{(self.transformed_solution)[8]:.4f}", NEWTONMETER))
        self.slider10.set((self.transformed_solution)[9])
        self.amount10label.configure(text=(f"{(self.transformed_solution)[9]:.4f}", BAR))
        self.slider11.set((self.transformed_solution)[10])
        self.amount11label.configure(text=(f"{(self.transformed_solution)[10]:.4f}", BAR))
        self.slider12.set((self.transformed_solution)[11])
        self.amount12label.configure(text=(f"{(self.transformed_solution)[11]:.4f}", CM))
        self.slider13.set((self.transformed_solution)[12])
        self.amount13label.configure(text=(f"{(self.transformed_solution)[12]:.4f}", CM3))

    def judge_quality(self):
        if(self.prediction < 4):
            self.image_label.configure(image=self.ausschuss_pic)
            return "Ausschuss"
        elif(self.prediction < 4.5):
            self.image_label.configure(image=self.inOrdnung_pic)
            return "in Ordnung"
        elif(self.prediction <= 5):
            self.image_label.configure(image=self.sollbereich_pic)
            return "Sollbereich"
        else:
            self.image_label.configure(image=self.ineffizient_pic)
            return "Unwirtschaftlich"

app = App()
app.mainloop()