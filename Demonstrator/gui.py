import pickle
import time

import customtkinter
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from algorithm import ga, sa
import algorithm.pso as pso
from helper.help_functions import create_scaler, get_X, plot_scores
from PIL import Image
from CTkToolTip import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

from helper.k_neighbour_executer import create_kn_classifier
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
        #self.geometry("1300x850")
        #width = self.winfo_screenwidth()
        #height = self.winfo_screenheight()
        #geometry = str(width) + "x" + str(height)
        #self.geometry(geometry)
        self.title("Spritzguss Demonstrator")
        
        self.min_max_scaler = create_scaler()
        self.model = pickle.load(open("nn/neural-net.sav", 'rb'))
        self.scores = []
        self.kn = create_kn_classifier(6, self.min_max_scaler)
        
        def update_label1(value):
            self.amount1label.configure(text=(f"{value:.2f}", CELSIUS))
            slider_change()
        def update_label2(value):
            self.amount2label.configure(text=(f"{value:.2f}", CELSIUS))
            slider_change()
        def update_label3(value):
            self.amount3label.configure(text=(f"{value:.2f}", SECONDS))
            slider_change()
        def update_label4(value):
            self.amount4label.configure(text=(f"{value:.2f}", SECONDS))
            slider_change()
        def update_label5(value):
            self.amount5label.configure(text=(f"{value:.2f}", SECONDS))
            slider_change()
        def update_label6(value):
            self.amount6label.configure(text=(f"{value:.2f}", NEWTON))
            slider_change()
        def update_label7(value):
            self.amount7label.configure(text=(f"{value:.2f}", NEWTON))
            slider_change()
        def update_label8(value):
            self.amount8label.configure(text=(f"{value:.2f}", NEWTONMETER))
            slider_change()
        def update_label9(value):
            self.amount9label.configure(text=(f"{value:.2f}", NEWTONMETER))
            slider_change()
        def update_label10(value):
            self.amount10label.configure(text=(f"{value:.2f}", BAR))
            slider_change()
        def update_label11(value):
            self.amount11label.configure(text=(f"{value:.2f}", BAR))
            slider_change()
        def update_label12(value):
            self.amount12label.configure(text=(f"{value:.2f}", CM))
            slider_change()
        def update_label13(value):
            self.amount13label.configure(text=(f"{value:.2f}", CM3))
            slider_change()
            
        def slider_change():
            self.produce_button.configure(state="disabled")
            self.quality_category_label.configure(text="")
            self.quality_category_label.configure(fg_color="transparent")
            self.produce_label.configure(text="")
            
            
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
        self.layers_text = customtkinter.CTkEntry(self.nnFrame, placeholder_text="64 32 16")
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
        
        self.timeLabel = customtkinter.CTkLabel(self.nnFrame, text="Trainingszeit: ", fg_color="transparent", font=("Arial", 16, "bold") )
        self.timeLabel.grid(row=8, column=0, padx=20, pady=10)
        self.timeValue = customtkinter.CTkLabel(self.nnFrame, text="-", fg_color="transparent", font=("Arial", 16, "bold") )
        self.timeValue.grid(row=8, column=1, padx=20, pady=10)
        
        def nn_button_pressed():
            start_time = time.time()
            layers = tuple(int(x) for x in self.layers_text.get().split())
            self.new_model, mse, perc = create_neural_network(self.min_max_scaler, hidden_layers=layers, acti_func=self.activVar.get(), solve_func=self.solverVar.get(), max_iterations=int(self.iterations_text.get()))
            total_time = time.time() - start_time
            total_time = f"{total_time:.4f} Sek."
            mse = f"{mse:.4f}"
            self.mseValue.configure(text=mse)
            perc = f"{perc:.4f}%"
            self.percValue.configure(text=perc)
            self.takeNNButton.configure(state="normal")
            self.timeValue.configure(text=total_time)
            
        def take_nn_button_pressed():
            self.model = self.new_model
        
        self.createNNButton = customtkinter.CTkButton(self.nnFrame, text="Erstelle Neuronales Netz", command=nn_button_pressed)
        self.createNNButton.grid(row=5, column=1, padx=20)
        self.takeNNButton = customtkinter.CTkButton(self.nnFrame, text="Übernehme erstelltes Neuronales Netz", command=take_nn_button_pressed, state="disabled")
        self.takeNNButton.grid(row=9, column=1, padx=20)
        
        # Maschinensimulation - Tab
        # Widgets:
        self.einstellParam_frame = customtkinter.CTkFrame(self.tabview.tab("Maschinensimulation"),border_width=2,border_color="gray")
        self.einstellParam_frame.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="nsw")
        self.parameterLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Parameter", fg_color="transparent", font=("Arial", 16, "bold") )
        self.parameterLabel.grid(row=0, column=0, padx=20, pady=10)
        parameter_tooltip_string = "Die hier gelisteten Parameter stellen die Prozessparameter des Industrieprozess der Herstellung von Kunststofflinsen dar"
        self.parameter_tooltip = CTkToolTip(self.parameterLabel, message=parameter_tooltip_string)
        self.parameterAmountLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Aktueller Wert", fg_color="transparent", font=("Arial", 16, "bold") )
        self.parameterAmountLabel.grid(row=0, column=2, padx=20, pady=10)
        self.sliderLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Regler", fg_color="transparent", font=("Arial", 16, "bold") )
        self.sliderLabel.grid(row=0, column=1, padx=20, pady=10)
        slider_label_tooltip_string = "Der Wertebereich jedes Prozessparameters basiert auf dem Wertebereich des jeweiligen Parameters im Kunststofflinsendatensatz. Hierbei weicht der minimal einstellbare Wert jedes Parameters um 50% zum minimalwert im Datensatz ab, und ebenso ist es bei dem maximal einstellbaren Wert."
        self.slider_label_tooltip = CTkToolTip(self.sliderLabel, message=slider_label_tooltip_string)
        self.slider1var = tk.DoubleVar(value=((155.032-81.747)/2)+81.747)
        self.slider1 = customtkinter.CTkSlider(self.einstellParam_frame, from_=81.747/1.5, to=155.032*1.5, variable=self.slider1var, command=update_label1)
        self.slider1.grid(row=1, column=1, padx=20, pady=10)
        self.slider1label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schmelztemperatur:", fg_color="transparent")
        self.slider1label.grid(row=1, column=0, padx=20, pady=10)
        self.amount1label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider1var.get()):.2f}", CELSIUS), fg_color="transparent")
        self.amount1label.grid(row=1, column=2, padx=20, pady=10)
        self.slider2var = tk.DoubleVar(value=((82.159-78.409)/2)+78.409)
        self.slider2 = customtkinter.CTkSlider(self.einstellParam_frame, from_=78.409/1.5, to=82.159*1.5, variable=self.slider2var, command=update_label2)
        self.slider2.grid(row=2, column=1, padx=20, pady=10)
        self.slider2label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Werkzeugtemperatur:", fg_color="transparent")
        self.slider2label.grid(row=2, column=0, padx=20, pady=10)
        self.amount2label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider2var.get()):.2f}", CELSIUS), fg_color="transparent")
        self.amount2label.grid(row=2, column=2, padx=20, pady=10)
        self.slider6var = tk.DoubleVar(value=((930.6-876.7)/2)+876.7)
        self.slider6 = customtkinter.CTkSlider(self.einstellParam_frame, from_=876.7/1.5, to=930.6*1.5, variable=self.slider6var, command=update_label6)
        self.slider6.grid(row=3, column=1, padx=20, pady=10)
        self.slider6label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schließkraft:", fg_color="transparent")
        self.slider6label.grid(row=3, column=0, padx=20, pady=10)
        self.amount6label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider6var.get()):.2f}", NEWTON), fg_color="transparent")
        self.amount6label.grid(row=3, column=2, padx=20, pady=10)
        self.slider10var = tk.DoubleVar(value=((155.5-144.8)/2)+144.8)
        self.slider10 = customtkinter.CTkSlider(self.einstellParam_frame, from_=144.8/1.5, to=155.5*1.5, variable=self.slider10var, command=update_label10)
        self.slider10.grid(row=4, column=1, padx=20, pady=10)
        self.slider10label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Gegendruck:", fg_color="transparent")
        self.slider10label.grid(row=4, column=0, padx=20, pady=10)
        self.amount10label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider10var.get()):.2f}", BAR), fg_color="transparent")
        self.amount10label.grid(row=4, column=2, padx=20, pady=10)
        self.slider11var = tk.DoubleVar(value=((943-780.5)/2)+780.5)
        self.slider11 = customtkinter.CTkSlider(self.einstellParam_frame, from_=780.5/1.5, to=943*1.5, variable=self.slider11var, command=update_label11)
        self.slider11.grid(row=5, column=1, padx=20, pady=10)
        self.slider11label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Einspritzdruck:", fg_color="transparent")
        self.slider11label.grid(row=5, column=0, padx=20, pady=10)
        self.amount11label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider11var.get()):.2f}", BAR), fg_color="transparent")
        self.amount11label.grid(row=5, column=2, padx=20, pady=10)
        self.slider13var = tk.DoubleVar(value=((19.23-18.51)/2)+18.51)
        self.slider13 = customtkinter.CTkSlider(self.einstellParam_frame, from_=18.51/1.5, to=19.23*1.5, variable=self.slider13var, command=update_label13)
        self.slider13.grid(row=6, column=1, padx=20, pady=10)
        self.slider13label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schussvolumen:", fg_color="transparent")
        self.slider13label.grid(row=6, column=0, padx=20, pady=10)
        self.amount13label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider13var.get()):.2f}", CM3), fg_color="transparent")
        self.amount13label.grid(row=6, column=2, padx=20, pady=10)
        
        print(type(self.get_kn_vals()))
        print(self.get_kn_vals())
        
        self.generated_paremeters = self.get_kn_vals()
        
        self.prozessParam_frame = customtkinter.CTkFrame(self.tabview.tab("Maschinensimulation"),border_width=2,border_color="gray")
        self.prozessParam_frame.grid(row=1, column=1, padx=20, pady=(10, 0), sticky="nsw")
        self.parameter2Label = customtkinter.CTkLabel(self.prozessParam_frame, text="Parameter", fg_color="transparent", font=("Arial", 16, "bold") )
        self.parameter2Label.grid(row=0, column=0, padx=20, pady=10)
        self.value2Label = customtkinter.CTkLabel(self.prozessParam_frame, text="Wert", fg_color="transparent", font=("Arial", 16, "bold") )
        self.value2Label.grid(row=0, column=1, padx=20, pady=10)
        self.slider3var = tk.DoubleVar(value=self.generated_paremeters[0][6])
        self.slider3label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Füllzeit:", fg_color="transparent")
        self.slider3label.grid(row=1, column=0, padx=20, pady=10)
        self.amount3label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider3var.get()):.2f}", SECONDS), fg_color="transparent")
        self.amount3label.grid(row=1, column=1, padx=20, pady=10)
        self.slider4var = tk.DoubleVar(value=self.generated_paremeters[0][7])
        self.slider4label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Plastizierzeit:", fg_color="transparent")
        self.slider4label.grid(row=2, column=0, padx=20, pady=10)
        self.amount4label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider4var.get()):.2f}", SECONDS), fg_color="transparent")
        self.amount4label.grid(row=2, column=1, padx=20, pady=10)
        self.slider5var = tk.DoubleVar(value=self.generated_paremeters[0][8])
        self.slider5label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Zykluszeit:", fg_color="transparent")
        self.slider5label.grid(row=3, column=0, padx=20, pady=10)
        self.amount5label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider5var.get()):.2f}", SECONDS), fg_color="transparent")
        self.amount5label.grid(row=3, column=1, padx=20, pady=10)
        self.slider7var = tk.DoubleVar(value=self.generated_paremeters[0][9])
        self.slider7label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Maximale Schließkraft:", fg_color="transparent")
        self.slider7label.grid(row=4, column=0, padx=20, pady=10)
        self.amount7label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider7var.get()):.2f}", NEWTON), fg_color="transparent")
        self.amount7label.grid(row=4, column=1, padx=20, pady=10)
        self.slider8var = tk.DoubleVar(value=self.generated_paremeters[0][10])
        self.slider8label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Maximales Schneckendrehmoment:", fg_color="transparent")
        self.slider8label.grid(row=5, column=0, padx=20, pady=10)
        self.amount8label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider8var.get()):.2f}", NEWTONMETER), fg_color="transparent")
        self.amount8label.grid(row=5, column=1, padx=20, pady=10)
        self.slider9var = tk.DoubleVar(value=self.generated_paremeters[0][11])
        self.slider9label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Mittleres Schneckendrehmoment:", fg_color="transparent")
        self.slider9label.grid(row=6, column=0, padx=20, pady=10)
        self.amount9label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider9var.get()):.2f}", NEWTONMETER), fg_color="transparent")
        self.amount9label.grid(row=6, column=1, padx=20, pady=10)
        self.slider12var = tk.DoubleVar(value=self.generated_paremeters[0][12])
        self.slider12label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Schneckenposition am Ende des Nachdrucks:", fg_color="transparent")
        self.slider12label.grid(row=7, column=0, padx=20, pady=10)
        self.amount12label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider12var.get()):.2f}", CM), fg_color="transparent")
        self.amount12label.grid(row=7, column=1, padx=20, pady=10)
        
        #Quality Widgets
        self.qual_frame = customtkinter.CTkFrame(self.tabview.tab("Maschinensimulation"),border_width=2,border_color="gray")
        self.qual_frame.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="nsw")
        self.qual_label = customtkinter.CTkLabel(self.qual_frame, text="Qualitätsergebnis:", fg_color="transparent")
        self.qual_label.grid(row=0, column=0, padx=20, pady=10)
        self.quality_category_label = customtkinter.CTkLabel(self.qual_frame, text="", fg_color="transparent")
        self.quality_category_label.grid(row=1, column=0, padx=20, pady=10)
        self.qual_value_label = customtkinter.CTkLabel(self.qual_frame, text="U0-Wert:", fg_color="transparent")
        self.qual_value_label.grid(row=0, column=1, padx=20, pady=10)
        self.produce_label = customtkinter.CTkLabel(self.qual_frame, text="", fg_color="transparent")
        self.produce_label.grid(row=1, column=1, padx=20, pady=10)
        
        # Production Widgets
        self.production_frame = customtkinter.CTkFrame(self.tabview.tab("Maschinensimulation"),border_width=2,border_color="gray")
        self.production_frame.grid(row=0, column=1, padx=20, pady=(10, 0), sticky="nsw")
        self.producing_button = customtkinter.CTkButton(self.production_frame, text="Produziere ", command=self.set_kn_vals)
        self.producing_button.grid(row=0, column=1, padx=20, pady=10)
        self.produce_button = customtkinter.CTkButton(self.production_frame, text="Qualitätsprüfung ", command=self.produce_func, state="disabled")
        self.produce_button.grid(row=0, column=2, padx=20, pady=10)
        self.algoOptionVar = customtkinter.StringVar(value="Partikelschwarmoptimierung")
        self.algoOption = customtkinter.CTkOptionMenu(self.production_frame, values=["Partikelschwarmoptimierung", "Genetischer Algorithmus", "Simulierte Abkühlung"],variable=self.algoOptionVar)
        self.algoOption.set("Partikelschwarmoptimierung")
        self.algoOption.grid(row=0, column=3, padx=20, pady=10)
        self.ai_button = customtkinter.CTkButton(self.production_frame, text="Generiere optimale Parameter ", command=self.use_algo, state="disabled") # Evtl mit combobox verschiedene Algorithmen anbieten
        self.ai_button.grid(row=0, column=4, padx=20, pady=10)
        ai_tooltip_string = ("{Die Parameter werden mit ", self.algoOptionVar.get(), " generiert. Dieser stochastische Optimierungsalgorithmus produziert nicht-deterministische Ergebnisse.}")
        self.ai_tooltip = CTkToolTip(self.ai_button, message=ai_tooltip_string)
        
        # Bild
        #self.placeholder_pic = customtkinter.CTkImage(light_image=Image.open('graphics/placeholder.jpg'), dark_image=Image.open('graphics/placeholder.jpg'), size=(200,200)) # WidthxHeight
        #self.ausschuss_pic = customtkinter.CTkImage(light_image=Image.open('graphics/ausschuss.jpg'), dark_image=Image.open('graphics/ausschuss.jpg'), size=(200,200)) # WidthxHeight
        #self.inOrdnung_pic = customtkinter.CTkImage(light_image=Image.open('graphics/inOrdnung.jpg'), dark_image=Image.open('graphics/inOrdnung.jpg'), size=(200,200)) # WidthxHeight
        #self.sollbereich_pic = customtkinter.CTkImage(light_image=Image.open('graphics/sollbereich.jpg'), dark_image=Image.open('graphics/sollbereich.jpg'), size=(200,200)) # WidthxHeight
        #self.ineffizient_pic = customtkinter.CTkImage(light_image=Image.open('graphics/ineffizient.jpg'), dark_image=Image.open('graphics/ineffizient.jpg'), size=(200,200)) # WidthxHeight
        #self.image_label = customtkinter.CTkLabel(self.tabview.tab("Maschinensimulation"), text="", image=self.placeholder_pic)
        #self.image_label.place(relx=0.95, rely=0.05, anchor="ne")
        
    def get_kn_vals(self, scale=True):
        self.vars = []
        self.vars.append(self.slider1var.get()) 
        self.vars.append(self.slider2var.get())  
        self.vars.append(self.slider6var.get()) 
        self.vars.append(self.slider10var.get()) 
        self.vars.append(self.slider11var.get()) 
        self.vars.append(self.slider13var.get())
        
        for i in range(0, 7):
           self.vars.append(0) 
        self.vars = np.array(self.vars).reshape(1, -1)
        self.vars = self.min_max_scaler.transform(self.vars)
        index = [6, 7, 8, 9, 10, 11, 12]
        self.vars = np.delete(self.vars, index)
        self.vars = self.vars.reshape(1, -1)
        self.vars = np.hstack([self.vars, self.kn.predict(self.vars)])
        if(scale):
            self.vars = self.min_max_scaler.inverse_transform(self.vars)
        return self.vars
    
    def set_kn_vals(self):
        self.generated_paremeters = self.get_kn_vals()
        self.amount3label.configure(text=(f"{(self.generated_paremeters)[0][6]:.2f}", SECONDS))
        self.amount4label.configure(text=(f"{(self.generated_paremeters)[0][7]:.2f}", SECONDS))
        self.amount5label.configure(text=(f"{(self.generated_paremeters)[0][8]:.2f}", SECONDS))
        self.amount7label.configure(text=(f"{(self.generated_paremeters)[0][9]:.2f}", NEWTON))
        self.amount8label.configure(text=(f"{(self.generated_paremeters)[0][10]:.2f}", NEWTONMETER))
        self.amount9label.configure(text=(f"{(self.generated_paremeters)[0][11]:.2f}", NEWTONMETER))
        self.amount12label.configure(text=(f"{(self.generated_paremeters)[0][12]:.2f}", CM))
        self.produce_button.configure(state="normal")
        
    def produce_func(self):
        self.vars = []
        self.vars.append(self.slider1var.get()) 
        self.vars.append(self.slider2var.get()) 
        self.vars.append(self.slider6var.get()) 
        self.vars.append(self.slider10var.get()) 
        self.vars.append(self.slider11var.get()) 
        self.vars.append(self.slider13var.get()) 
        
        for i in range(0, 7):
           self.vars.append(0) 
        self.vars = np.array(self.vars).reshape(1, -1)
        self.vars = self.min_max_scaler.transform(self.vars)
        index = [6, 7, 8, 9, 10, 11, 12]
        self.vars = np.delete(self.vars, index)
        self.vars = self.vars.reshape(1, -1)
        print(self.vars)
        print(self.kn.predict(self.vars))
        self.vars = np.hstack([self.vars, self.kn.predict(self.vars)])
        print(self.vars)
        
        self.vars = np.array(self.vars).reshape(1, -1)
        #self.vars = self.min_max_scaler.transform(self.vars)
        print(self.vars)
        print((self.model.predict(self.vars)).item())
        self.prediction = (self.model.predict(self.vars)).item()
        
        if(self.prediction < 0):
            self.prediction = 0
        if(self.prediction > 10):
            self.prediction = 10
        
        quality_cat = self.judge_quality()
        self.prediction = self.prediction * 0.1
        msg1 = f"{self.prediction:.2f}"
        self.produce_label.configure(text=msg1)
        msg2 = f"{quality_cat}"
        self.quality_category_label.configure(text=msg2)
        if(quality_cat == "Ausschuss"):
            self.quality_category_label.configure(fg_color="red")
        if(quality_cat == "in Ordnung"):
            self.quality_category_label.configure(fg_color="yellow")
        if(quality_cat == "Sollbereich"):
            self.quality_category_label.configure(fg_color="green")
        if(quality_cat == "Unwirtschaftlich"):
            self.quality_category_label.configure(fg_color="orange")
        produce_tooltip_string = "Die Qualität wird durch ein Neuronales Netz berechnet. Dieses wurde mit einem Datensatz der die Herstellung von Kunststofflinsen abbildet trainiert."
        self.produce_tooltip = CTkToolTip(self.produce_label, message=produce_tooltip_string)
        self.ai_button.configure(state="normal")
        
        
    
    def use_algo(self):
        if(self.algoOptionVar=="Genetischer Algorithmus"):
            solution_std, fitness, self.scores, iterations = ga.ga(self.model, get_X(self.min_max_scaler), self.kn, 5.0, 30, 200, 0.2, "tournament", "blend")
        elif(self.algoOptionVar=="Simulierte Abkühlung"):
            solution_std, fitness, self.scores, iterations = sa.simulated_annealing(self.model, get_X(self.min_max_scaler), self.kn, 5.0, 100, 200, 0.1, False, "exponential")
        else:
            solution_std, fitness, self.scores, it = pso.pso(self.model, get_X(self.min_max_scaler), self.kn, 5.0, pop_size=30, iterations=200, w=0.6, c1=1, c2=2)
            print(solution_std)
            print(fitness)
            print(self.model.predict(solution_std))
            print(solution_std[0][0:6])
            print(solution_std[0][6:13])
            print(self.kn.predict(solution_std[0][0:6].reshape(1, -1)))
        solution = self.min_max_scaler.inverse_transform(solution_std)
        self.transformed_solution = solution.squeeze()
        self.transformed_solution = self.transformed_solution[0:6]
        self.Optparamterlabel0 = customtkinter.CTkLabel(self.einstellParam_frame, text="Empfehlung", fg_color="transparent", font=("Arial", 16, "bold"))
        self.Optparamterlabel0.grid(row=0, column=3, padx=20, pady=10)
        self.optParameterLabels = []
        for i, value in enumerate(self.transformed_solution):
            optParameterLabel = customtkinter.CTkLabel(self.einstellParam_frame, text=f"{value:.2f}", fg_color="transparent")
            optParameterLabel.grid(row=i + 1, column=3, padx=20, pady=10)
            self.optParameterLabels.append(optParameterLabel)
        self.useOptimizedButton = customtkinter.CTkButton(self.einstellParam_frame, text="Übernehme generierte Werte", command=self.useOptimizedFunc)
        self.useOptimizedButton.grid(row=7, column=3, padx=20, pady=10)
        
    def useOptimizedFunc(self):
        self.slider1.set((self.transformed_solution)[0])
        self.amount1label.configure(text=(f"{(self.transformed_solution)[0]:.2f}", CELSIUS))
        self.slider2.set((self.transformed_solution)[1])
        self.amount2label.configure(text=(f"{(self.transformed_solution)[1]:.2f}", CELSIUS))
        self.slider6.set((self.transformed_solution)[2])
        self.amount6label.configure(text=(f"{(self.transformed_solution)[2]:.2f}", NEWTON))
        self.slider10.set((self.transformed_solution)[3])
        self.amount10label.configure(text=(f"{(self.transformed_solution)[3]:.2f}", BAR))
        self.slider11.set((self.transformed_solution)[4])
        self.amount11label.configure(text=(f"{(self.transformed_solution)[4]:.2f}", BAR))
        self.slider13.set((self.transformed_solution)[5])
        self.amount13label.configure(text=(f"{(self.transformed_solution)[5]:.2f}", CM3))
        self.produce_button.configure(state="disabled")

    def judge_quality(self):
        if(self.prediction < 4):
            #self.image_label.configure(image=self.ausschuss_pic)
            return "Ausschuss"
        elif(self.prediction < 4.5):
            #self.image_label.configure(image=self.inOrdnung_pic)
            return "in Ordnung"
        elif(self.prediction <= 5):
            #self.image_label.configure(image=self.sollbereich_pic)
            return "Sollbereich"
        else:
            #self.image_label.configure(image=self.ineffizient_pic)
            return "Unwirtschaftlich"

app = App()
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

# Fenstergröße setzen
app.geometry(f"{screen_width}x{screen_height}+0+0")
app.mainloop()