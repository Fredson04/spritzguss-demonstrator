import customtkinter
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import algorithm.pso as pso
from nn.neural_network import create_neural_network

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("900x750")
        self.title("Spritzguss Demonstrator")
        
        self.model, self.min_max_scaler = create_neural_network()
        
        # Widgets:
        self.parameterLabel = customtkinter.CTkLabel(self, text="Prozessparameter", fg_color="transparent")
        self.parameterLabel.grid(row=0, column=0, padx=20, pady=10)
        self.sliderLabel = customtkinter.CTkLabel(self, text="Anpassen der Prozessparameter", fg_color="transparent")
        self.sliderLabel.grid(row=0, column=1, padx=20, pady=10)
        self.slider1var = tk.DoubleVar(value=((155.032-81.747)/2)+81.747)
        self.slider1 = customtkinter.CTkSlider(self, from_=81.747, to=155.032, variable=self.slider1var)
        self.slider1.grid(row=1, column=1, padx=20, pady=10)
        self.slider1label = customtkinter.CTkLabel(self, text="Temperatur des Polymer vor Injektion in den Guss", fg_color="transparent")
        self.slider1label.grid(row=1, column=0, padx=20, pady=10)
        self.slider2var = tk.DoubleVar(value=((82.159-78.409)/2)+78.409)
        self.slider2 = customtkinter.CTkSlider(self, from_=78.409, to=82.159, variable=self.slider2var)
        self.slider2.grid(row=2, column=1, padx=20, pady=10)
        self.slider2label = customtkinter.CTkLabel(self, text="Temperatur des Gusses", fg_color="transparent")
        self.slider2label.grid(row=2, column=0, padx=20, pady=10)
        self.slider3var = tk.DoubleVar(value=((11.232-6.084)/2)+6.084)
        self.slider3 = customtkinter.CTkSlider(self, from_=6.084, to=11.232, variable=self.slider3var)
        self.slider3.grid(row=3, column=1, padx=20, pady=10)
        self.slider3label = customtkinter.CTkLabel(self, text="Zeit um den Guss zu füllen", fg_color="transparent")
        self.slider3label.grid(row=3, column=0, padx=20, pady=10)
        self.slider4var = tk.DoubleVar(value=((6.61-2.78)/2)+2.78)
        self.slider4 = customtkinter.CTkSlider(self, from_=2.78, to=6.61, variable=self.slider4var)
        self.slider4.grid(row=4, column=1, padx=20, pady=10)
        self.slider4label = customtkinter.CTkLabel(self, text="Zeit um das Produkt zu plastizieren", fg_color="transparent")
        self.slider4label.grid(row=4, column=0, padx=20, pady=10)
        self.slider5var = tk.DoubleVar(value=((75.79-74.78)/2)+74.78)
        self.slider5 = customtkinter.CTkSlider(self, from_=74.78, to=75.79, variable=self.slider5var)
        self.slider5.grid(row=5, column=1, padx=20, pady=10)
        self.slider5label = customtkinter.CTkLabel(self, text="Zeit um den Prozess für ein Produkt zu beenden", fg_color="transparent")
        self.slider5label.grid(row=5, column=0, padx=20, pady=10)
        self.slider6var = tk.DoubleVar(value=((930.6-876.7)/2)+876.7)
        self.slider6 = customtkinter.CTkSlider(self, from_=876.7, to=930.6, variable=self.slider6var)
        self.slider6.grid(row=6, column=1, padx=20, pady=10)
        self.slider6label = customtkinter.CTkLabel(self, text="Schließmoment des Gusses", fg_color="transparent")
        self.slider6label.grid(row=6, column=0, padx=20, pady=10)
        self.slider7var = tk.DoubleVar(value=((946.5-894.8)/2)+894.8)
        self.slider7 = customtkinter.CTkSlider(self, from_=894.8, to=946.5, variable=self.slider7var)
        self.slider7.grid(row=7, column=1, padx=20, pady=10)
        self.slider7label = customtkinter.CTkLabel(self, text="Höchstwert des Schließmoment des Gusses", fg_color="transparent")
        self.slider7label.grid(row=7, column=0, padx=20, pady=10)
        self.slider8var = tk.DoubleVar(value=((130.3-94.2)/2)+94.2)
        self.slider8 = customtkinter.CTkSlider(self, from_=94.2, to=130.3, variable=self.slider8var)
        self.slider8.grid(row=8, column=1, padx=20, pady=10)
        self.slider8label = customtkinter.CTkLabel(self, text="Höchstwert des Drehmoments der Einspritzschnecke", fg_color="transparent")
        self.slider8label.grid(row=8, column=0, padx=20, pady=10)
        self.slider9var = tk.DoubleVar(value=((114.9-76.5)/2)+76.5)
        self.slider9 = customtkinter.CTkSlider(self, from_=76.5, to=114.9, variable=self.slider9var)
        self.slider9.grid(row=9, column=1, padx=20, pady=10)
        self.slider9label = customtkinter.CTkLabel(self, text="Mittelwert des Drehmoments der Einspritzschnecke", fg_color="transparent")
        self.slider9label.grid(row=9, column=0, padx=20, pady=10)
        self.slider10var = tk.DoubleVar(value=((155.5-144.8)/2)+144.8)
        self.slider10 = customtkinter.CTkSlider(self, from_=144.8, to=155.5, variable=self.slider10var)
        self.slider10.grid(row=10, column=1, padx=20, pady=10)
        self.slider10label = customtkinter.CTkLabel(self, text="Höchstwert des Widerstands der Einspritzschnecke", fg_color="transparent")
        self.slider10label.grid(row=10, column=0, padx=20, pady=10)
        self.slider11var = tk.DoubleVar(value=((943-780.5)/2)+780.5)
        self.slider11 = customtkinter.CTkSlider(self, from_=780.5, to=943, variable=self.slider11var)
        self.slider11.grid(row=11, column=1, padx=20, pady=10)
        self.slider11label = customtkinter.CTkLabel(self, text="Höchstwert des Drucks beim Spritzen", fg_color="transparent")
        self.slider11label.grid(row=11, column=0, padx=20, pady=10)
        self.slider12var = tk.DoubleVar(value=((9.06-8.33)/2)+8.33)
        self.slider12 = customtkinter.CTkSlider(self, from_=8.33, to=9.06, variable=self.slider12var)
        self.slider12.grid(row=12, column=1, padx=20, pady=10)
        self.slider12label = customtkinter.CTkLabel(self, text="Position der Einspritzschnecke am Ende des Prozesses", fg_color="transparent")
        self.slider12label.grid(row=12, column=0, padx=20, pady=10)
        self.slider13var = tk.DoubleVar(value=((19.23-18.51)/2)+18.51)
        self.slider13 = customtkinter.CTkSlider(self, from_=18.51, to=19.23, variable=self.slider13var)
        self.slider13.grid(row=13, column=1, padx=20, pady=10)
        self.slider13label = customtkinter.CTkLabel(self, text="Volumen des Spritzers", fg_color="transparent")
        self.slider13label.grid(row=13, column=0, padx=20, pady=10)
        
        self.produce_button = customtkinter.CTkButton(self, text="Produziere ", command=self.produce_func)
        self.produce_button.grid(row=12, column=5, padx=20, pady=10)
        self.produce_label = customtkinter.CTkLabel(self, text="-", fg_color="transparent")
        self.produce_label.grid(row=11, column=5, padx=20, pady=10)
        self.ai_button = customtkinter.CTkButton(self, text="Generiere optimale Parameter ", command=self.use_algo)
        self.ai_button.grid(row=13, column=5, padx=20, pady=10)
        
    def produce_func(self):#Importiere self um alles leichter zu machen
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
        self.prediction = self.model.predict(self.vars)
        
        #Gebe QUalitätsklasse an
        #Evtl mit Farbe
        quality_cat = self.judge_quality()
        
        msg = "Sie haben Qualität", self.prediction, " erzielt! Wertung:", quality_cat
        self.produce_label.configure(text=msg)
    
    def use_algo(self):
        solution_std, fitness, scores, it = pso.pso(self.model, self.vars, 5.0, pop_size=30, iterations=100, w=0.6, c1=1, c2=2)
        #Am leichtesten wäre jetzt slider zu setten

    def judge_quality(self):
        if(self.prediction < 4):
            return "Ausschuss"
        elif(self.prediction < 4.5):
            return "in Ordnung"
        elif(self.prediction <= 5):
            return "Sollbereich"
        else:
            return "Unwirtschaftlich"

app = App()
app.mainloop()