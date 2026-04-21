import pickle

import customtkinter
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import algorithm.pso as pso
from helper.help_functions import create_scaler

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1100x750")
        self.title("Spritzguss Demonstrator")
        
        self.min_max_scaler = create_scaler()
        self.model = pickle.load(open("nn/neural-net.sav", 'rb'))
        
        def update_label1(value):
            self.slider1label.configure(text=f"Temperatur des Polymer vor Injektion in den Guss: {value:.4f}")
        def update_label2(value):
            self.slider2label.configure(text=f"Temperatur des Gusses: {value:.4f}")
        def update_label3(value):
            self.slider3label.configure(text=f"Zeit um den Guss zu füllen: {value:.4f}")
        def update_label4(value):
            self.slider4label.configure(text=f"Zeit um das Produkt zu plastizieren: {value:.4f}")
        def update_label5(value):
            self.slider5label.configure(text=f"Zeit um den Prozess für ein Produkt zu beenden: {value:.4f}")
        def update_label6(value):
            self.slider6label.configure(text=f"Schließmoment des Gusses: {value:.4f}")
        def update_label7(value):
            self.slider7label.configure(text=f"Höchstwert des Schließmoment des Gusses: {value:.4f}")
        def update_label8(value):
            self.slider8label.configure(text=f"Höchstwert des Drehmoments der Einspritzschnecke: {value:.4f}")
        def update_label9(value):
            self.slider9label.configure(text=f"Mittelwert des Drehmoments der Einspritzschnecke: {value:.4f}")
        def update_label10(value):
            self.slider10label.configure(text=f"Höchstwert des Widerstands der Einspritzschnecke: {value:.4f}")
        def update_label11(value):
            self.slider11label.configure(text=f"Höchstwert des Drucks beim Spritzen: {value:.4f}")
        def update_label12(value):
            self.slider12label.configure(text=f"Position der Einspritzschnecke am Ende des Prozesses: {value:.4f}")
        def update_label13(value):
            self.slider13label.configure(text=f"Volumen des Spritzers: {value:.4f}")
            
        STEPS=100
        
        # Widgets:
        self.parameterLabel = customtkinter.CTkLabel(self, text="Prozessparameter", fg_color="transparent")
        self.parameterLabel.grid(row=0, column=0, padx=20, pady=10)
        self.sliderLabel = customtkinter.CTkLabel(self, text="Anpassen der Prozessparameter", fg_color="transparent")
        self.sliderLabel.grid(row=0, column=1, padx=20, pady=10)
        self.slider1var = tk.DoubleVar(value=((155.032-81.747)/2)+81.747)
        self.slider1 = customtkinter.CTkSlider(self, from_=81.747/1.5, to=155.032*1.5, variable=self.slider1var, command=update_label1, number_of_steps=STEPS)
        self.slider1.grid(row=1, column=1, padx=20, pady=10)
        self.slider1label = customtkinter.CTkLabel(self, text=f"Temperatur des Polymer vor Injektion in den Guss: {(self.slider1var.get()):.4f}", fg_color="transparent")
        self.slider1label.grid(row=1, column=0, padx=20, pady=10)
        self.slider2var = tk.DoubleVar(value=((82.159-78.409)/2)+78.409)
        self.slider2 = customtkinter.CTkSlider(self, from_=78.409/1.5, to=82.159*1.5, variable=self.slider2var, command=update_label2, number_of_steps=STEPS)
        self.slider2.grid(row=2, column=1, padx=20, pady=10)
        self.slider2label = customtkinter.CTkLabel(self, text=f"Temperatur des Gusses: {(self.slider2var.get()):.4f}", fg_color="transparent")
        self.slider2label.grid(row=2, column=0, padx=20, pady=10)
        self.slider3var = tk.DoubleVar(value=((11.232-6.084)/2)+6.084)
        self.slider3 = customtkinter.CTkSlider(self, from_=6.084/1.5, to=11.232*1.5, variable=self.slider3var, command=update_label3, number_of_steps=STEPS)
        self.slider3.grid(row=3, column=1, padx=20, pady=10)
        self.slider3label = customtkinter.CTkLabel(self, text=f"Zeit um den Guss zu füllen: {(self.slider3var.get()):.4f}", fg_color="transparent")
        self.slider3label.grid(row=3, column=0, padx=20, pady=10)
        self.slider4var = tk.DoubleVar(value=((6.61-2.78)/2)+2.78)
        self.slider4 = customtkinter.CTkSlider(self, from_=2.78/1.5, to=6.61*1.5, variable=self.slider4var, command=update_label4, number_of_steps=STEPS)
        self.slider4.grid(row=4, column=1, padx=20, pady=10)
        self.slider4label = customtkinter.CTkLabel(self, text=f"Zeit um das Produkt zu plastizieren: {(self.slider4var.get()):.4f}", fg_color="transparent")
        self.slider4label.grid(row=4, column=0, padx=20, pady=10)
        self.slider5var = tk.DoubleVar(value=((75.79-74.78)/2)+74.78)
        self.slider5 = customtkinter.CTkSlider(self, from_=74.78/1.5, to=75.79*1.5, variable=self.slider5var, command=update_label5, number_of_steps=STEPS)
        self.slider5.grid(row=5, column=1, padx=20, pady=10)
        self.slider5label = customtkinter.CTkLabel(self, text=f"Zeit um den Prozess für ein Produkt zu beenden: {(self.slider5var.get()):.4f}", fg_color="transparent")
        self.slider5label.grid(row=5, column=0, padx=20, pady=10)
        self.slider6var = tk.DoubleVar(value=((930.6-876.7)/2)+876.7)
        self.slider6 = customtkinter.CTkSlider(self, from_=876.7/1.5, to=930.6*1.5, variable=self.slider6var, command=update_label6, number_of_steps=STEPS)
        self.slider6.grid(row=6, column=1, padx=20, pady=10)
        self.slider6label = customtkinter.CTkLabel(self, text=f"Schließmoment des Gusses: {(self.slider6var.get()):.4f}", fg_color="transparent")
        self.slider6label.grid(row=6, column=0, padx=20, pady=10)
        self.slider7var = tk.DoubleVar(value=((946.5-894.8)/2)+894.8)
        self.slider7 = customtkinter.CTkSlider(self, from_=894.8/1.5, to=946.5*1.5, variable=self.slider7var, command=update_label7, number_of_steps=STEPS)
        self.slider7.grid(row=7, column=1, padx=20, pady=10)
        self.slider7label = customtkinter.CTkLabel(self, text=f"Höchstwert des Schließmoment des Gusses: {(self.slider7var.get()):.4f}", fg_color="transparent")
        self.slider7label.grid(row=7, column=0, padx=20, pady=10)
        self.slider8var = tk.DoubleVar(value=((130.3-94.2)/2)+94.2)
        self.slider8 = customtkinter.CTkSlider(self, from_=94.2/1.5, to=130.3*1.5, variable=self.slider8var, command=update_label8, number_of_steps=STEPS)
        self.slider8.grid(row=8, column=1, padx=20, pady=10)
        self.slider8label = customtkinter.CTkLabel(self, text=f"Höchstwert des Drehmoments der Einspritzschnecke: {(self.slider8var.get()):.4f}", fg_color="transparent")
        self.slider8label.grid(row=8, column=0, padx=20, pady=10)
        self.slider9var = tk.DoubleVar(value=((114.9-76.5)/2)+76.5)
        self.slider9 = customtkinter.CTkSlider(self, from_=76.5/1.5, to=114.9*1.5, variable=self.slider9var, command=update_label9, number_of_steps=STEPS)
        self.slider9.grid(row=9, column=1, padx=20, pady=10)
        self.slider9label = customtkinter.CTkLabel(self, text=f"Mittelwert des Drehmoments der Einspritzschnecke: {(self.slider9var.get()):.4f}", fg_color="transparent")
        self.slider9label.grid(row=9, column=0, padx=20, pady=10)
        self.slider10var = tk.DoubleVar(value=((155.5-144.8)/2)+144.8)
        self.slider10 = customtkinter.CTkSlider(self, from_=144.8/1.5, to=155.5*1.5, variable=self.slider10var, command=update_label10, number_of_steps=STEPS)
        self.slider10.grid(row=10, column=1, padx=20, pady=10)
        self.slider10label = customtkinter.CTkLabel(self, text=f"Höchstwert des Widerstands der Einspritzschnecke: {(self.slider10var.get()):.4f}", fg_color="transparent")
        self.slider10label.grid(row=10, column=0, padx=20, pady=10)
        self.slider11var = tk.DoubleVar(value=((943-780.5)/2)+780.5)
        self.slider11 = customtkinter.CTkSlider(self, from_=780.5/1.5, to=943*1.5, variable=self.slider11var, command=update_label11, number_of_steps=STEPS)
        self.slider11.grid(row=11, column=1, padx=20, pady=10)
        self.slider11label = customtkinter.CTkLabel(self, text=f"Höchstwert des Drucks beim Spritzen: {(self.slider11var.get()):.4f}", fg_color="transparent")
        self.slider11label.grid(row=11, column=0, padx=20, pady=10)
        self.slider12var = tk.DoubleVar(value=((9.06-8.33)/2)+8.33)
        self.slider12 = customtkinter.CTkSlider(self, from_=8.33/1.5, to=9.06*1.5, variable=self.slider12var, command=update_label12, number_of_steps=STEPS)
        self.slider12.grid(row=12, column=1, padx=20, pady=10)
        self.slider12label = customtkinter.CTkLabel(self, text=f"Position der Einspritzschnecke am Ende des Prozesses: {(self.slider12var.get()):.4f}", fg_color="transparent")
        self.slider12label.grid(row=12, column=0, padx=20, pady=10)
        self.slider13var = tk.DoubleVar(value=((19.23-18.51)/2)+18.51)
        self.slider13 = customtkinter.CTkSlider(self, from_=18.51/1.5, to=19.23*1.5, variable=self.slider13var, command=update_label13, number_of_steps=STEPS)
        self.slider13.grid(row=13, column=1, padx=20, pady=10)
        self.slider13label = customtkinter.CTkLabel(self, text=f"Volumen des Spritzers: {(self.slider13var.get()):.4f}", fg_color="transparent")
        self.slider13label.grid(row=13, column=0, padx=20, pady=10)
        
        
        self.produce_button = customtkinter.CTkButton(self, text="Produziere ", command=self.produce_func)
        self.produce_button.grid(row=12, column=6, padx=20, pady=10)
        self.produce_label = customtkinter.CTkLabel(self, text="-", fg_color="transparent")
        self.produce_label.grid(row=11, column=6, padx=20, pady=10)
        self.ai_button = customtkinter.CTkButton(self, text="Generiere optimale Parameter ", command=self.use_algo, state="disabled") # Evtl mit combobox verschiedene Algorithmen anbieten
        self.ai_button.grid(row=13, column=6, padx=20, pady=10)
        
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
        self.prediction = (self.model.predict(self.vars)).item()
        
        quality_cat = self.judge_quality()
        msg = f"Sie haben Qualität {self.prediction:.4f} erzielt! Wertung: {quality_cat}"
        self.produce_label.configure(text=msg)
        self.ai_button.configure(state="normal")
    
    def use_algo(self):
        solution_std, fitness, scores, it = pso.pso(self.model, self.vars, 5.0, pop_size=30, iterations=100, w=0.6, c1=1, c2=2)
        solution = self.min_max_scaler.inverse_transform(solution_std)
        self.vars = solution.squeeze()
        self.Optparamterlabel0 = customtkinter.CTkLabel(self, text="Optimierte Parameter", fg_color="transparent")
        self.Optparamterlabel0.grid(row=0, column=2, padx=20, pady=10)
        self.optParameterLabels = []
        for i, value in enumerate(self.vars):
            optParameterLabel = customtkinter.CTkLabel(self, text=f"{value:.4f}", fg_color="transparent")
            optParameterLabel.grid(row=i + 1, column=2, padx=20, pady=10, sticky="w")
            self.optParameterLabels.append(optParameterLabel)
        self.useOptimizedButton = customtkinter.CTkButton(self, text="Übernehme generierte Werte", command=self.useOptimizedFunc)
        self.useOptimizedButton.grid(row=14, column=2, padx=20, pady=10)
        
    def useOptimizedFunc(self):
        self.slider1.set((self.vars)[0])
        self.slider1label.configure(text=f"Temperatur des Polymer vor Injektion in den Guss: {(self.vars)[0]:.4f}")
        self.slider2.set((self.vars)[1])
        self.slider2label.configure(text=f"Temperatur des Gusses: {(self.vars)[1]:.4f}")
        self.slider3.set((self.vars)[2])
        self.slider3label.configure(text=f"Zeit um den Guss zu füllen: {(self.vars)[2]:.4f}")
        self.slider4.set((self.vars)[3])
        self.slider4label.configure(text=f"Zeit um das Produkt zu plastizieren: {(self.vars)[3]:.4f}")
        self.slider5.set((self.vars)[4])
        self.slider5label.configure(text=f"Zeit um den Prozess für ein Produkt zu beenden: {(self.vars)[4]:.4f}")
        self.slider6.set((self.vars)[5])
        self.slider6label.configure(text=f"Schließmoment des Gusses: {(self.vars)[5]:.4f}")
        self.slider7.set((self.vars)[6])
        self.slider7label.configure(text=f"Höchstwert des Schließmoment des Gusses: {(self.vars)[6]:.4f}")
        self.slider8.set((self.vars)[7])
        self.slider8label.configure(text=f"Höchstwert des Drehmoments der Einspritzschnecke: {(self.vars)[7]:.4f}")
        self.slider9.set((self.vars)[8])
        self.slider9label.configure(text=f"Mittelwert des Drehmoments der Einspritzschnecke: {(self.vars)[8]:.4f}")
        self.slider10.set((self.vars)[9])
        self.slider10label.configure(text=f"Höchstwert des Widerstands der Einspritzschnecke: {(self.vars)[9]:.4f}")
        self.slider11.set((self.vars)[10])
        self.slider11label.configure(text=f"Höchstwert des Drucks beim Spritzen: {(self.vars)[10]:.4f}")
        self.slider12.set((self.vars)[11])
        self.slider12label.configure(text=f"Position der Einspritzschnecke am Ende des Prozesses: {(self.vars)[11]:.4f}")
        self.slider13.set((self.vars)[12])
        self.slider13label.configure(text=f"Volumen des Spritzers: {(self.vars)[12]:.4f}")

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