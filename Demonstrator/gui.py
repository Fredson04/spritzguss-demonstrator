import datetime
import pickle
import time

import customtkinter
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from algorithm import ga, sa
import algorithm.pso as pso
from helper.help_functions import create_scaler, create_scaler2, get_X, plot_scores
from PIL import Image
from CTkToolTip import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

from helper.k_neighbour_executer import create_kn_classifier
from helper.nn_executer import create_neural_network

# Konstanten
STEPS=1000 # Nur in dem Wertebereich der Schritte können den Slidern Werte zugeordnet werden, also muss die Schrittmenge hoch (oder nonexistent) sein damit die Qualität der optimierten Parameter erreicht werden kann
DATE=datetime.datetime.now().strftime("%Y%m%d")
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
        self.scaler2 = create_scaler2()
        self.model = pickle.load(open("nn/neural-net-new.sav", 'rb'))
        self.scores = []
        self.kn = create_kn_classifier(6, self.min_max_scaler)
        
        def update_label1(value):
            self.amount1label.configure(text=(f"{value:.1f}", CELSIUS))
            self.slider_change()
        def update_label2(value):
            self.amount2label.configure(text=(f"{value:.1f}", CELSIUS))
            self.slider_change()
        def update_label3(value):
            self.amount3label.configure(text=(f"{value:.1f}", SECONDS))
            self.slider_change()
        def update_label4(value):
            self.amount4label.configure(text=(f"{value:.1f}", SECONDS))
            self.slider_change()
        def update_label5(value):
            self.amount5label.configure(text=(f"{value:.1f}", SECONDS))
            self.slider_change()
        def update_label6(value):
            self.amount6label.configure(text=(f"{value:.1f}", NEWTON))
            self.slider_change()
        def update_label7(value):
            self.amount7label.configure(text=(f"{value:.1f}", NEWTON))
            self.slider_change()
        def update_label8(value):
            self.amount8label.configure(text=(f"{value:.1f}", NEWTONMETER))
            self.slider_change()
        def update_label9(value):
            self.amount9label.configure(text=(f"{value:.1f}", NEWTONMETER))
            self.slider_change()
        def update_label10(value):
            self.amount10label.configure(text=(f"{value:.1f}", BAR))
            self.slider_change()
        def update_label11(value):
            self.amount11label.configure(text=(f"{value:.1f}", BAR))
            self.slider_change()
        def update_label12(value):
            self.amount12label.configure(text=(f"{value:.1f}", CM))
            self.slider_change()
        def update_label13(value):
            self.amount13label.configure(text=(f"{value:.1f}", CM3))
            self.slider_change()
            
        # Tabs    
            
        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.pack(padx=0, pady=0)

        self.tab1 = self.tabview.add("KI Live testen")
        self.tab2 = self.tabview.add("Einstellungen ändern")
        self.tab3 = self.tabview.add("KI selbst trainieren")
        self.tab4 =self.tabview.add("Optimierungsalgorithmus")
        self.tabview.set("KI Live testen")
        
        # Einstellungen ändern Tab
        
        def createProdLaufFrame():
            self.eAeFrame.forget()
            self.prodLaufFrame = customtkinter.CTkFrame(self.tab2, width=200, height=200)
            self.prodLaufFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
            self.returneAe_button = customtkinter.CTkButton(self.prodLaufFrame, text="Zurück zur Auswahl", command=zurueckAuswahl, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.returneAe_button.grid(row=0, column=0, padx=20, sticky="nw")
            self.vermFrame = customtkinter.CTkFrame(self.prodLaufFrame, fg_color="#e8e8e8", corner_radius=8)
            self.vermFrame.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
            
            self.kreis1 = customtkinter.CTkLabel(self.vermFrame, text=str(1), width=36, height=36, corner_radius=18, fg_color="#1f6aa5", text_color="white", font=customtkinter.CTkFont(size=14, weight="bold"))
            self.kreis1.grid(row=0, column=0, padx=(20, 8), pady=15)
            self.vermLabel1 = customtkinter.CTkLabel(self.vermFrame, text="Deine Vermutung", font=customtkinter.CTkFont(size=13), text_color="black")
            self.vermLabel1.grid(row=0, column=1, sticky="w")
            self.linie1 = customtkinter.CTkFrame(self.vermFrame, height=2, fg_color="#333333", width=80)
            self.linie1.grid(row=0, column=2, padx=15, sticky="ew")
            
            self.kreis2 = customtkinter.CTkLabel(self.vermFrame, text=str(2), width=36, height=36, corner_radius=18, fg_color="#d0d0d0", text_color="#333333", font=customtkinter.CTkFont(size=14, weight="bold"))
            self.kreis2.grid(row=0, column=3, padx=(20, 8), pady=15)
            self.vermLabel2 = customtkinter.CTkLabel(self.vermFrame, text="Ergebnis ansehen", font=customtkinter.CTkFont(size=13), text_color="black")
            self.vermLabel2.grid(row=0, column=4, sticky="w")
            self.linie2 = customtkinter.CTkFrame(self.vermFrame, height=2, fg_color="#333333", width=80)
            self.linie2.grid(row=0, column=5, padx=15, sticky="ew")
            
            self.kreis3 = customtkinter.CTkLabel(self.vermFrame, text=str(3), width=36, height=36, corner_radius=18, fg_color="#d0d0d0", text_color="#333333", font=customtkinter.CTkFont(size=14, weight="bold"))
            self.kreis3.grid(row=0, column=6, padx=(20, 8), pady=15)
            self.vermLabel3 = customtkinter.CTkLabel(self.vermFrame, text="Erklärung", font=customtkinter.CTkFont(size=13), text_color="black")
            self.vermLabel3.grid(row=0, column=7, sticky="w")
            
            self.denkFrame = customtkinter.CTkFrame(self.prodLaufFrame, border_width=2,border_color="gray")
            self.denkFrame.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nsw")
            self.denkFrameCap1 = customtkinter.CTkLabel(self.denkFrame, text="Was denkst du – wie viele Produktionsläufe braucht die KI?​", fg_color="transparent", font=("Arial", 18, "bold"))
            self.denkFrameCap1.grid(row=0, column=0, padx=20, pady=10, sticky="w")
            self.denkFrameCap2 = customtkinter.CTkLabel(self.denkFrame, text="Die KI lernt aus echten Produktionsläufen einer Maschine und den dazugehörigen Qualitätsprüfungen. Je mehr Beispiele die KI gesehen hat,\ndesto besser kann sie die Qualität vorhersagen. Aber wie viele Läufe sind genug?​", fg_color="transparent")
            self.denkFrameCap2.grid(row=1, column=0, padx=20, pady=10, sticky="w")
            self.denkFrameCap3 = customtkinter.CTkLabel(self.denkFrame, text="Wähle eine Datenmenge:", fg_color="transparent", font=("Arial", 18, "bold"))
            self.denkFrameCap3.grid(row=2, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.denkFrameCapFrame = customtkinter.CTkFrame(self.denkFrame, border_width=2,border_color="gray")
            self.denkFrameCapFrame.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="nsw")
            
            def do_denkFrameButton1():
                self.denkFrameButton1.configure(fg_color="#c5d8ed")
                self.denkFrameButton2.configure(fg_color="#dce8f5")
                self.denkFrameButton3.configure(fg_color="#dce8f5")
                self.denkFrameButton4.configure(fg_color="#dce8f5")
                self.denkFrameCap_button_var = customtkinter.StringVar(value="250 Läufe")
            
            def do_denkFrameButton2():
                self.denkFrameButton2.configure(fg_color="#c5d8ed")
                self.denkFrameButton1.configure(fg_color="#dce8f5")
                self.denkFrameButton3.configure(fg_color="#dce8f5")
                self.denkFrameButton4.configure(fg_color="#dce8f5")
                self.denkFrameCap_button_var = customtkinter.StringVar(value="500 Läufe")
                
            def do_denkFrameButton3():
                self.denkFrameButton3.configure(fg_color="#c5d8ed")
                self.denkFrameButton1.configure(fg_color="#dce8f5")
                self.denkFrameButton2.configure(fg_color="#dce8f5")
                self.denkFrameButton4.configure(fg_color="#dce8f5")
                self.denkFrameCap_button_var = customtkinter.StringVar(value="1000 Läufe")
                
            def do_denkFrameButton4():
                self.denkFrameButton4.configure(fg_color="#c5d8ed")
                self.denkFrameButton1.configure(fg_color="#dce8f5")
                self.denkFrameButton2.configure(fg_color="#dce8f5")
                self.denkFrameButton3.configure(fg_color="#dce8f5")
                self.denkFrameCap_button_var = customtkinter.StringVar(value="1451 Läufe")
            
            self.denkFrameButton1 = customtkinter.CTkButton(self.denkFrameCapFrame, text="250\nLäufe", command=do_denkFrameButton1,corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.denkFrameButton1.grid(row=0, column=0, padx=20, pady=10)
            self.denkFrameButton2 = customtkinter.CTkButton(self.denkFrameCapFrame, text="500\nLäufe", command=do_denkFrameButton2, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.denkFrameButton2.grid(row=0, column=1, padx=20, pady=10)
            self.denkFrameButton3 = customtkinter.CTkButton(self.denkFrameCapFrame, text="1000\nLäufe", command=do_denkFrameButton3, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.denkFrameButton3.grid(row=0, column=2, padx=20, pady=10)
            self.denkFrameButton4 = customtkinter.CTkButton(self.denkFrameCapFrame, text="1451\nLäufe", command=do_denkFrameButton4, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.denkFrameButton4.grid(row=0, column=3, padx=20, pady=10)
            
            self.denkFrameButton1.configure(fg_color="#c5d8ed")
            self.denkFrameButton2.configure(fg_color="#dce8f5")
            self.denkFrameButton3.configure(fg_color="#dce8f5")
            self.denkFrameButton4.configure(fg_color="#dce8f5")
            self.denkFrameCap_button_var = customtkinter.StringVar(value="250 Läufe")
            
            self.denkFrameCap4 = customtkinter.CTkLabel(self.denkFrame, text="Wie gut wird die KI mit dieser Datenmenge sein?", fg_color="transparent", font=("Arial", 18, "bold"))
            self.denkFrameCap4.grid(row=4, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.denkFrameCapFrame2 = customtkinter.CTkFrame(self.denkFrame, border_width=2,border_color="gray")
            self.denkFrameCapFrame2.grid(row=5, column=0, padx=10, pady=(10, 0), sticky="nsw")
            def do_denkFrameButton5():
                self.denkFrameButton5.configure(fg_color="#c5d8ed")
                self.denkFrameButton6.configure(fg_color="#dce8f5")
                self.denkFrameButton7.configure(fg_color="#dce8f5")
                self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Eher schlecht")
            
            def do_denkFrameButton6():
                self.denkFrameButton6.configure(fg_color="#c5d8ed")
                self.denkFrameButton5.configure(fg_color="#dce8f5")
                self.denkFrameButton7.configure(fg_color="#dce8f5")
                self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Mittel")
                
            def do_denkFrameButton7():
                self.denkFrameButton7.configure(fg_color="#c5d8ed")
                self.denkFrameButton5.configure(fg_color="#dce8f5")
                self.denkFrameButton6.configure(fg_color="#dce8f5")
                self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Gut")
            
            self.denkFrameButton5 = customtkinter.CTkButton(self.denkFrameCapFrame2, text="Eher schlecht​\n(unter 60% richtig\n vorhergesagt)", command=do_denkFrameButton5,corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.denkFrameButton5.grid(row=0, column=0, padx=20, pady=10)
            self.denkFrameButton6 = customtkinter.CTkButton(self.denkFrameCapFrame2, text="Mittel\n(60-80%) richtig\n vorhergesagt)", command=do_denkFrameButton6, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.denkFrameButton6.grid(row=0, column=1, padx=20, pady=10)
            self.denkFrameButton7 = customtkinter.CTkButton(self.denkFrameCapFrame2, text="Gut\n(über 80% richtig\n vorhergesagt​", command=do_denkFrameButton7, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
            self.denkFrameButton7.grid(row=0, column=2, padx=20, pady=10)
            
            self.denkFrameButton5.configure(fg_color="#c5d8ed")
            self.denkFrameButton6.configure(fg_color="#dce8f5")
            self.denkFrameButton7.configure(fg_color="#dce8f5")
            self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Eher schlecht")
            self.denkFrameExit = customtkinter.CTkButton(self.denkFrame, text="Ergebnis anzeigen", command=create_ergebnis_ansehen_frame, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a", font=("Arial", 18, "bold"))
            self.denkFrameExit.grid(row=5, column=3, padx=20, pady=10)
        
        def create_ergebnis_ansehen_frame():
            self.denkFrame.destroy()
            self.kreis1.configure(fg_color="#d0d0d0", text_color="#333333")
            self.kreis2.configure(fg_color="#1f6aa5", text_color="white")
            self.ergebnisAnsehenFrame = customtkinter.CTkFrame(self.prodLaufFrame, border_width=2,border_color="gray")
            self.ergebnisAnsehenFrame.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nsw")
            self.ergebnisAnsehenLabel1 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=(f"Ergebnis: KI trainiert mit {(self.denkFrameCap_button_var.get())} Läufen"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.ergebnisAnsehenLabel1.grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.ergebnisAnsehenLabel2 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("Getestet an 300 Produktionsläufen, die das KI-Modell beim Training nicht gesehen hat"), fg_color="transparent")
            self.ergebnisAnsehenLabel2.grid(row=1, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            seg1 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color="#C00000", corner_radius=4, width=100, height=50)
            seg1.grid(row=2, column=0, padx=2, pady=(0, 10), sticky="nsew")
            seg1.grid_propagate(False)
            self.segLabel1 = customtkinter.CTkLabel(seg1, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel1.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg1.grid_rowconfigure(0, weight=1)
            seg1.grid_columnconfigure(0, weight=1)
            
            seg2 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color="#FFC000", corner_radius=4, width=100, height=50)
            seg2.grid(row=2, column=1, padx=2, pady=(0, 10), sticky="nsew")
            seg2.grid_propagate(False)
            self.segLabel2 = customtkinter.CTkLabel(seg2, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel2.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg2.grid_rowconfigure(0, weight=1)
            seg2.grid_columnconfigure(0, weight=1)
            
            seg3 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color="#8ED973", corner_radius=4, width=100, height=50)
            seg3.grid(row=2, column=2, padx=2, pady=(0, 10), sticky="nsew")
            seg3.grid_propagate(False)
            self.segLabel3 = customtkinter.CTkLabel(seg3, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel3.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg3.grid_rowconfigure(0, weight=1)
            seg3.grid_columnconfigure(0, weight=1)
            
            seg4 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color="#E97132", corner_radius=4, width=100, height=50)
            seg4.grid(row=2, column=3, padx=2, pady=(0, 10), sticky="nsew")
            seg4.grid_propagate(False)
            self.segLabel4 = customtkinter.CTkLabel(seg4, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel4.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg4.grid_rowconfigure(0, weight=1)
            seg4.grid_columnconfigure(0, weight=1)
            
            self.ergebnisAnsehenLabel3 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("Gesamttrefferquote:"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.ergebnisAnsehenLabel3.grid(row=3, column=0, padx=20, pady=10, sticky="w")
            self.progressbar = customtkinter.CTkProgressBar(self.ergebnisAnsehenFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",border_width=0)
            self.progressbar.grid(row=3, column=1, columnspan=3, padx=20, pady=10, sticky="w")
            if(self.denkFrameCap_button_var.get() == "250 Läufe"):
                self.progressbar.configure(progress_color="#C00000")
                self.progressbar.set(0.41)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("41%"), fg_color="transparent", font=("Arial", 18, "bold"))
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("Placeholder text"), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n25,9%")
                self.segLabel2.configure(text="Akzeptabel:\n40,9%")
                self.segLabel3.configure(text="Sollbereich:\n60,8%")
                self.segLabel4.configure(text="Ineffizient:\n36,1%")
            if(self.denkFrameCap_button_var.get() == "500 Läufe"):
                self.progressbar.configure(progress_color="#FFC000")
                self.progressbar.set(0.79)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("79%"), fg_color="transparent", font=("Arial", 18, "bold"))
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("500 Läufe reichen für erste brauchbare Vorhersagen. Aber es gibt trotzdem noch Verwechslungen und falsche Vorhersagen.​"), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n61,9%")
                self.segLabel2.configure(text="Akzeptabel:\n82,2%")
                self.segLabel3.configure(text="Sollbereich:\n80,4%")
                self.segLabel4.configure(text="Ineffizient:\n83,2%")
            if(self.denkFrameCap_button_var.get() == "1000 Läufe"):
                self.progressbar.configure(progress_color="#FFC000")
                self.progressbar.set(0.83)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("83%"), fg_color="transparent", font=("Arial", 18, "bold"))
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("Placeholder text"), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n68,9%")
                self.segLabel2.configure(text="Akzeptabel:\n84,6%")
                self.segLabel3.configure(text="Sollbereich:\n85,6%")
                self.segLabel4.configure(text="Ineffizient:\n88,5%")
            if(self.denkFrameCap_button_var.get() == "1451 Läufe"):
                self.progressbar.configure(progress_color="#8ED973")
                self.progressbar.set(0.84)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("84%"), fg_color="transparent", font=("Arial", 18, "bold"))
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("Placeholder text"), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n73,0%")
                self.segLabel2.configure(text="Akzeptabel:\n84,5%")
                self.segLabel3.configure(text="Sollbereich:\n87,2%")
                self.segLabel4.configure(text="Ineffizient:\n90,0%")
            self.ergebnisAnsehenLabel4.grid(row=3, column=4, padx=20, pady=10, sticky="w")
            self.ergebnisAnsehenLabel5.grid(row=4, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            self.ergebnisAnsehenFrameExit = customtkinter.CTkButton(self.ergebnisAnsehenFrame, text="Was bedeutet das?", command=create_erklaerung_frame, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a", font=("Arial", 18, "bold"))
            self.ergebnisAnsehenFrameExit.grid(row=6, column=3, padx=20, pady=10)
        
        def create_erklaerung_frame():
            self.ergebnisAnsehenFrame.destroy()
            #self.ergebnisAnsehenFrame.grid_forget() 
            self.kreis2.configure(fg_color="#d0d0d0", text_color="#333333")
            self.kreis3.configure(fg_color="#1f6aa5", text_color="white")
            self.erklaerungFrame = customtkinter.CTkFrame(self.prodLaufFrame, border_width=2,border_color="gray")
            self.erklaerungFrame.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nsw")
            self.erklaerungLabel1 = customtkinter.CTkLabel(self.erklaerungFrame, text=("Ergebnis: So verändert die Datenmenge die Qualität der KI"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungLabel1.grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.erklaerungLabel2 = customtkinter.CTkLabel(self.erklaerungFrame, text=("Je mehr Produktionsläufe und Qualitätstests die KI gesehen hat, desto zuverlässiger die Vorhersage.​"), fg_color="transparent")
            self.erklaerungLabel2.grid(row=1, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel1 = customtkinter.CTkLabel(self.erklaerungFrame, text=("250 Läufe​"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel1.grid(row=2, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar1 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color="#C00000",border_width=0)
            self.erklaerungBar1.grid(row=2, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar1.set(0.41)
            self.erklaerungBarLabel11 = customtkinter.CTkLabel(self.erklaerungFrame, text=("41%"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel11.grid(row=2, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel2 = customtkinter.CTkLabel(self.erklaerungFrame, text=("500 Läufe​"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel2.grid(row=3, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar2 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color="#FFC000",border_width=0)
            self.erklaerungBar2.grid(row=3, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar2.set(0.79)
            self.erklaerungBarLabel22 = customtkinter.CTkLabel(self.erklaerungFrame, text=("79%"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel22.grid(row=3, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel3 = customtkinter.CTkLabel(self.erklaerungFrame, text=("1000 Läufe​"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel3.grid(row=4, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar3 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color="#FFC000",border_width=0)
            self.erklaerungBar3.grid(row=4, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar3.set(0.83)
            self.erklaerungBarLabel33 = customtkinter.CTkLabel(self.erklaerungFrame, text=("83%"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel33.grid(row=4, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel4 = customtkinter.CTkLabel(self.erklaerungFrame, text=("1451 Läufe​"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel4.grid(row=5, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar4 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color="#8ED973",border_width=0)
            self.erklaerungBar4.grid(row=5, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar4.set(0.84)
            self.erklaerungBarLabel44 = customtkinter.CTkLabel(self.erklaerungFrame, text=("84%"), fg_color="transparent", font=("Arial", 18, "bold"))
            self.erklaerungBarLabel44.grid(row=5, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungLabel3 = customtkinter.CTkLabel(self.erklaerungFrame, text=("Damit eine KI gute Vorhersagen machen kann, ist sowohl die Menge der Daten als auch die \nsogenannte Qualität der Daten ausschlaggebend. Gute Daten sind vollständig. Das heißt alle \nWerte (z.B. Einstellparameter und Qualitätsprüfung) sind vorhanden.​"), fg_color="transparent")
            self.erklaerungLabel3.grid(row=6, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            self.erklaerungReturnButton = customtkinter.CTkButton(self.erklaerungFrame, text="Andere Datenmenge testen", command=andereDatenmengen, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a", font=("Arial", 18, "bold"))
            self.erklaerungReturnButton.grid(row=7, column=0, padx=20, pady=10)
            self.erklaerungReturnButton2 = customtkinter.CTkButton(self.erklaerungFrame, text="Zum Algorithmenvergleich", corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a", font=("Arial", 18, "bold"))
            self.erklaerungReturnButton2.grid(row=7, column=2, padx=20, pady=10)
            
        def andereDatenmengen():
            self.erklaerungFrame.destroy()
            self.prodLaufFrame.destroy()
            createProdLaufFrame()
        
        def zurueckAuswahl():
            self.prodLaufFrame.destroy()
            self.eAeFrame.grid()
            
        
        self.eAeFrame = customtkinter.CTkFrame(self.tab2, width=200, height=200)
        self.eAeFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.eAe_headline_Label = customtkinter.CTkLabel(self.eAeFrame, text="Was möchtest du untersuchen?​", fg_color="transparent", font=("Arial", 20, "bold") )
        self.eAe_headline_Label.grid(row=0, column=1, padx=20, pady=10)
        self.eAe_text_Label = customtkinter.CTkLabel(self.eAeFrame, text="Die KI besteht aus zwei Teilen: Einem KI-Modell, das die Qualität vorhersagt\nund einem Modell, das die besten Einstellparameter findet.\nHier kannst du beide gezielt verändern und beobachten, was passiert.​​", fg_color="transparent")
        self.eAe_text_Label.grid(row=1, column=1, padx=20, pady=10)
        self.eAeFrame1 = customtkinter.CTkFrame(self.eAeFrame, width=200, height=200)
        self.eAeFrame1.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.eAeFrame1_headline_Label = customtkinter.CTkLabel(self.eAeFrame1, text="Einfluss der Datenmenge verstehen​​", fg_color="transparent", font=("Arial", 20, "bold") )
        self.eAeFrame1_headline_Label.grid(row=0, column=0, padx=20, pady=10)
        self.eAeFrame1_text_Label = customtkinter.CTkLabel(self.eAeFrame1, text="Was passiert, wenn das KI-Modell weniger Produktionsläufe zum Lernen hat?\nWie viele Daten braucht die KI, um zuverlässig zu sein?​​​", fg_color="transparent")
        self.eAeFrame1_text_Label.grid(row=1, column=0, padx=20, pady=10)
        self.eAeFrame1_button = customtkinter.CTkButton(self.eAeFrame1, text="Zum Vergleich", command=createProdLaufFrame)
        self.eAeFrame1_button.grid(row=2, column=0, padx=20)
        self.eAeFrame2 = customtkinter.CTkFrame(self.eAeFrame, width=200, height=200)
        self.eAeFrame2.grid(row=2, column=2, padx=20, pady=20, sticky="nsew")
        self.eAeFrame2_headline_Label = customtkinter.CTkLabel(self.eAeFrame2, text="Optimierungsalgorithmus wechseln​", fg_color="transparent", font=("Arial", 20, "bold") )
        self.eAeFrame2_headline_Label.grid(row=0, column=0, padx=20, pady=10)
        self.eAeFrame2_text_Label = customtkinter.CTkLabel(self.eAeFrame2, text="Welches Modell findet die besten Einstellparameter?\nVergleiche verschiedene Methoden direkt miteinander.​", fg_color="transparent")
        self.eAeFrame2_text_Label.grid(row=1, column=0, padx=20, pady=10)
        self.eAeFrame2_button = customtkinter.CTkButton(self.eAeFrame2, text="Zum Vergleich")
        self.eAeFrame2_button.grid(row=2, column=0, padx=20)
        
        # Optimierungsalgorithmus - Tab
        
        #self.optAlgoFrame = customtkinter.CTkFrame(self.tabview.tab("Optimierungsalgorithmus"), width=200, height=200) 
        
        #fig = plot_scores(self.scores)
        #canvas = FigureCanvasTkAgg(fig, self.tabview.tab("Optimierungsalgorithmus"))  # A tk.DrawingArea.
        #canvas.draw()
        #canvas.get_tk_widget().grid(row=0, column=0)  
        
        # Neuronales Netz
        
        self.nnFrame = customtkinter.CTkFrame(self.tab3, width=200, height=200)
        self.nnFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.layerLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die Größe und Anzahl der Versteckten Schichten an", fg_color="transparent", font=("Arial", 18, "bold") )
        self.layerLabel.grid(row=1, column=0, padx=20, pady=10)
        self.layers_text = customtkinter.CTkEntry(self.nnFrame, placeholder_text="64 32 16")
        self.layers_text.grid(row=1, column=1, padx=20)
        #self.activLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die zu nutzende Aktivierungsfunktion an", fg_color="transparent", font=("Arial", 18, "bold") )
        #self.activLabel.grid(row=2, column=0, padx=20, pady=10)
        self.activVar = customtkinter.StringVar(value="relu")
        #self.activOption = customtkinter.CTkOptionMenu(self.nnFrame,values=["relu", "identity", "logistic", "tanh"],                                 variable=self.activVar)
        #self.activOption.grid(row=2, column=1, padx=20, pady=10)
        #self.solverLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die zu nutzende Solver Funktion an", fg_color="transparent", font=("Arial", 18, "bold") )
        #self.solverLabel.grid(row=3, column=0, padx=20, pady=10)
        self.solverVar = customtkinter.StringVar(value="adam")
        #self.solverOption = customtkinter.CTkOptionMenu(self.nnFrame,values=["lbfgs", "sgd", "adam"],                                 variable=self.solverVar)
        #self.solverOption.grid(row=3, column=1, padx=20, pady=10)
        self.iterationsLabel = customtkinter.CTkLabel(self.nnFrame, text="Gebe die maximale Anzahl an Iterationen an", fg_color="transparent", font=("Arial", 18, "bold") )
        self.iterationsLabel.grid(row=2, column=0, padx=20, pady=10)
        self.iterations_text = customtkinter.CTkEntry(self.nnFrame, placeholder_text="500")
        self.iterations_text.grid(row=2, column=1, padx=20)
        
        self.mseLabel = customtkinter.CTkLabel(self.nnFrame, text="Mean Squared Error des Neuronalen Netzes: ", fg_color="transparent", font=("Arial", 18, "bold") )
        self.mseLabel.grid(row=4, column=0, padx=20, pady=10)
        self.mseValue = customtkinter.CTkLabel(self.nnFrame, text="-", fg_color="transparent", font=("Arial", 18, "bold") )
        self.mseValue.grid(row=4, column=1, padx=20, pady=10)
        
        self.percLabel = customtkinter.CTkLabel(self.nnFrame, text="Akkuratheit des Neuronalen Netzes: ", fg_color="transparent", font=("Arial", 18, "bold") )
        self.percLabel.grid(row=5, column=0, padx=20, pady=10)
        self.percValue = customtkinter.CTkLabel(self.nnFrame, text="-", fg_color="transparent", font=("Arial", 18, "bold") )
        self.percValue.grid(row=5, column=1, padx=20, pady=10)
        
        self.timeLabel = customtkinter.CTkLabel(self.nnFrame, text="Trainingszeit: ", fg_color="transparent", font=("Arial", 18, "bold") )
        self.timeLabel.grid(row=6, column=0, padx=20, pady=10)
        self.timeValue = customtkinter.CTkLabel(self.nnFrame, text="-", fg_color="transparent", font=("Arial", 18, "bold") )
        self.timeValue.grid(row=6, column=1, padx=20, pady=10)
        
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
        self.createNNButton.grid(row=3, column=1, padx=20)
        self.takeNNButton = customtkinter.CTkButton(self.nnFrame, text="Übernehme erstelltes Neuronales Netz", command=take_nn_button_pressed, state="disabled")
        self.takeNNButton.grid(row=7, column=1, padx=20)
        
        # Maschinensimulation - Tab
        # Widgets:
        self.einstellParam_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray")
        self.einstellParam_frame.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsw")
        self.einstellParam_label = customtkinter.CTkLabel(self.einstellParam_frame, text="Einstellparameter", font=("Arial", 20, "bold"))
        self.einstellParam_label.grid(row=0, column=0, columnspan=3, sticky="we", padx=10, pady=10)
        self.parameterLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Parameter", fg_color="transparent", font=("Arial", 18, "bold") )
        self.parameterLabel.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        parameter_tooltip_string = "Die hier gelisteten Parameter stellen die Prozessparameter des Industrieprozess der Herstellung von Kunststofflinsen dar"
        self.parameter_tooltip = CTkToolTip(self.parameterLabel, message=parameter_tooltip_string)
        self.parameterAmountLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Aktueller Wert", fg_color="transparent", font=("Arial", 18, "bold") )
        self.parameterAmountLabel.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.sliderLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Regler", fg_color="transparent", font=("Arial", 18, "bold") )
        self.sliderLabel.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        slider_label_tooltip_string = "Der Wertebereich jedes Prozessparameters basiert auf dem Wertebereich des jeweiligen Parameters im Kunststofflinsendatensatz. Hierbei weicht der minimal einstellbare Wert jedes Parameters um 50% zum minimalwert im Datensatz ab, und ebenso ist es bei dem maximal einstellbaren Wert."
        self.slider_label_tooltip = CTkToolTip(self.sliderLabel, message=slider_label_tooltip_string)
        self.slider1var = tk.DoubleVar(value=((155.032-81.747)/2)+81.747)
        self.slider1 = customtkinter.CTkSlider(self.einstellParam_frame, from_=81.747/1.5, to=155.032*1.5, variable=self.slider1var, command=update_label1)
        self.slider1.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.slider1label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schmelztemperatur:", fg_color="transparent", font=("Arial", 14))
        self.slider1label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.amount1label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider1var.get()):.1f}", CELSIUS), fg_color="transparent", font=("Arial", 14))
        self.amount1label.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.slider2var = tk.DoubleVar(value=((82.159-78.409)/2)+78.409)
        self.slider2 = customtkinter.CTkSlider(self.einstellParam_frame, from_=78.409/1.5, to=82.159*1.5, variable=self.slider2var, command=update_label2)
        self.slider2.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.slider2label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Werkzeugtemperatur:", fg_color="transparent", font=("Arial", 14))
        self.slider2label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.amount2label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider2var.get()):.1f}", CELSIUS), fg_color="transparent", font=("Arial", 14))
        self.amount2label.grid(row=3, column=2, padx=10, pady=5, sticky="w")
        self.slider6var = tk.DoubleVar(value=((930.6-876.7)/2)+876.7)
        self.slider6 = customtkinter.CTkSlider(self.einstellParam_frame, from_=876.7/1.5, to=930.6*1.5, variable=self.slider6var, command=update_label6)
        self.slider6.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.slider6label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schließkraft:", fg_color="transparent", font=("Arial", 14))
        self.slider6label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.amount6label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider6var.get()):.1f}", NEWTON), fg_color="transparent", font=("Arial", 14))
        self.amount6label.grid(row=4, column=2, padx=10, pady=5, sticky="w")
        self.slider10var = tk.DoubleVar(value=((155.5-144.8)/2)+144.8)
        self.slider10 = customtkinter.CTkSlider(self.einstellParam_frame, from_=144.8/1.5, to=155.5*1.5, variable=self.slider10var, command=update_label10)
        self.slider10.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        self.slider10label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Gegendruck:", fg_color="transparent", font=("Arial", 14))
        self.slider10label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.amount10label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider10var.get()):.1f}", BAR), fg_color="transparent", font=("Arial", 14))
        self.amount10label.grid(row=5, column=2, padx=10, pady=5, sticky="w")
        self.slider11var = tk.DoubleVar(value=((943-780.5)/2)+780.5)
        self.slider11 = customtkinter.CTkSlider(self.einstellParam_frame, from_=780.5/1.5, to=943*1.5, variable=self.slider11var, command=update_label11)
        self.slider11.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        self.slider11label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Einspritzdruck:", fg_color="transparent", font=("Arial", 14))
        self.slider11label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.amount11label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider11var.get()):.1f}", BAR), fg_color="transparent", font=("Arial", 14))
        self.amount11label.grid(row=6, column=2, padx=10, pady=5, sticky="w")
        self.slider13var = tk.DoubleVar(value=((19.23-18.51)/2)+18.51)
        self.slider13 = customtkinter.CTkSlider(self.einstellParam_frame, from_=18.51/1.5, to=19.23*1.5, variable=self.slider13var, command=update_label13)
        self.slider13.grid(row=7, column=1, padx=10, pady=5, sticky="w")
        self.slider13label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schussvolumen:", fg_color="transparent", font=("Arial", 14))
        self.slider13label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.amount13label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider13var.get()):.1f}", CM3), fg_color="transparent", font=("Arial", 14))
        self.amount13label.grid(row=7, column=2, padx=10, pady=5, sticky="w")
        
        self.generated_paremeters = self.get_kn_vals()
        
        self.prozessParam_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray")
        self.prozessParam_frame.grid(row=1, column=1, padx=10, pady=(10, 0), sticky="nsw")
        self.prozessParam_label = customtkinter.CTkLabel(self.prozessParam_frame, text="Prozessparameter", font=("Arial", 20, "bold"))
        self.prozessParam_label.grid(row=0, column=0, columnspan=2, sticky="we", padx=10, pady=10)
        self.parameter2Label = customtkinter.CTkLabel(self.prozessParam_frame, text="Parameter", fg_color="transparent", font=("Arial", 18, "bold") )
        self.parameter2Label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.value2Label = customtkinter.CTkLabel(self.prozessParam_frame, text="Wert", fg_color="transparent", font=("Arial", 18, "bold") )
        self.value2Label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.slider3var = tk.DoubleVar(value=self.generated_paremeters[0][6])
        self.slider3label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Füllzeit:", fg_color="transparent", font=("Arial", 14))
        self.slider3label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.amount3label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider3var.get()):.1f}", SECONDS), fg_color="transparent", font=("Arial", 14))
        self.amount3label.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.slider4var = tk.DoubleVar(value=self.generated_paremeters[0][7])
        self.slider4label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Plastizierzeit:", fg_color="transparent", font=("Arial", 14))
        self.slider4label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.amount4label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider4var.get()):.1f}", SECONDS), fg_color="transparent", font=("Arial", 14))
        self.amount4label.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.slider5var = tk.DoubleVar(value=self.generated_paremeters[0][8])
        self.slider5label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Zykluszeit:", fg_color="transparent", font=("Arial", 14))
        self.slider5label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.amount5label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider5var.get()):.1f}", SECONDS), fg_color="transparent", font=("Arial", 14))
        self.amount5label.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.slider7var = tk.DoubleVar(value=self.generated_paremeters[0][9])
        self.slider7label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Maximale Schließkraft:", fg_color="transparent", font=("Arial", 14))
        self.slider7label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.amount7label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider7var.get()):.1f}", NEWTON), fg_color="transparent", font=("Arial", 14))
        self.amount7label.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        self.slider8var = tk.DoubleVar(value=self.generated_paremeters[0][10])
        self.slider8label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Maximales Schneckendrehmoment:", fg_color="transparent", font=("Arial", 14))
        self.slider8label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.amount8label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider8var.get()):.1f}", NEWTONMETER), fg_color="transparent", font=("Arial", 14))
        self.amount8label.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        self.slider9var = tk.DoubleVar(value=self.generated_paremeters[0][11])
        self.slider9label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Mittleres Schneckendrehmoment:", fg_color="transparent", font=("Arial", 14))
        self.slider9label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.amount9label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider9var.get()):.1f}", NEWTONMETER), fg_color="transparent", font=("Arial", 14))
        self.amount9label.grid(row=7, column=1, padx=10, pady=5, sticky="w")
        self.slider12var = tk.DoubleVar(value=self.generated_paremeters[0][12])
        self.slider12label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Finale Schneckenposition:", fg_color="transparent", font=("Arial", 14))
        self.slider12label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.amount12label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider12var.get()):.1f}", CM), fg_color="transparent", font=("Arial", 14))
        self.amount12label.grid(row=8, column=1, padx=10, pady=5, sticky="w")
        
        #Quality Widgets
        self.qual_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray")
        self.qual_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsw")
        self.qual_label = customtkinter.CTkLabel(self.qual_frame, text="Qualitätsergebnis:", fg_color="transparent", font=("Arial", 16,"bold"))
        self.qual_label.grid(row=0, column=0, padx=10, pady=10)
        self.border_frame = customtkinter.CTkFrame(self.qual_frame, fg_color="transparent")
        self.border_frame.grid(row=1, column=0, padx=10, pady=10)
        self.quality_category_label = customtkinter.CTkLabel(self.border_frame, text="", fg_color="transparent")
        self.quality_category_label.grid(row=0, column=0, padx=3, pady=3)
        self.qual_value_label = customtkinter.CTkLabel(self.qual_frame, text="U\u2080-Wert:", fg_color="transparent", font=("Arial", 16,"bold"))
        self.qual_value_label.grid(row=0, column=1, padx=10, pady=10)
        self.produce_label = customtkinter.CTkLabel(self.qual_frame, text="", fg_color="transparent", font=("Arial",18, "bold"))
        self.produce_label.grid(row=1, column=1, padx=10, pady=10)
        
        segmente = [
            ("Ausschuss\nU₀ < 0,4",           "#C00000", 100),
            ("Akzeptabel\n0,4 ≤ U₀ < 0,45",  "#FFC000", 130),
            ("Sollbereich\n0,45 ≤ U₀ ≤ 0,5", "#8ED973", 130),
            ("Ineffizient\nU₀ > 0,5",         "#E97132", 100),
        ]

        self.qualitySkala_label = customtkinter.CTkLabel(self.qual_frame, text="Qualitätsskala:", font=("Arial", 16,"bold"))
        self.qualitySkala_label.grid(row=0, column=2, columnspan=len(segmente), sticky="w", padx=10, pady=10)

        for col, (text, farbe, breite) in enumerate(segmente):
            seg = customtkinter.CTkFrame(self.qual_frame, fg_color=farbe, corner_radius=4, width=breite, height=50)
            seg.grid(row=1, column=col+2, padx=2, pady=(0, 10), sticky="nsew")
            seg.grid_propagate(False)
            customtkinter.CTkLabel(seg, text=text, font=customtkinter.CTkFont(size=12), text_color="black",justify="center").grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg.grid_rowconfigure(0, weight=1)
            seg.grid_columnconfigure(0, weight=1)
        
        
        # Production Widgets
        self.production_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray")
        self.production_frame.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nsw")
        self.production_frame.grid_rowconfigure((0), weight=1)
        self.producing_button = customtkinter.CTkButton(self.production_frame, text="1.\nProduktion starten", command=self.set_kn_vals, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
        self.producing_button.grid(row=0, column=1, padx=10, pady=10, sticky="ns")
        self.produce_button = customtkinter.CTkButton(self.production_frame, text="2.\nQualität vorhersagen ", command=self.produce_func, state="disabled", corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
        self.produce_button.grid(row=0, column=2, padx=10, pady=10, sticky="ns")
        self.algoOptionVar = customtkinter.StringVar(value="Partikelschwarmoptimierung")
        self.algoOption = customtkinter.CTkOptionMenu(self.production_frame, values=["Partikelschwarmoptimierung", "Genetischer Algorithmus", "Simulierte Abkühlung"],variable=self.algoOptionVar, corner_radius=12,fg_color="#dce8f5",text_color="#1a1a1a")
        self.algoOption.set("Partikelschwarmoptimierung")
        self.algoOption.grid(row=0, column=3, padx=10, pady=10, sticky="ns")
        self.ai_button = customtkinter.CTkButton(self.production_frame, text="3.\nEinstellempfehlung generieren", command=self.use_algo, state="disabled", corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
        self.ai_button.grid(row=0, column=4, padx=10, pady=10, sticky="ns")
        ai_tooltip_string = ("{Die Parameter werden mit ", self.algoOptionVar.get(), " generiert. Dieser stochastische Optimierungsalgorithmus produziert nicht-deterministische Ergebnisse.}")
        self.ai_tooltip = CTkToolTip(self.ai_button, message=ai_tooltip_string)
        
        # Linse fertigung:
        self.lense_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray")
        self.lense_frame.grid(row=2, column=1, padx=10, pady=(10, 0), sticky="nsw")
        self.lense_frame.grid_forget()
        self.lens_pic = customtkinter.CTkImage(light_image=Image.open('graphics/lens.png'), dark_image=Image.open('graphics/lens.png'), size=(150,150))
        self.lens_image_label = customtkinter.CTkLabel(self.lense_frame, text="", image=self.lens_pic)
        self.lens_image_label.grid(row=0, column=0, rowspan=6, padx=20, pady=10)
        self.lens_produced_label1 = customtkinter.CTkLabel(self.lense_frame, text="Status:", fg_color="transparent", font=("Arial",16, "bold") )
        self.lens_produced_label1.grid(row=0, column=1, padx=10, pady=0, sticky="w")
        self.lens_produced_label2 = customtkinter.CTkLabel(self.lense_frame, text="Linse produziert", fg_color="transparent",font=("Arial",16, "bold"))
        self.lens_produced_label2.grid(row=0, column=2, padx=10, pady=0, sticky="w")
        self.article_produced_label1 = customtkinter.CTkLabel(self.lense_frame, text="Artikelnummer:", fg_color="transparent",font=("Arial",14, "bold"))
        self.article_produced_label1.grid(row=1, column=1, padx=10, pady=0, sticky="w")
        self.article_produced_label2 = customtkinter.CTkLabel(self.lense_frame, text="LNS-4712", fg_color="transparent")
        self.article_produced_label2.grid(row=1, column=2, padx=10, pady=0, sticky="w")
        self.material_label1 = customtkinter.CTkLabel(self.lense_frame, text="Material:", fg_color="transparent", font=("Arial",14, "bold"))
        self.material_label1.grid(row=2, column=1, padx=10, pady=0, sticky="w")
        self.material_label2 = customtkinter.CTkLabel(self.lense_frame, text="PMMA", fg_color="transparent")
        self.material_label2.grid(row=2, column=2, padx=10, pady=0, sticky="w")
        self.current_iter = "000"
        self.current_charge = f"{DATE}-M1-{self.current_iter}"
        self.charge_label1 = customtkinter.CTkLabel(self.lense_frame, text="Charge: ", fg_color="transparent", font=("Arial",14, "bold"))
        self.charge_label1.grid(row=3, column=1, padx=10, pady=0, sticky="w")
        self.charge_label2 = customtkinter.CTkLabel(self.lense_frame, text=self.current_charge, fg_color="transparent")
        self.charge_label2.grid(row=3, column=2, padx=10, pady=0, sticky="w")
        
    def slider_change(self):
        self.produce_button.configure(state="disabled")
        self.quality_category_label.configure(text="")
        self.quality_category_label.configure(fg_color="transparent")
        self.produce_label.configure(text="")
            
        mode = customtkinter.get_appearance_mode()
        tuple = customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"]
        colour = tuple[0] if mode == "Light" else tuple[1]
        self.border_frame.configure(fg_color=colour, border_width=0,border_color=colour)
        
    def update_charge(self):
        self.current_iter = str(int(self.current_iter) + 1).zfill(len(self.current_iter))
        self.current_charge = f"{DATE}-M1-{self.current_iter}"
        self.charge_label2.configure(text=self.current_charge)
        
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
        self.amount3label.configure(text=(f"{(self.generated_paremeters)[0][6]:.1f}", SECONDS))
        self.amount4label.configure(text=(f"{(self.generated_paremeters)[0][7]:.1f}", SECONDS))
        self.amount5label.configure(text=(f"{(self.generated_paremeters)[0][8]:.1f}", SECONDS))
        self.amount7label.configure(text=(f"{(self.generated_paremeters)[0][9]:.1f}", NEWTON))
        self.amount8label.configure(text=(f"{(self.generated_paremeters)[0][10]:.1f}", NEWTONMETER))
        self.amount9label.configure(text=(f"{(self.generated_paremeters)[0][11]:.1f}", NEWTONMETER))
        self.amount12label.configure(text=(f"{(self.generated_paremeters)[0][12]:.1f}", CM))
        self.lense_frame.grid()
        self.lense_frame.grid(row=2, column=1, padx=10, pady=(10, 0), sticky="nsw")
        self.update_charge()
        self.slider_change()
        self.produce_button.configure(state="normal")
        
    def produce_func(self):
        self.vars = []
        self.vars.append(self.slider1var.get()) 
        self.vars.append(self.slider2var.get()) 
        self.vars.append(self.slider6var.get()) 
        self.vars.append(self.slider10var.get()) 
        self.vars.append(self.slider11var.get()) 
        self.vars.append(self.slider13var.get()) 


        self.vars = np.array(self.vars).reshape(1, -1)
        self.vars = self.scaler2.transform(self.vars)
        self.prediction = self.model.predict(self.vars).item()
        
        if(self.prediction < 0):
            self.prediction = 0
        if(self.prediction > 10):
            self.prediction = 10
        
        #self.prediction = round(self.prediction, 1)
        
        quality_cat = self.judge_quality()
        self.prediction = self.prediction * 0.1
        msg1 = f"{self.prediction:.2f}"
        self.produce_label.configure(text=msg1)
        msg2 = f"{quality_cat}"
        self.quality_category_label.configure(text=msg2)
        if(quality_cat == "Ausschuss"):
            self.border_frame.configure(fg_color="#C00000", border_width=2,border_color="black")
        if(quality_cat == "Akzeptabel"):
            self.border_frame.configure(fg_color="#FFC000", border_width=2,border_color="black")
        if(quality_cat == "Sollbereich"):
            self.border_frame.configure(fg_color="#8ED973", border_width=2,border_color="black")
        if(quality_cat == "Ineffizient"):
            self.border_frame.configure(fg_color="#E97132", border_width=2,border_color="black")
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

        self.lense_frame.grid_forget()
        
        solution = self.scaler2.inverse_transform(solution_std)
        self.transformed_solution = solution.squeeze()
        self.transformed_solution = self.transformed_solution[0:6]
        self.Optparamterlabel0 = customtkinter.CTkLabel(self.einstellParam_frame, text="Empfehlung", fg_color="transparent", font=("Arial", 18, "bold"))
        self.Optparamterlabel0.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.optParameterLabels = []
        for i, value in enumerate(self.transformed_solution):
            optParameterLabel = customtkinter.CTkLabel(self.einstellParam_frame, text=f"{value:.1f}", fg_color="transparent", font=("Arial", 14))
            optParameterLabel.grid(row=i + 2, column=3, padx=10, pady=5, sticky="w")
            self.optParameterLabels.append(optParameterLabel)
        self.useOptimizedButton = customtkinter.CTkButton(self.einstellParam_frame, text="4.\nEmpfehlung übernehmen", command=self.useOptimizedFunc, corner_radius=12,border_width=1,border_color="#a0b4c8",fg_color="#dce8f5",hover_color="#c5d8ed",text_color="#1a1a1a")
        self.useOptimizedButton.grid(row=8, column=3, padx=10, pady=5)
        
    def useOptimizedFunc(self):
        self.slider1.set((self.transformed_solution)[0])
        self.amount1label.configure(text=(f"{(self.transformed_solution)[0]:.1f}", CELSIUS))
        self.slider2.set((self.transformed_solution)[1])
        self.amount2label.configure(text=(f"{(self.transformed_solution)[1]:.1f}", CELSIUS))
        self.slider6.set((self.transformed_solution)[2])
        self.amount6label.configure(text=(f"{(self.transformed_solution)[2]:.1f}", NEWTON))
        self.slider10.set((self.transformed_solution)[3])
        self.amount10label.configure(text=(f"{(self.transformed_solution)[3]:.1f}", BAR))
        self.slider11.set((self.transformed_solution)[4])
        self.amount11label.configure(text=(f"{(self.transformed_solution)[4]:.1f}", BAR))
        self.slider13.set((self.transformed_solution)[5])
        self.amount13label.configure(text=(f"{(self.transformed_solution)[5]:.1f}", CM3))
        self.produce_button.configure(state="disabled")

    def judge_quality(self):
        if(self.prediction < 4):
            return "Ausschuss"
        elif(self.prediction < 4.5):
            return "Akzeptabel"
        elif(self.prediction <= 5):
            return "Sollbereich"
        else:
            return "Ineffizient"

app = App()
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

# Fenstergröße setzen
app.geometry(f"{screen_width}x{screen_height}+0+0")
app.mainloop()