import datetime
import pickle
import time
from pathlib import Path

import customtkinter
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import GifLabel
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
from helper.shap_executer import shap_explainer

# Konstanten
STEPS=1000 # Nur in dem Wertebereich der Schritte können den Slidern Werte zugeordnet werden, also muss die Schrittmenge hoch (oder nonexistent) sein damit die Qualität der optimierten Parameter erreicht werden kann
DATE=datetime.datetime.now().strftime("%Y%m%d")
CELSIUS="°C"
SECONDS="s"
NEWTON="N"
NEWTONMETER="N\u22C5m"
BAR="bar"
CM="cm"
CM3="cm\u00B3"
BACKGROUND_COLOR="#ffffff"
FONT_SMALL = ("Arial",14, "bold")
FONT_SMALL_LIGHT = ("Arial",14)
FONT_MEDIUM = ("Arial",16, "bold")
FONT_MEDIUM_LIGHT = ("Arial",16)
FONT_LARGE = ("Arial", 18, "bold")
FONT_LARGE_LIGHT = ("Arial", 18)
FONT_EXTRALARGE = ("Arial", 20, "bold")
RED ="#C00000"
YELLOW = "#FFC000"
GREEN = "#8ED973"
ORANGE = "#E97132"
TURQUOISE_HELL = "#DDEEEF"
TURQUOISE = "#5ab4b4"
BLACK = "#2d2d00"
GREY = "#7d7d7d"
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        #self.geometry("1300x850")
        #width = self.winfo_screenwidth()
        #height = self.winfo_screenheight()
        #geometry = str(width) + "x" + str(height)
        #self.geometry(geometry)
        self.title("Spritzguss Demonstrator")
        self.configure(fg_color=BACKGROUND_COLOR)
        
        self.min_max_scaler = create_scaler()
        self.scaler2 = create_scaler2()
        self.model = pickle.load(open("nn/neural-net-new.sav", 'rb'))
        self.scores = []
        self.kn = create_kn_classifier(6, self.min_max_scaler)
        
        self.wirksam_logo = customtkinter.CTkImage(light_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), dark_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), size=(160, 87))
        
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
        self.tabview = customtkinter.CTkTabview(self, fg_color=BACKGROUND_COLOR, border_color=BLACK, text_color=BLACK, segmented_button_selected_color=TURQUOISE, segmented_button_selected_hover_color=TURQUOISE_HELL)#, corner_radius=0, , width=self.winfo_screenwidth()
        self.tabview.pack(padx=0, pady=0)

        self.tab1 = self.tabview.add("1. KI live testen")
        self.tab5 = self.tabview.add("2. KI hinterfragen")
        self.tab2 = self.tabview.add("3. Einflussfaktoren untersuchen")
        self.tab3 = self.tabview.add("4. KI selbst trainieren")
        #self.tab4 =self.tabview.add("4. Optimierungsalgorithmus")
        self.tab6 =self.tabview.add("5. Funktionsweise verstehen")
        self.tabview.set("1. KI live testen")
        self.tab1.configure(fg_color=BACKGROUND_COLOR, border_color=BACKGROUND_COLOR, border_width=10, corner_radius=12)
        self.tab2.configure(fg_color=BACKGROUND_COLOR, border_color=BACKGROUND_COLOR, border_width=10, corner_radius=12)
        self.tab3.configure(fg_color=BACKGROUND_COLOR, border_color=BACKGROUND_COLOR, border_width=10, corner_radius=12)
        #self.tab4.configure(fg_color=BACKGROUND_COLOR, border_color=BACKGROUND_COLOR)
        self.tab5.configure(fg_color=BACKGROUND_COLOR, border_color=BACKGROUND_COLOR, border_width=10, corner_radius=12)
        self.tab6.configure(fg_color=BACKGROUND_COLOR, border_color=BACKGROUND_COLOR, border_width=10, corner_radius=12)
        
        self.tabview._segmented_button.configure(corner_radius=8, width=140, border_width=10, font=FONT_MEDIUM)
        
        # Einstellungen ändern Tab
        
        def createProdLaufFrame():
            self.eAeFrame.forget()
            self.prodLaufFrame = customtkinter.CTkFrame(self.tab2, width=200, height=200, fg_color=BACKGROUND_COLOR)
            self.prodLaufFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
            self.returneAe_button = customtkinter.CTkButton(self.prodLaufFrame, text="Zurück zur Auswahl", command=zurueckAuswahl1, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.returneAe_button.grid(row=0, column=0, padx=20, sticky="nw")
            self.vermFrame = customtkinter.CTkFrame(self.prodLaufFrame, fg_color=BACKGROUND_COLOR, corner_radius=8)
            self.vermFrame.grid(row=1, column=1, padx=20, pady=20, sticky="ew")
            
            self.kreis1 = customtkinter.CTkLabel(self.vermFrame, text=str(1), width=36, height=36, corner_radius=18, fg_color=TURQUOISE, text_color="white", font=FONT_SMALL)
            self.kreis1.grid(row=0, column=0, padx=(20, 8), pady=15)
            self.vermLabel1 = customtkinter.CTkLabel(self.vermFrame, text="Deine Vermutung", font=customtkinter.CTkFont(size=13), text_color="black")
            self.vermLabel1.grid(row=0, column=1, sticky="w")
            self.linie1 = customtkinter.CTkFrame(self.vermFrame, height=2, fg_color="#333333", width=80)
            self.linie1.grid(row=0, column=2, padx=15, sticky="ew")
            
            self.kreis2 = customtkinter.CTkLabel(self.vermFrame, text=str(2), width=36, height=36, corner_radius=18, fg_color="#d0d0d0", text_color="#333333", font=FONT_SMALL)
            self.kreis2.grid(row=0, column=3, padx=(20, 8), pady=15)
            self.vermLabel2 = customtkinter.CTkLabel(self.vermFrame, text="Ergebnis ansehen", font=customtkinter.CTkFont(size=13), text_color="black")
            self.vermLabel2.grid(row=0, column=4, sticky="w")
            self.linie2 = customtkinter.CTkFrame(self.vermFrame, height=2, fg_color="#333333", width=80)
            self.linie2.grid(row=0, column=5, padx=15, sticky="ew")
            
            self.kreis3 = customtkinter.CTkLabel(self.vermFrame, text=str(3), width=36, height=36, corner_radius=18, fg_color="#d0d0d0", text_color="#333333", font=FONT_SMALL)
            self.kreis3.grid(row=0, column=6, padx=(20, 8), pady=15)
            self.vermLabel3 = customtkinter.CTkLabel(self.vermFrame, text="Erklärung", font=customtkinter.CTkFont(size=13), text_color="black")
            self.vermLabel3.grid(row=0, column=7, sticky="w")
            
            self.denkFrame = customtkinter.CTkFrame(self.prodLaufFrame, border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
            self.denkFrame.grid(row=2, column=1, padx=10, pady=10, sticky="nsw")
            self.denkFrameCap1 = customtkinter.CTkLabel(self.denkFrame, text="Was denkst du – wie viele Produktionsläufe braucht die KI?", fg_color="transparent", font=FONT_LARGE)
            self.denkFrameCap1.grid(row=0, column=0, padx=20, pady=10, sticky="w")
            self.denkFrameCap2 = customtkinter.CTkLabel(self.denkFrame, text="Die KI lernt aus vergangenen Produktionsläufen und den zugehörigen Qualitätsprüfungen. Doch wie viele Daten braucht sie,\n um zuverlässige Vorhersagen treffen zu können? Wähle eine Datenmenge und schätze anschließend ein,\n wie gut die KI damit sein wird.", fg_color="transparent", justify="left")
            self.denkFrameCap2.grid(row=1, column=0, padx=20, pady=10, sticky="w")
            self.denkFrameCap3 = customtkinter.CTkLabel(self.denkFrame, text="Wähle eine Datenmenge:", fg_color="transparent", font=FONT_LARGE)
            self.denkFrameCap3.grid(row=2, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.denkFrameCapFrame = customtkinter.CTkFrame(self.denkFrame, border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
            self.denkFrameCapFrame.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="nsw")
            
            def do_denkFrameButton1():
                self.denkFrameButton1.configure(fg_color=TURQUOISE)
                self.denkFrameButton2.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton3.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton4.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameCap_button_var = customtkinter.StringVar(value="250 Läufe")
            
            def do_denkFrameButton2():
                self.denkFrameButton2.configure(fg_color=TURQUOISE)
                self.denkFrameButton1.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton3.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton4.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameCap_button_var = customtkinter.StringVar(value="500 Läufe")
                
            def do_denkFrameButton3():
                self.denkFrameButton3.configure(fg_color=TURQUOISE)
                self.denkFrameButton1.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton2.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton4.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameCap_button_var = customtkinter.StringVar(value="1000 Läufe")
                
            def do_denkFrameButton4():
                self.denkFrameButton4.configure(fg_color=TURQUOISE)
                self.denkFrameButton1.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton2.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton3.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameCap_button_var = customtkinter.StringVar(value="1451 Läufe")
            
            self.denkFrameButton1 = customtkinter.CTkButton(self.denkFrameCapFrame, text="250\nLäufe", command=do_denkFrameButton1,corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.denkFrameButton1.grid(row=0, column=0, padx=20, pady=10)
            self.denkFrameButton2 = customtkinter.CTkButton(self.denkFrameCapFrame, text="500\nLäufe", command=do_denkFrameButton2, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.denkFrameButton2.grid(row=0, column=1, padx=20, pady=10)
            self.denkFrameButton3 = customtkinter.CTkButton(self.denkFrameCapFrame, text="1000\nLäufe", command=do_denkFrameButton3, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.denkFrameButton3.grid(row=0, column=2, padx=20, pady=10)
            self.denkFrameButton4 = customtkinter.CTkButton(self.denkFrameCapFrame, text="1451\nLäufe", command=do_denkFrameButton4, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.denkFrameButton4.grid(row=0, column=3, padx=20, pady=10)
            
            self.denkFrameButton1.configure(fg_color=TURQUOISE)
            self.denkFrameButton2.configure(fg_color=TURQUOISE_HELL)
            self.denkFrameButton3.configure(fg_color=TURQUOISE_HELL)
            self.denkFrameButton4.configure(fg_color=TURQUOISE_HELL)
            self.denkFrameCap_button_var = customtkinter.StringVar(value="250 Läufe")
            
            self.denkFrameCap4 = customtkinter.CTkLabel(self.denkFrame, text="Wie gut wird die KI mit dieser Datenmenge sein?", fg_color="transparent", font=FONT_LARGE)
            self.denkFrameCap4.grid(row=4, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.denkFrameCapFrame2 = customtkinter.CTkFrame(self.denkFrame, border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
            self.denkFrameCapFrame2.grid(row=5, column=0, padx=10, pady=10, sticky="nsw")
            def do_denkFrameButton5():
                self.denkFrameButton5.configure(fg_color=TURQUOISE)
                self.denkFrameButton6.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton7.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Eher schlecht")
            
            def do_denkFrameButton6():
                self.denkFrameButton6.configure(fg_color=TURQUOISE)
                self.denkFrameButton5.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton7.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Mittel")
                
            def do_denkFrameButton7():
                self.denkFrameButton7.configure(fg_color=TURQUOISE)
                self.denkFrameButton5.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameButton6.configure(fg_color=TURQUOISE_HELL)
                self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Gut")
            
            self.denkFrameButton5 = customtkinter.CTkButton(self.denkFrameCapFrame2, text="Eher schlecht\n(unter 60 % richtig\nvorhergesagt)", command=do_denkFrameButton5,corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.denkFrameButton5.grid(row=0, column=0, padx=20, pady=10)
            self.denkFrameButton6 = customtkinter.CTkButton(self.denkFrameCapFrame2, text="Mittel\n(60–80 % richtig\nvorhergesagt)", command=do_denkFrameButton6, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.denkFrameButton6.grid(row=0, column=1, padx=20, pady=10)
            self.denkFrameButton7 = customtkinter.CTkButton(self.denkFrameCapFrame2, text="Gut\n(über 80 % richtig\nvorhergesagt)", command=do_denkFrameButton7, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.denkFrameButton7.grid(row=0, column=2, padx=20, pady=10)
            
            self.denkFrameButton5.configure(fg_color=TURQUOISE)
            self.denkFrameButton6.configure(fg_color=TURQUOISE_HELL)
            self.denkFrameButton7.configure(fg_color=TURQUOISE_HELL)
            self.denkFrameCap_button_var2 = customtkinter.StringVar(value="Eher schlecht")
            self.denkFrameExit = customtkinter.CTkButton(self.denkFrame, text="Ergebnis anzeigen", command=create_ergebnis_ansehen_frame, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", font=FONT_LARGE, text_color_disabled="#585858")
            self.denkFrameExit.grid(row=5, column=3, padx=20, pady=10)
        
        def create_ergebnis_ansehen_frame():
            self.denkFrame.destroy()
            self.kreis1.configure(fg_color="#d0d0d0", text_color="#333333")
            self.kreis2.configure(fg_color=TURQUOISE, text_color="white")
            self.ergebnisAnsehenFrame = customtkinter.CTkFrame(self.prodLaufFrame, border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
            self.ergebnisAnsehenFrame.grid(row=2, column=1, padx=10, pady=(10, 0), sticky="nsw")
            self.ergebnisAnsehenLabel1 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=(f"Ergebnis: KI trainiert mit {(self.denkFrameCap_button_var.get())}"), fg_color="transparent", font=FONT_LARGE)
            self.ergebnisAnsehenLabel1.grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.ergebnisAnsehenLabel2 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("Die KI wurde mit 300 neuen Produktionsläufen getestet, die das KI-Modell noch nicht kennt.\nDie Werte zeigen, wie zuverlässig die einzelnen Qualitätsklassen erkannt werden.\nDie Gesamttrefferquote zeigt, wie viele Vorhersagen insgesamt richtig waren."), fg_color="transparent", justify="left")
            self.ergebnisAnsehenLabel2.grid(row=1, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            seg1 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color=RED, corner_radius=4, width=100, height=50)
            seg1.grid(row=2, column=0, padx=2, pady=(0, 10), sticky="nse")
            seg1.grid_propagate(False)
            self.segLabel1 = customtkinter.CTkLabel(seg1, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel1.grid(row=0, column=0, sticky="nswe", padx=4, pady=4)
            seg1.grid_rowconfigure(0, weight=1)
            seg1.grid_columnconfigure(0, weight=1)
            
            seg2 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color=YELLOW, corner_radius=4, width=100, height=50)
            seg2.grid(row=2, column=1, padx=2, pady=(0, 10), sticky="nsew")
            seg2.grid_propagate(False)
            self.segLabel2 = customtkinter.CTkLabel(seg2, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel2.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg2.grid_rowconfigure(0, weight=1)
            seg2.grid_columnconfigure(0, weight=1)
            
            seg3 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color=GREEN, corner_radius=4, width=100, height=50)
            seg3.grid(row=2, column=2, padx=2, pady=(0, 10), sticky="nsew")
            seg3.grid_propagate(False)
            self.segLabel3 = customtkinter.CTkLabel(seg3, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel3.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg3.grid_rowconfigure(0, weight=1)
            seg3.grid_columnconfigure(0, weight=1)
            
            seg4 = customtkinter.CTkFrame(self.ergebnisAnsehenFrame, fg_color=ORANGE, corner_radius=4, width=100, height=50)
            seg4.grid(row=2, column=3, padx=2, pady=(0, 10), sticky="nsw")
            seg4.grid_propagate(False)
            self.segLabel4 = customtkinter.CTkLabel(seg4, text="", font=customtkinter.CTkFont(size=12), text_color="black",justify="center")
            self.segLabel4.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            seg4.grid_rowconfigure(0, weight=1)
            seg4.grid_columnconfigure(0, weight=1)
            
            self.ergebnisAnsehenLabel3 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("Gesamttrefferquote:"), fg_color="transparent", font=FONT_LARGE)
            self.ergebnisAnsehenLabel3.grid(row=3, column=0, padx=20, pady=10, sticky="w")
            self.progressbar = customtkinter.CTkProgressBar(self.ergebnisAnsehenFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",border_width=0)
            self.progressbar.grid(row=3, column=1, columnspan=3, padx=20, pady=10, sticky="w")
            if(self.denkFrameCap_button_var.get() == "250 Läufe"):
                self.progressbar.configure(progress_color=RED)
                self.progressbar.set(0.41)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("41%"), fg_color="transparent", font=FONT_LARGE)
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("250 Produktionsläufe liefern der KI erst wenige Beispiele zum Lernen. Deshalb verwechselt sie die Qualitätsklassen noch häufig und macht falsche Vorhersagen."), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n25,9%")
                self.segLabel2.configure(text="Akzeptabel:\n40,9%")
                self.segLabel3.configure(text="Sollbereich:\n60,8%")
                self.segLabel4.configure(text="Ineffizient:\n36,1%")
            if(self.denkFrameCap_button_var.get() == "500 Läufe"):
                self.progressbar.configure(progress_color=YELLOW)
                self.progressbar.set(0.79)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("79%"), fg_color="transparent", font=FONT_LARGE)
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("500 Produktionsläufe reichen für gute Vorhersagen. Trotzdem verwechselt die KI einzelne Qualitätsklassen und macht noch falsche Vorhersagen."), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n61,9%")
                self.segLabel2.configure(text="Akzeptabel:\n82,2%")
                self.segLabel3.configure(text="Sollbereich:\n80,4%")
                self.segLabel4.configure(text="Ineffizient:\n83,2%")
            if(self.denkFrameCap_button_var.get() == "1000 Läufe"):
                self.progressbar.configure(progress_color=YELLOW)
                self.progressbar.set(0.83)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("83%"), fg_color="transparent", font=FONT_LARGE)
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("1000 Produktionsläufe liefern genügend Beispiele zum Lernen. Die KI erkennt die meisten Zusammenhänge nun zuverlässiger. Falsche Vorhersagen treten nur noch gelegentlich auf."), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n68,9%")
                self.segLabel2.configure(text="Akzeptabel:\n84,6%")
                self.segLabel3.configure(text="Sollbereich:\n85,6%")
                self.segLabel4.configure(text="Ineffizient:\n88,5%")
            if(self.denkFrameCap_button_var.get() == "1451 Läufe"):
                self.progressbar.configure(progress_color=GREEN)
                self.progressbar.set(0.84)
                self.ergebnisAnsehenLabel4 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("84%"), fg_color="transparent", font=FONT_LARGE)
                self.ergebnisAnsehenLabel5 = customtkinter.CTkLabel(self.ergebnisAnsehenFrame, text=("1451 Produktionsläufe bringen gegenüber 1000 Läufen nur noch eine geringe Verbesserung. Die KI erkennt die Zusammenhänge bereits sehr zuverlässig. Zusätzliche Daten erhöhen die Trefferquote kaum noch."), fg_color="transparent")
                self.segLabel1.configure(text="Ausschuss:\n73,0%")
                self.segLabel2.configure(text="Akzeptabel:\n84,5%")
                self.segLabel3.configure(text="Sollbereich:\n87,2%")
                self.segLabel4.configure(text="Ineffizient:\n90,0%")
            self.ergebnisAnsehenLabel4.grid(row=3, column=4, padx=20, pady=10, sticky="w")
            self.ergebnisAnsehenLabel5.grid(row=4, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            self.ergebnisAnsehenFrameExit = customtkinter.CTkButton(self.ergebnisAnsehenFrame, text="Was bedeutet das?", command=create_erklaerung_frame, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", font=FONT_LARGE, text_color_disabled="#585858")
            self.ergebnisAnsehenFrameExit.grid(row=6, column=3, padx=20, pady=10)
        
        def create_erklaerung_frame():
            self.ergebnisAnsehenFrame.destroy()
            #self.ergebnisAnsehenFrame.grid_forget() 
            self.kreis2.configure(fg_color="#d0d0d0", text_color="#333333")
            self.kreis3.configure(fg_color=TURQUOISE, text_color="white")
            self.erklaerungFrame = customtkinter.CTkFrame(self.prodLaufFrame, border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
            self.erklaerungFrame.grid(row=2, column=1, padx=10, pady=(10, 0), sticky="nsw")
            self.erklaerungLabel1 = customtkinter.CTkLabel(self.erklaerungFrame, text=("Ergebnis: So verändert die Datenmenge die Qualität der KI"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungLabel1.grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            self.erklaerungLabel2 = customtkinter.CTkLabel(self.erklaerungFrame, text=("Grundsätzlich gilt: Je mehr Produktionsläufe und Qualitätstests die KI kennt, desto zuverlässiger sind ihre Vorhersagen."), fg_color="transparent")
            self.erklaerungLabel2.grid(row=1, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel1 = customtkinter.CTkLabel(self.erklaerungFrame, text=("250 Läufe"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel1.grid(row=2, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar1 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color=RED,border_width=0)
            self.erklaerungBar1.grid(row=2, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar1.set(0.41)
            self.erklaerungBarLabel11 = customtkinter.CTkLabel(self.erklaerungFrame, text=("41%"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel11.grid(row=2, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel2 = customtkinter.CTkLabel(self.erklaerungFrame, text=("500 Läufe"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel2.grid(row=3, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar2 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color=YELLOW,border_width=0)
            self.erklaerungBar2.grid(row=3, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar2.set(0.79)
            self.erklaerungBarLabel22 = customtkinter.CTkLabel(self.erklaerungFrame, text=("79%"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel22.grid(row=3, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel3 = customtkinter.CTkLabel(self.erklaerungFrame, text=("1000 Läufe"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel3.grid(row=4, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar3 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color=GREEN,border_width=0)
            self.erklaerungBar3.grid(row=4, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar3.set(0.83)
            self.erklaerungBarLabel33 = customtkinter.CTkLabel(self.erklaerungFrame, text=("83%"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel33.grid(row=4, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungBarLabel4 = customtkinter.CTkLabel(self.erklaerungFrame, text=("1451 Läufe"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel4.grid(row=5, column=0, padx=20, pady=10, sticky="w")
            self.erklaerungBar4 = customtkinter.CTkProgressBar(self.erklaerungFrame, orientation="horizontal", corner_radius=0,fg_color="#a8a8a8",progress_color=GREEN,border_width=0)
            self.erklaerungBar4.grid(row=5, column=1, columnspan=2, padx=20, pady=10, sticky="w")
            self.erklaerungBar4.set(0.84)
            self.erklaerungBarLabel44 = customtkinter.CTkLabel(self.erklaerungFrame, text=("84%"), fg_color="transparent", font=FONT_LARGE)
            self.erklaerungBarLabel44.grid(row=5, column=3, padx=20, pady=10, sticky="w")
            
            self.erklaerungLabel3 = customtkinter.CTkLabel(self.erklaerungFrame, text=("Damit eine KI gute Vorhersagen machen kann, sind die Datenmenge und die Datenqualität wichtig.\nGute Daten sind vollständig. Das heißt, alle benötigten Werte sind vorhanden, zum Beispiel\nMaschineneinstellungen und Ergebnisse der Qualitätsprüfung.\n\nDu hast gesehen: Mehr Daten helfen, aber irgendwann werden die Verbesserungen kleiner.\nDie zusätzlichen 451 Produktionsläufe erhöhen die Trefferquote nur noch wenig. Für eine bessere KI\nreicht es deshalb nicht immer aus, einfach mehr Daten zu sammeln. Auch das KI-Modell kann angepasst\noder ein anderes Modell ausprobiert werden. Es gibt nicht die eine KI-Lösung für jede Aufgabe.\n\nWelche weiteren Stellschrauben gibt es?\nWechsle zu „4. KI selbst trainieren“ und probiere verschiedene Einstellungen des KI-Modells aus."), fg_color="transparent", justify="left")
            self.erklaerungLabel3.grid(row=6, column=0, columnspan=4, padx=20, pady=10, sticky="w")
            
            self.erklaerungReturnButton = customtkinter.CTkButton(self.erklaerungFrame, text="Andere Datenmenge testen", command=andereDatenmengen, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", font=FONT_LARGE, text_color_disabled="#585858")
            self.erklaerungReturnButton.grid(row=7, column=1, padx=20, pady=10)
            
        def andereDatenmengen():
            self.erklaerungFrame.destroy()
            self.prodLaufFrame.destroy()
            createProdLaufFrame()
        
        def zurueckAuswahl1():
            self.prodLaufFrame.destroy()
            self.eAeFrame.grid()
        
        def zurueckAuswahl2():
            self.scoresFrame.destroy()
            self.eAeFrame.grid()
            
        def showBestQuality():
            self.eAeFrame.forget()
            self.scoresFrame = customtkinter.CTkFrame(self.tab2, width=200, height=200, fg_color=BACKGROUND_COLOR, border_color="#1a1a1a")
            self.scoresFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
            self.scoresSubFrame1 = customtkinter.CTkFrame(self.scoresFrame, width=200, height=200, fg_color=BACKGROUND_COLOR, border_color="#1a1a1a")
            self.scoresSubFrame1.grid(row=0, column=0, padx=20, pady=5, sticky="nsew")
            
            self.returnScores_button = customtkinter.CTkButton(self.scoresSubFrame1, text="Zurück", command=zurueckAuswahl2, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
            self.returnScores_button.grid(row=0, column=0, padx=20, sticky="nw")
            self.scoresCaptionLabel = customtkinter.CTkLabel(self.scoresSubFrame1, text=(" Wie findet das mathematische Modell die besten Einstellungen?"), fg_color="transparent", justify="left", font=FONT_MEDIUM)
            self.scoresCaptionLabel.grid(row=0, column=1, padx=20, pady=5, sticky="we")
            self.scoresLabel = customtkinter.CTkLabel(self.scoresSubFrame1, text=("Der Optimierungsalgorithmus ist ein mathematisches Verfahren. Er testet Schritt für Schritt verschiedene Einstellungen der Maschine.\nIn jedem Durchlauf, einer sogenannten Iteration, sagt das KI-Modell die erwartete Qualität voraus. Gute Einstellungskombinationen werden\nausgewählt, miteinander kombiniert und leicht verändert. So entstehen nach und nach neue Vorschläge. Mit jedem Durchlauf findet das Verfahren\nbessere Einstellungen für die gewünschte Produktqualität. Drücke auf „Starten“, um zu sehen, wie das Verfahren arbeitet."), fg_color="transparent", justify="left", font=FONT_MEDIUM_LIGHT)
            self.scoresLabel.grid(row=1, column=1, padx=20, pady=5, sticky="we")
            self.scoresSubFrame2 = customtkinter.CTkFrame(self.scoresFrame, fg_color=BACKGROUND_COLOR, corner_radius=8)
            self.scoresSubFrame2.grid(row=1, column=0, padx=20, pady=5, sticky="new")
            
            self.gif = GifLabel.GifLabel(self.scoresSubFrame2, "graphics/scores.gif", width=500, height=500)
            self.gif.grid(row=0, column=1, padx=10, pady=10)
            self.gif.stop()
            
            def stopGif():
                self.gif.stop()
                self.continue_button.configure(state="active")
                self.stop_button.configure(state="disabled")
            def continueGif():
                self.gif.start()
                self.stop_button.configure(state="active")
                self.continue_button.configure(state="disabled", text="Fortsetzen")
            
            self.stop_button = customtkinter.CTkButton(self.scoresSubFrame2, text="Pausieren", command=stopGif, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", state="disabled", text_color_disabled="#585858")
            self.stop_button.grid(row=1, column=0, padx=20, sticky="we")
            self.continue_button = customtkinter.CTkButton(self.scoresSubFrame2, text="Starten", command=continueGif, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", state="active", text_color_disabled="#585858")
            self.continue_button.grid(row=1, column=2, padx=20, sticky="we")
            
        
        self.eAeFrame = customtkinter.CTkFrame(self.tab2, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.eAeFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.eAe_headline_Label = customtkinter.CTkLabel(self.eAeFrame, text="Was möchtest du untersuchen?", fg_color="transparent", font=FONT_EXTRALARGE)
        self.eAe_headline_Label.grid(row=0, column=1, padx=20, pady=10)
        self.eAe_text_Label = customtkinter.CTkLabel(self.eAeFrame, text="Im Demonstrator arbeiten zwei Modelle zusammen: Das KI-Modell sagt die Qualität der\nLinsen voraus. Ein mathematisches Verfahren sucht auf dieser Grundlage nach\ngeeigneten Maschineneinstellungen. Dieses Verfahren heißt Optimierungsalgorithmus.\nHier kannst du beide Modelle gezielt untersuchen.", fg_color="transparent")
        self.eAe_text_Label.grid(row=1, column=1, padx=20, pady=10)
        self.eAeFrame1 = customtkinter.CTkFrame(self.eAeFrame, width=200, height=200, fg_color=BACKGROUND_COLOR, border_width=2,border_color="gray")
        self.eAeFrame1.grid(row=2, column=2, padx=20, pady=20, sticky="nsew")
        self.eAeFrame1_headline_Label = customtkinter.CTkLabel(self.eAeFrame1, text="Einfluss der Datenmenge verstehen", fg_color="transparent", font=FONT_EXTRALARGE)
        self.eAeFrame1_headline_Label.grid(row=0, column=0, padx=20, pady=10)
        self.eAeFrame1_text_Label = customtkinter.CTkLabel(self.eAeFrame1, text="Was passiert, wenn das KI-Modell weniger Produktionsläufe zum\nLernen hat? Wie viele Daten braucht die KI,\num zuverlässig zu sein?", fg_color="transparent")
        self.eAeFrame1_text_Label.grid(row=1, column=0, padx=20, pady=10)
        self.eAeFrame1_button = customtkinter.CTkButton(self.eAeFrame1, text="Jetzt verstehen", command=createProdLaufFrame, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858", font=FONT_MEDIUM)
        self.eAeFrame1_button.grid(row=2, column=0, padx=20)
        self.eAeFrame2 = customtkinter.CTkFrame(self.eAeFrame, width=200, height=200, fg_color=BACKGROUND_COLOR, border_width=2,border_color="gray")
        self.eAeFrame2.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.eAeFrame2_headline_Label = customtkinter.CTkLabel(self.eAeFrame2, text="Optimierungsalgorithmus entdecken", fg_color="transparent", font=FONT_EXTRALARGE)
        self.eAeFrame2_headline_Label.grid(row=0, column=0, padx=20, pady=10)
        self.eAeFrame2_text_Label = customtkinter.CTkLabel(self.eAeFrame2, text="Wie findet der Optimierungsalgorithmus gute Maschineneinstellungen?\nSchau dem mathematischen Verfahren live dabei zu, wie es aus vielen\nmöglichen Einstellungen eine passende Kombination findet.", fg_color="transparent")
        self.eAeFrame2_text_Label.grid(row=1, column=0, padx=20, pady=10)
        self.eAeFrame2_button = customtkinter.CTkButton(self.eAeFrame2, text="Jetzt anschauen", command=showBestQuality, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858", font=FONT_MEDIUM)
        self.eAeFrame2_button.grid(row=2, column=0, padx=20)
        
        # Wirksam Logo
        self.wirksam_frame4 = customtkinter.CTkFrame(self.tab2,border_width=2,border_color=BACKGROUND_COLOR, fg_color=BACKGROUND_COLOR)
        self.wirksam_frame4.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="se")
        #self.wirksam_logo4 = customtkinter.CTkImage(light_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), dark_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), size=(320, 174))
        self.wirksam_logo_image_label4 = customtkinter.CTkLabel(self.wirksam_frame4, text="", image=self.wirksam_logo)
        self.wirksam_logo_image_label4.grid(row=0, column=0, rowspan=6, padx=20, pady=10, sticky="se")
        
        # Optimierungsalgorithmus - Tab
        
        #self.optAlgoFrame = customtkinter.CTkFrame(self.tabview.tab("Optimierungsalgorithmus"), width=200, height=200) 
        
        #fig = plot_scores(self.scores)
        #canvas = FigureCanvasTkAgg(fig, self.tabview.tab("Optimierungsalgorithmus"))  # A tk.DrawingArea.
        #canvas.draw()
        #canvas.get_tk_widget().grid(row=0, column=0)  
        
        # Neuronales Netz
        
        self.nnFrame = customtkinter.CTkFrame(self.tab3, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.nnFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.activVar = customtkinter.StringVar(value="relu")
        self.solverVar = customtkinter.StringVar(value="adam")
        
        self.nnsubFrame1 = customtkinter.CTkFrame(self.nnFrame, width=200, height=200, fg_color=BACKGROUND_COLOR, border_color=BACKGROUND_COLOR, border_width=2)
        self.nnsubFrame1.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)
        
        self.nnsubFrame6 = customtkinter.CTkFrame(self.nnFrame, width=200, height=200, fg_color=BACKGROUND_COLOR, border_width=0)
        self.nnsubFrame6.grid(row=0, column=1, padx=20, pady=10, sticky="nsew", rowspan=2)
                
        self.NN_graphic = []
        self.NN_graphic.append(customtkinter.CTkImage(light_image=Image.open('graphics/neural_net_gfx/5_5.PNG'), dark_image=Image.open('graphics/neural_net_gfx/5_5.PNG'), size=(337, 600)))
        self.NN_graphic.append(customtkinter.CTkImage(light_image=Image.open('graphics/neural_net_gfx/32_32_32.PNG'), dark_image=Image.open('graphics/neural_net_gfx/32_32_32.PNG'), size=(337, 600)))
        self.NN_graphic.append(customtkinter.CTkImage(light_image=Image.open('graphics/neural_net_gfx/64_32_16.PNG'), dark_image=Image.open('graphics/neural_net_gfx/64_32_16.PNG'), size=(337, 600)))
        self.NN_graphic.append(customtkinter.CTkImage(light_image=Image.open('graphics/neural_net_gfx/256_128_64_32_16.PNG'), dark_image=Image.open('graphics/neural_net_gfx/256_128_64_32_16.PNG'), size=(337, 600)))
                
        self.NN_graphic_label = customtkinter.CTkLabel(self.nnsubFrame6, text="", image=self.NN_graphic[2])
        self.NN_graphic_label.grid(row=0, column=0, rowspan=2, padx=20, pady=10)
                
        def change_nn_graphic(choice):
                    #"64 32 16", "5 5", "32 32 32", "256 128 64 32 16"
            if(choice == "Großes Modell: 64, 32, 16"):
                self.NN_graphic_label.configure(image=self.NN_graphic[2])
            if(choice == "Kleines Modell: 5, 5"):
                self.NN_graphic_label.configure(image=self.NN_graphic[0])
            if(choice == "Mittleres Modell: 32, 32, 32"):
                self.NN_graphic_label.configure(image=self.NN_graphic[1])
            if(choice == "Sehr großes Modell: 256, 128, 64, 32, 16"):
                self.NN_graphic_label.configure(image=self.NN_graphic[3])

        self.aiLabelTitle = customtkinter.CTkLabel(self.nnsubFrame1, text="Trainiere dein eigenes KI-Modell", fg_color="transparent", font=FONT_LARGE)
        self.aiLabelTitle.grid(row=0, column=0, padx=10, pady=(10, 2), columnspan=2, sticky="w")

        self.aiLabel = customtkinter.CTkLabel(self.nnsubFrame1, text="Das Training einer KI bedeutet, dass sie aus vielen Produktionsläufen lernt, um später gute Vorhersagen zu machen.\nLege fest, wie oft das mathematische Modell beim Lernen angepasst wird. Diese Wiederholungen heißen Iterationen. Wähle außerdem den Aufbau des KI-Modells aus.", fg_color="transparent", justify="left")
        self.aiLabel.grid(row=1, column=0, padx=10, pady=(2, 10), columnspan=2, sticky="w")
        
        #self.aiLabel = customtkinter.CTkLabel(self.nnsubFrame1, text="Trainiere dein eigenes KI-Modell\nStelle ein, wie oft das KI-Modell aus vergangenen Produktionsläufen lernt (Iterationen) und wie viele Verarbeitungsschritte es\ndabei durchläuft (Schichten). Trainiere es dann und beobachte, wie gut die Qualitätsvorhersage ist.", fg_color="transparent", font=FONT_LARGE)
        #self.aiLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
        
        self.nnsubFrameLeft = customtkinter.CTkFrame(self.nnFrame, width=200, height=200, fg_color=BACKGROUND_COLOR, border_color="gray", border_width=2)
        self.nnsubFrameLeft.grid(row=1, column=0, padx=20, pady=10, sticky="nw")
        
        self.nnsubFrame2 = customtkinter.CTkFrame(self.nnsubFrameLeft, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.nnsubFrame2.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        self.iterationsLabel = customtkinter.CTkLabel(self.nnsubFrame2, text="Iterationen: Wie oft das mathematische Modell beim Lernen angepasst wird", fg_color="transparent", font=FONT_LARGE )
        self.iterationsLabel.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        def change_iterationsSlider_val(value):
            self.iterationsSliderVar = int(value)
            self.iterationsLabel2.configure(text=f"{(self.iterationsSlider.get()):.0f}")
        self.iterationsSliderVar = tk.IntVar(value=250)
        self.iterationsLabel2 = customtkinter.CTkLabel(self.nnsubFrame2, text=f"{(self.iterationsSliderVar.get()):.0f}", fg_color="transparent", font=FONT_LARGE)
        self.iterationsLabel2.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.iterationsSlider = customtkinter.CTkSlider(self.nnsubFrame2, from_=1, to=500, variable=self.iterationsSliderVar, command=change_iterationsSlider_val, number_of_steps=499, button_color=TURQUOISE, hover=False)
        self.iterationsSlider.grid(row=1, column=0, padx=20)
        
        
        self.layerLabel = customtkinter.CTkLabel(self.nnsubFrame2, text="Aufbau des KI-Modells: Versteckte Schichten und Recheneinheiten", fg_color="transparent", font=FONT_LARGE)
        self.layerLabel.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.layersVar = customtkinter.StringVar(value="Großes Modell: 64, 32, 16")
        self.layersOption = customtkinter.CTkOptionMenu(self.nnsubFrame2, values=["Kleines Modell: 5, 5", "Mittleres Modell: 32, 32, 32", "Großes Modell: 64, 32, 16", "Sehr großes Modell: 256, 128, 64, 32, 16"],variable=self.layersVar, corner_radius=12,fg_color=TURQUOISE_HELL,text_color="#1a1a1a", command=change_nn_graphic, button_color=TURQUOISE, hover=False, width=300)
        self.layersOption.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.layersOption.set("Großes Modell: 64, 32, 16")
        
        def nn_button_pressed():
            self.iterationsSliderVar = int(self.iterationsSlider.get())
            start_time = time.time()
            layers = tuple(int(x.strip()) for x in self.layersOption.get().split(":", 1)[1].split(","))
            self.new_model, mse, perc = create_neural_network(self.min_max_scaler, hidden_layers=layers, acti_func=self.activVar.get(), solve_func=self.solverVar.get(), max_iterations=self.iterationsSliderVar)
            total_time = time.time() - start_time
            total_time = f"{total_time:.1f} Sek."
            mse = f"{mse:.1f}"
            #self.mseValue.configure(text=mse)
            judge_perc(perc)
            perc = f"{perc:.1f}%"
            self.timeLabel2.configure(text="")
            self.percValue.configure(text=perc)
            #self.takeNNButton.configure(state="normal")
            self.timeValue.configure(text=total_time)
            list_built_nns(self.iterationsSliderVar, layers, perc)
        
        self.createNNButton = customtkinter.CTkButton(self.nnsubFrame2, text="KI-Modell trainieren", command=nn_button_pressed, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.createNNButton.grid(row=3, column=0, padx=20, columnspan=3, sticky="nsew")
        
        self.nnsubFrame3 = customtkinter.CTkFrame(self.nnsubFrameLeft, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.nnsubFrame3.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
        
        self.percLabel = customtkinter.CTkLabel(self.nnsubFrame3, text="Trefferquote (Qualitätsvorhersage):", fg_color="transparent", font=FONT_LARGE)
        self.percLabel.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.percValue = customtkinter.CTkLabel(self.nnsubFrame3, text="", fg_color="transparent", font=FONT_LARGE)
        self.percValue.grid(row=0, column=1, padx=20, pady=10, sticky="w")
        self.percLabel2 = customtkinter.CTkLabel(self.nnsubFrame3, text="", fg_color="transparent", font=("Arial", 16) )
        self.percLabel2.grid(row=0, column=2, padx=20, pady=10, sticky="w")
        
        self.timeLabel = customtkinter.CTkLabel(self.nnsubFrame3, text="Trainingszeit:", fg_color="transparent", font=FONT_LARGE)
        self.timeLabel.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.timeValue = customtkinter.CTkLabel(self.nnsubFrame3, text="", fg_color="transparent", font=FONT_LARGE)
        self.timeValue.grid(row=1, column=1, padx=20, pady=10, sticky="w")
        self.timeLabel2 = customtkinter.CTkLabel(self.nnsubFrame3, text="", fg_color="transparent", font=("Arial", 16) )
        self.timeLabel2.grid(row=1, column=2, padx=20, pady=10, sticky="w")
        
        #self.mseLabel = customtkinter.CTkLabel(self.nnsubFrame3, text="Mean Squared Error des Neuronalen Netzes: ", fg_color="transparent", font=FONT_LARGE)
        #self.mseLabel.grid(row=0, column=2, padx=20, pady=10)
        #self.mseValue = customtkinter.CTkLabel(self.nnsubFrame3, text="", fg_color="transparent", font=FONT_LARGE)
        #self.mseValue.grid(row=1, column=2, padx=20, pady=10)
        
        self.nnsubFrame4 = customtkinter.CTkFrame(self.nnsubFrameLeft, width=200, height=80, fg_color=BACKGROUND_COLOR) # Rückmeldung zwischen Trainingszeit und bisherigen Versuchen
        self.nnsubFrame4.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.nnsubFrame4.grid_remove()
        
        self.nnsubFrame5 = customtkinter.CTkFrame(self.nnsubFrameLeft, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.nnsubFrame5.grid(row=3, column=0, padx=20, pady=10, sticky="nw")
        
        #self.nnsubFrame5.grid_columnconfigure(0, weight=1)
        #self.nnsubFrame5.grid_columnconfigure(1, weight=1)
        #self.nnsubFrame5.grid_columnconfigure(2, weight=1)
        
        self.built_nns = 0
        self.built_nns_iterations = []
        self.built_nns_layers = []
        self.built_nns_percentages = []
        self.built_nns_caption = customtkinter.CTkLabel(self.nnsubFrame5, text="Bisherige Versuche", fg_color="transparent", font=FONT_LARGE, justify="left" )
        self.built_nns_caption.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.nnsubsubFrame1 = customtkinter.CTkFrame(self.nnsubFrame5, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.nnsubsubFrame1.grid(row=1, column=0, padx=20, pady=0, sticky="nw")
        self.built_nns_subcaption1 = customtkinter.CTkLabel(self.nnsubsubFrame1, text="Iterationen", fg_color="transparent", font=FONT_MEDIUM )
        self.built_nns_subcaption1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.built_nns_subcaption2 = customtkinter.CTkLabel(self.nnsubsubFrame1, text="Schichten", fg_color="transparent", font=FONT_MEDIUM )
        self.built_nns_subcaption2.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.built_nns_subcaption3 = customtkinter.CTkLabel(self.nnsubsubFrame1, text="Trefferquote", fg_color="transparent", font=FONT_MEDIUM )
        self.built_nns_subcaption3.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        
        
        def clear_built_nns():
            for i in range (0, self.built_nns):
                self.built_nns_iterations[i].grid_forget()
                self.built_nns_layers[i].grid_forget()
                self.built_nns_percentages[i].grid_forget()
            self.built_nns_iterations.clear()
            self.built_nns_layers.clear()
            self.built_nns_percentages.clear()
            
            self.built_nns = 0
        
        self.built_nns_redo = customtkinter.CTkButton(self.nnsubFrame5, text="Neu starten", command=clear_built_nns, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.built_nns_redo.grid(row=0, column=2, padx=20)
            
        def take_nn_button_pressed():
            self.model = self.new_model
            
        def judge_perc(perc):
            if not hasattr(self, "nn_advice_label"):
                self.nn_advice_label = customtkinter.CTkLabel(
                    self.nnsubFrame4,
                    text="",
                    corner_radius=12,
                    text_color="#1a1a1a",
                    wraplength=700,
                    justify="left",
                    anchor="w"
                )
            if(perc < 50):
                self.percLabel2.configure(text="Noch unzuverlässig")
                self.nn_advice_label.configure(
                    text="Die Vorhersagen sind noch unzuverlässig. Probiere mehr Iterationen oder einen größeren Modellaufbau aus.",
                    fg_color=RED
                )
            elif(perc < 75):
                self.percLabel2.configure(text="Teilweise zuverlässig")
                self.nn_advice_label.configure(
                    text="Es geht noch besser: Erhöhe die Anzahl der Iterationen oder wähle einen anderen Modellaufbau.",
                    fg_color=YELLOW
                )
            else:
                self.percLabel2.configure(text="Zuverlässig")
                self.nn_advice_label.configure(
                    text="Sehr gut. Das KI-Modell erreicht bereits zuverlässige Vorhersagen.",
                    fg_color=GREEN
                )
            
            self.nnsubFrame4.grid()
            self.nn_advice_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
            
            self.nnsubFrame4.grid_columnconfigure(0, weight=1)
        
        self.nnsubSubFrame = customtkinter.CTkScrollableFrame (self.nnsubFrame5, width=400, height=200, fg_color=BACKGROUND_COLOR)
        self.nnsubSubFrame.grid(row=2, column=0, padx=0, pady=0, sticky="nw", columnspan=3)
        
        def list_built_nns(iterations, layers, percentage):
            i = self.built_nns
            built_nn_iteration = customtkinter.CTkLabel(self.nnsubSubFrame, text=iterations, fg_color="transparent", font=FONT_MEDIUM_LIGHT )
            built_nn_iteration.grid(row=i+2, column=0, padx=30, pady=10, sticky="w")
            self.built_nns_iterations.append(built_nn_iteration)
            built_nn_layer = customtkinter.CTkLabel(self.nnsubSubFrame, text=layers, fg_color="transparent", font=FONT_MEDIUM_LIGHT)
            built_nn_layer.grid(row=i+2, column=1, padx=30, pady=10, sticky="w")
            self.built_nns_layers.append(built_nn_layer)
            built_nn_percentage = customtkinter.CTkLabel(self.nnsubSubFrame, text=percentage, fg_color="transparent", font=FONT_MEDIUM_LIGHT)
            built_nn_percentage.grid(row=i+2, column=2, padx=30, pady=10, sticky="w")
            self.built_nns_percentages.append(built_nn_percentage)
            
            self.built_nns = self.built_nns + 1
            

        self.nnsubFrame7 = customtkinter.CTkFrame(self.nnFrame, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.nnsubFrame7.grid(row=0, column=2, padx=20, pady=20, sticky="se", rowspan=3)
        
        # Wirksam Logo
        self.wirksam_frame3 = customtkinter.CTkFrame(self.nnsubFrame7,border_width=2,border_color=BACKGROUND_COLOR, fg_color=BACKGROUND_COLOR)
        self.wirksam_frame3.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="se")
        #self.wirksam_logo3 = customtkinter.CTkImage(light_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), dark_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), size=(320, 174))
        self.wirksam_logo_image_label3 = customtkinter.CTkLabel(self.nnsubFrame7, text="", image=self.wirksam_logo)
        self.wirksam_logo_image_label3.grid(row=0, column=0, padx=20, pady=10, sticky="se")
    
        #self.takeNNButton = customtkinter.CTkButton(self.nnsubFrame6, text="Übernehme erstelltes Neuronales Netz", command=take_nn_button_pressed, state="disabled")
        #self.takeNNButton.grid(row=2, column=0, padx=20)
        
        # KI Hinterfragen - Tab
        self.KiHFrame = customtkinter.CTkFrame(self.tab5, width=200, height=200, fg_color=BACKGROUND_COLOR)
        self.KiHFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Wirksam Logo
        self.wirksam_frame2 = customtkinter.CTkFrame(self.tab5,border_width=2,border_color=BACKGROUND_COLOR, fg_color=BACKGROUND_COLOR)
        self.wirksam_frame2.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="se")
        #self.wirksam_logo2 = customtkinter.CTkImage(light_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), dark_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), size=(320, 174))
        self.wirksam_logo_image_label2 = customtkinter.CTkLabel(self.wirksam_frame2, text="", image=self.wirksam_logo)
        self.wirksam_logo_image_label2.grid(row=0, column=0, rowspan=6, padx=20, pady=10, sticky="se")
        
        def KiHReset():
            self.KiHSubmission = 0
            self.KiHQuestionNr = 1
            self.KiHAnswersCorrect = []
            
            if hasattr(self, "KiHFrame2") and self.KiHFrame2.winfo_exists():
                self.KiHFrame2.grid_forget()
            if hasattr(self, "KiHFrame3") and self.KiHFrame3.winfo_exists():
                self.KiHFrame3.grid_forget()
            if hasattr(self, "KiHFrame4") and self.KiHFrame4.winfo_exists():
                self.KiHFrame4.grid_forget()
            
            if hasattr(self, "KiHReturnBtn") and self.KiHReturnBtn.winfo_exists():
                self.KiHReturnBtn.grid_forget()
            if hasattr(self, "KiHFrame5") and self.KiHFrame5.winfo_exists():
                self.KiHFrame5.grid_forget()
            if hasattr(self, "KiHFrame6") and self.KiHFrame6.winfo_exists():
                self.KiHFrame6.grid_forget()
            
            if hasattr(self, "KiHkreis1") and self.KiHkreis1.winfo_exists():
                self.KiHkreis1.configure(fg_color="#62717c", text_color="black")
            if hasattr(self, "KiHkreis2") and self.KiHkreis2.winfo_exists():
                self.KiHkreis2.configure(fg_color="#62717c", text_color="black")
            if hasattr(self, "KiHkreis3") and self.KiHkreis3.winfo_exists():
                self.KiHkreis3.configure(fg_color="#62717c", text_color="black")
            if hasattr(self, "KiHkreis4") and self.KiHkreis4.winfo_exists():
                self.KiHkreis4.configure(fg_color="#62717c", text_color="black")
            
            if hasattr(self, "FragenFrame") and self.FragenFrame.winfo_exists():
                self.FragenFrame.grid_forget()
            
                        
            self.KiH_headline_Label.grid(row=0, column=0, padx=20, pady=10)
            self.KiH_text_Label.grid(row=1, column=0, padx=20, pady=10)
            self.KiH_spawn_question_button.grid(row=2, column=0, padx=20)
        
        def createKIHinterfragen():
            self.KiH_headline_Label.grid_forget()
            self.KiH_text_Label.grid_forget()
            self.KiH_spawn_question_button.grid_forget()
        
            self.FragenFrame = customtkinter.CTkFrame(self.KiHFrame, fg_color=BACKGROUND_COLOR, corner_radius=8)
            self.FragenFrame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
                
            self.KiHkreis1 = customtkinter.CTkLabel(self.FragenFrame, text=str(1), width=36, height=36, corner_radius=18, fg_color=TURQUOISE, text_color="white", font=FONT_SMALL)
            self.KiHkreis1.grid(row=0, column=0, padx=(20, 8), pady=15)
            self.KiHkreisLabel1 = customtkinter.CTkLabel(self.FragenFrame, text="Frage 1", font=customtkinter.CTkFont(size=13), text_color="black")
            self.KiHkreisLabel1.grid(row=0, column=1, sticky="w")
            self.KiHlinie1 = customtkinter.CTkFrame(self.FragenFrame, height=2, fg_color="#333333", width=80)
            self.KiHlinie1.grid(row=0, column=2, padx=15, sticky="ew")
            
            self.KiHkreis2 = customtkinter.CTkLabel(self.FragenFrame, text=str(2), width=36, height=36, corner_radius=18, fg_color="#62717c", text_color="black", font=FONT_SMALL)
            self.KiHkreis2.grid(row=0, column=3, padx=(20, 8), pady=15)
            self.KiHkreisLabel2 = customtkinter.CTkLabel(self.FragenFrame, text="Frage 2", font=customtkinter.CTkFont(size=13), text_color="black")
            self.KiHkreisLabel2.grid(row=0, column=4, sticky="w")
            self.KiHlinie2 = customtkinter.CTkFrame(self.FragenFrame, height=2, fg_color="#333333", width=80)
            self.KiHlinie2.grid(row=0, column=5, padx=15, sticky="ew")

            self.KiHkreis3 = customtkinter.CTkLabel(self.FragenFrame, text=str(3), width=36, height=36, corner_radius=18, fg_color="#62717c", text_color="black", font=FONT_SMALL)
            self.KiHkreis3.grid(row=0, column=6, padx=(20, 8), pady=15)
            self.KiHkreisLabel3 = customtkinter.CTkLabel(self.FragenFrame, text="Frage 3", font=customtkinter.CTkFont(size=13), text_color="black")
            self.KiHkreisLabel3.grid(row=0, column=7, sticky="w")
            self.KiHlinie3 = customtkinter.CTkFrame(self.FragenFrame, height=2, fg_color="#333333", width=80)
            self.KiHlinie3.grid(row=0, column=8, padx=15, sticky="ew")
            
            self.KiHkreis4 = customtkinter.CTkLabel(self.FragenFrame, text=str(4), width=36, height=36, corner_radius=18, fg_color="#62717c", text_color="black", font=FONT_SMALL)
            self.KiHkreis4.grid(row=0, column=9, padx=(20, 8), pady=15)
            self.KiHkreisLabel4 = customtkinter.CTkLabel(self.FragenFrame, text="Ergebnis", font=customtkinter.CTkFont(size=13), text_color="black")
            self.KiHkreisLabel4.grid(row=0, column=10, sticky="w")
            
            self.KiHFrame2 = customtkinter.CTkFrame(self.KiHFrame, width=200, height=200, fg_color=BACKGROUND_COLOR)
            self.KiHFrame2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
            
            self.KiHFragenNrLabel = customtkinter.CTkLabel(self.KiHFrame2, text="Frage 1 von 3", font=FONT_SMALL_LIGHT)
            self.KiHFragenNrLabel.grid(row=0, column=0, sticky="w", padx=10, pady=10)
            
            self.KiHFragenLabel = customtkinter.CTkLabel(self.KiHFrame2, text="Das KI-Modell wurde für eine bestimmte Linse trainiert. Nun soll eine\nandere Linse produziert werden. Zwar werden dieselbe Maschine und\nderselbe Kunststoff verwendet, die Geometrie der Linse ist\njedoch verändert. Kann für das neue Produkt dasselbe KI-Modell genutzt werden?", font=FONT_MEDIUM, justify="left")
            self.KiHFragenLabel.grid(row=1, column=0, sticky="w", padx=10, pady=15)
            
            self.KiHFrame3 = customtkinter.CTkFrame(self.KiHFrame, width=200, height=200, fg_color=BACKGROUND_COLOR)
            self.KiHFrame3.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
            
            self.KiHSubmission = 0
            self.KiHQuestionNr = 1
            self.KiHAnswersCorrect = [] # 0 Falsch, 1 Richtig
            
            def set_question1():
                self.KiHQuestionBtn1.configure(fg_color=TURQUOISE)
                self.KiHQuestionBtn2.configure(fg_color=TURQUOISE_HELL)
                self.KiHQuestionBtn3.configure(fg_color=TURQUOISE_HELL)
                self.KiHSubmission = 1
                if(self.KiHQuestionNr == 1 & len(self.KiHAnswersCorrect) > 0):
                    self.KiHSubmitBtn.configure(state="disabled")
                elif(self.KiHQuestionNr == 2 & len(self.KiHAnswersCorrect) > 1):
                    self.KiHSubmitBtn.configure(state="disabled")
                elif(self.KiHQuestionNr == 3 & len(self.KiHAnswersCorrect) > 2):
                    self.KiHSubmitBtn.configure(state="disabled")
                else:
                    self.KiHSubmitBtn.configure(state="active")
            
            def set_question2():
                self.KiHQuestionBtn2.configure(fg_color=TURQUOISE)
                self.KiHQuestionBtn1.configure(fg_color=TURQUOISE_HELL)
                self.KiHQuestionBtn3.configure(fg_color=TURQUOISE_HELL)
                self.KiHSubmission = 2
                if(self.KiHQuestionNr == 1 & len(self.KiHAnswersCorrect) > 0):
                    self.KiHSubmitBtn.configure(state="disabled")
                elif(self.KiHQuestionNr == 2 & len(self.KiHAnswersCorrect) > 1):
                    self.KiHSubmitBtn.configure(state="disabled")
                elif(self.KiHQuestionNr == 3 & len(self.KiHAnswersCorrect) > 2):
                    self.KiHSubmitBtn.configure(state="disabled")
                else:
                    self.KiHSubmitBtn.configure(state="active")
                
            def set_question3():
                self.KiHQuestionBtn3.configure(fg_color=TURQUOISE)
                self.KiHQuestionBtn1.configure(fg_color=TURQUOISE_HELL)
                self.KiHQuestionBtn2.configure(fg_color=TURQUOISE_HELL)
                self.KiHSubmission = 3
                if(self.KiHQuestionNr == 1 & len(self.KiHAnswersCorrect) > 0):
                    self.KiHSubmitBtn.configure(state="disabled")
                elif(self.KiHQuestionNr == 2 & len(self.KiHAnswersCorrect) > 1):
                    self.KiHSubmitBtn.configure(state="disabled")
                elif(self.KiHQuestionNr == 3 & len(self.KiHAnswersCorrect) > 2):
                    self.KiHSubmitBtn.configure(state="disabled")
                else:
                    self.KiHSubmitBtn.configure(state="active")
        
            self.KiHQuestionBtn1 = customtkinter.CTkButton(self.KiHFrame3, text="Ja, das KI-Modell funktioniert auch für neue Produkte", command=set_question1, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", font=FONT_MEDIUM_LIGHT, text_color_disabled="#585858")
            self.KiHQuestionBtn1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
            self.KiHQuestionBtn2 = customtkinter.CTkButton(self.KiHFrame3, text="Nein, für neue Produkte werden neue Trainingsdaten benötigt", command=set_question2, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", font=FONT_MEDIUM_LIGHT, text_color_disabled="#585858")
            self.KiHQuestionBtn2.grid(row=1, column=0, padx=10, pady=10, sticky="w")
            self.KiHQuestionBtn3 = customtkinter.CTkButton(self.KiHFrame3, text="Das KI-Modell passt sich automatisch an", command=set_question3, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", font=FONT_MEDIUM_LIGHT, text_color_disabled="#585858")
            self.KiHQuestionBtn3.grid(row=2, column=0, padx=10, pady=10, sticky="w")
            
            self.KiHFeedbackLabel = customtkinter.CTkLabel(self.KiHFrame3, text="Platzhalter", font=FONT_MEDIUM, corner_radius=12,fg_color=TURQUOISE_HELL,text_color="#1a1a1a", justify="left")
            self.KiHFeedbackLabel.grid(row=3, column=0, sticky="w", padx=10, pady=15)
            self.KiHErklärungLabel = customtkinter.CTkLabel(self.KiHFrame3, text="Platzhalter", font=FONT_MEDIUM_LIGHT,text_color="#1a1a1a", justify="left")
            self.KiHErklärungLabel.grid(row=4, column=0, sticky="w", padx=10, pady=15)
            self.KiHFeedbackLabel.grid_forget()
            self.KiHErklärungLabel.grid_forget()
            
            def submit_answer():
                self.KiHContinueBtn.configure(state="active")
                self.KiHSubmitBtn.configure(state="disabled")
                if(self.KiHQuestionNr == 1):
                    if(self.KiHSubmission == 1):
                        self.KiHFeedbackLabel.configure(fg_color=RED, text="Nicht korrekt. Das Modell kennt nur das Produkt, mit dem es trainiert wurde.")
                        self.KiHAnswersCorrect.append(0)
                    if(self.KiHSubmission == 2):
                        self.KiHFeedbackLabel.configure(fg_color=GREEN, text="Richtig. Das Modell muss auf Daten des neuen Produkts trainiert werden.")
                        self.KiHAnswersCorrect.append(1)
                    if(self.KiHSubmission == 3):
                        self.KiHFeedbackLabel.configure(fg_color=RED, text="Nicht korrekt. Neue Produkte erfordern neue Daten und ein erneutes Training.")
                        self.KiHAnswersCorrect.append(0)
                    self.KiHErklärungLabel.configure(text="Das KI-Modell hat mit den Produktionsdaten einer bestimmten Linse gelernt. Es kennt daher nur\ndie Zusammenhänge, die bei diesem Produkt aufgetreten sind. Bei einer neuen Linse können sich\ndiese Zusammenhänge ändern, selbst wenn Material und Maschine gleich bleiben. Deshalb müssen\nzunächst Daten für die neue Linse gesammelt werden. Anschließend wird das KI-Modell angepasst\noder neu trainiert.")
                
                if(self.KiHQuestionNr == 2):
                    if(self.KiHSubmission == 1):
                        self.KiHFeedbackLabel.configure(fg_color=RED, text="Nicht korrekt. Der Prozess verändert sich auf Wegen, die das Modell nie gesehen hat.")
                        self.KiHAnswersCorrect.append(0)
                    if(self.KiHSubmission == 2):
                        self.KiHFeedbackLabel.configure(fg_color=ORANGE, text="Nah dran – aber nicht präzise genug. Das Problem ist grundlegender: Das\nModell weiß nicht einmal, dass sich etwas verändert hat, da die Umgebungstemperatur\nnicht in den Daten aufgenommen wird.")
                        self.KiHAnswersCorrect.append(0)
                    if(self.KiHSubmission == 3):
                        self.KiHFeedbackLabel.configure(fg_color=GREEN, text="Richtig. Das Modell hat Sommerbedingungen nie gesehen und\nbemerkt die Veränderung nicht – ohne Warnsignal.")
                        self.KiHAnswersCorrect.append(1)
                    self.KiHErklärungLabel.configure(text="Auch bei gleichen Maschineneinstellungen kann die Umgebungstemperatur den Prozess verändern.\nDas Granulat im Trichter erwärmt sich. Dadurch können sich beispielsweise Füllzeit, Druck und\nZykluszeit verändern. Das Modell wurde nicht mit Daten aus einer warmen Produktionshalle trainiert.\nTrotzdem gibt es eine Vorhersage aus und erkennt nicht, dass die Bedingungen neu sind.")
                
                if(self.KiHQuestionNr == 3):
                    if(self.KiHSubmission == 1):
                        self.KiHFeedbackLabel.configure(fg_color=RED, text="Nicht korrekt. Die KI kann nur die Informationen nutzen,\ndie in den verfügbaren Daten enthalten sind.")
                        self.KiHAnswersCorrect.append(0)
                    if(self.KiHSubmission == 2):
                        self.KiHFeedbackLabel.configure(fg_color=GREEN, text="Richtig. Die KI kann Zusammenhänge in den vorhandenen Daten erkennen und Vorschläge machen.\nErfahrungswissen der Beschäftigten bleibt jedoch unverzichtbar.")
                        self.KiHAnswersCorrect.append(1)
                    if(self.KiHSubmission == 3):
                        self.KiHFeedbackLabel.configure(fg_color=ORANGE, text="Zu streng. Die KI kann eine wertvolle Unterstützung sein und\nEinstellprozesse gerade für neue Mitarbeitende erleichtern.")
                        self.KiHAnswersCorrect.append(0)
                    self.KiHErklärungLabel.configure(text="Die KI lernt aus verfügbaren Prozesswerten und kann daraus\npassende Maschineneinstellungen ableiten. Es gibt jedoch Einflussfaktoren,\ndie in diesen Daten nicht enthalten sind. So können beispielsweise\ndie Feuchtigkeit oder Temperatur des Granulats abhängig vom Lagerort\nvariieren und das Prozessergebnis beeinflussen. Erfahrene\nMaschinenführer erkennen solche Situationen frühzeitig und\nbeziehen sie in ihre Entscheidungen ein.")
                    
                    
                self.KiHFeedbackLabel.grid(row=3, column=0, sticky="w", padx=10, pady=15)
                self.KiHErklärungLabel.grid(row=4, column=0, sticky="w", padx=10, pady=10)
            
            def next_question():
                self.KiHSubmitBtn.configure(state="disabled")
                self.KiHContinueBtn.configure(state="disabled")
                self.KiHFeedbackLabel.grid_forget()
                self.KiHErklärungLabel.grid_forget()
                self.KiHQuestionBtn1.configure(fg_color=TURQUOISE_HELL)
                self.KiHQuestionBtn2.configure(fg_color=TURQUOISE_HELL)
                self.KiHQuestionBtn3.configure(fg_color=TURQUOISE_HELL)
                self.KiHSubmission = 0
                if(self.KiHQuestionNr == 3):
                    self.KiHFrame2.grid_forget()
                    self.KiHFrame3.grid_forget()
                    self.KiHFrame4.grid_forget()
                    self.KiHkreis4.configure(fg_color=TURQUOISE, text_color="white")
                    
                    correct_answers = self.KiHAnswersCorrect.count(1)
                    self.KiHFrame5 = customtkinter.CTkFrame(self.KiHFrame, width=200, height=200, border_width=2,border_color="gray")
                    self.KiHFrame5.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
                    self.KiHCorrectAnswersLabel = customtkinter.CTkLabel(self.KiHFrame5, text="", font=FONT_EXTRALARGE)
                    self.KiHCorrectAnswersLabel.grid(row=0, column=0, rowspan=2, sticky="w", padx=20, pady=10)
                    self.KiHCorrectAnswersTitleLabel = customtkinter.CTkLabel(self.KiHFrame5, text="", font=FONT_LARGE_LIGHT, justify="left")
                    self.KiHCorrectAnswersTitleLabel.grid(row=0, column=1, sticky="w", padx=10, pady=10)
                    self.KiHCorrectAnswersSubtitleLabel = customtkinter.CTkLabel(self.KiHFrame5, text="", font=FONT_MEDIUM_LIGHT, justify="left")
                    self.KiHCorrectAnswersSubtitleLabel.grid(row=1, column=1, sticky="w", padx=10, pady=10)
                    if(correct_answers == 0):
                        self.KiHFrame5.configure(fg_color=RED)
                        self.KiHCorrectAnswersLabel.configure(text="0/3")
                        self.KiHCorrectAnswersTitleLabel.configure(text="Noch Luft nach oben")
                        self.KiHCorrectAnswersSubtitleLabel.configure(text="Die Fragen waren nicht einfach – schau dir die Erklärungen nochmal an. Die drei\nKernaussagen sind wichtig, um KI in der Produktion richtig einzuschätzen.")
                    elif(correct_answers == 1):
                        self.KiHFrame5.configure(fg_color=ORANGE)
                        self.KiHCorrectAnswersLabel.configure(text="1/3")
                        self.KiHCorrectAnswersTitleLabel.configure(text="Ein guter Anfang")
                        self.KiHCorrectAnswersSubtitleLabel.configure(text="Du hast erste wichtige Zusammenhänge erkannt. Bei zwei Fragen lagen die\nAntworten nah dran – lies die Erklärungen nochmal, die Nuancen machen in der Praxis den Unterschied.")
                    elif(correct_answers == 2):
                        self.KiHFrame5.configure(fg_color=YELLOW)
                        self.KiHCorrectAnswersLabel.configure(text="2/3")
                        self.KiHCorrectAnswersTitleLabel.configure(text="Schon sehr gut")
                        self.KiHCorrectAnswersSubtitleLabel.configure(text="Du hast die meisten Zusammenhänge richtig eingeschätzt.\nEine Frage war besonders knifflig.")
                    elif(correct_answers == 3):
                        self.KiHFrame5.configure(fg_color=GREEN)
                        self.KiHCorrectAnswersLabel.configure(text="3/3")
                        self.KiHCorrectAnswersTitleLabel.configure(text="Alles richtig")
                        self.KiHCorrectAnswersSubtitleLabel.configure(text="Du hast alle drei Fragen richtig beantwortet. Das zeigt: Du weißt, wo KI ihre Grenzen\nhat – und das ist genauso wichtig wie zu wissen, was sie kann.")
                    
                    self.KiHFrame6 = customtkinter.CTkFrame(self.KiHFrame, width=200, height=200, fg_color=BACKGROUND_COLOR, border_width=2,border_color="gray")
                    self.KiHFrame6.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
                    self.KiHKernaussagenTitleLabel = customtkinter.CTkLabel(self.KiHFrame6, text="Die drei Kernaussagen", font=FONT_MEDIUM)
                    self.KiHKernaussagenTitleLabel.grid(row=0, column=0, sticky="w", padx=10, pady=10)
                    self.KiHKernaussagenSubtitle1Label = customtkinter.CTkLabel(self.KiHFrame6, text="✓ Für ein neues Produkt werden passende Produktionsdaten benötigt. Das KI-Modell muss angepasst oder neu trainiert werden.", font=FONT_MEDIUM)
                    self.KiHKernaussagenSubtitle1Label.grid(row=1, column=0, sticky="w", padx=20, pady=10)
                    self.KiHKernaussagenSubtitle2Label = customtkinter.CTkLabel(self.KiHFrame6, text="✓ Veränderte Bedingungen, z. B. unterschiedliche Jahreszeiten, können das Modell unbemerkt unzuverlässig machen.", font=FONT_MEDIUM)
                    self.KiHKernaussagenSubtitle2Label.grid(row=2, column=0, sticky="w", padx=20, pady=10)
                    self.KiHKernaussagenSubtitle3Label = customtkinter.CTkLabel(self.KiHFrame6, text="✓ Was nicht gemessen wird, kann die KI nicht lernen. Menschliche Erfahrung bleibt unersetzlich.", font=FONT_MEDIUM)
                    self.KiHKernaussagenSubtitle3Label.grid(row=3, column=0, sticky="w", padx=20, pady=10)
                    
                    self.KiHReturnBtn = customtkinter.CTkButton(self.KiHFrame, text="Noch einmal starten", command=KiHReset, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
                    self.KiHReturnBtn.grid(row=3, column=0, padx=10, pady=10, sticky="we")
                    
                    
                if(self.KiHQuestionNr == 2):
                    self.KiHQuestionNr = 3
                    self.KiHkreis3.configure(fg_color=TURQUOISE, text_color="white")
                    self.KiHFragenNrLabel.configure(text="Frage 3 von 3")
                    self.KiHFragenLabel.configure(text="Die KI schlägt Maschineneinstellungen auf Basis von verfügbaren Prozesswerten vor.\nBedeutet das, dass nun jeder die Maschine auch ohne Erfahrung bedienen kann?")
                    self.KiHQuestionBtn1.configure(text="Ja, die KI übernimmt das Wissen der erfahrenen Maschinenführer vollständig")
                    self.KiHQuestionBtn2.configure(text="Teilweise. Die KI unterstützt bei der Maschineneinstellung, Erfahrung bleibt aber weiterhin unersetzlich.")
                    self.KiHQuestionBtn3.configure(text="Nein. Die KI ist für die Maschineneinstellung nicht hilfreich")
                # Umgedreht, damit nicht beide ifs nacheinander laufen (sicher auch eleganter lösbar)
                if(self.KiHQuestionNr == 1):
                    self.KiHQuestionNr = 2
                    self.KiHkreis2.configure(fg_color=TURQUOISE, text_color="white")
                    self.KiHFragenNrLabel.configure(text="Frage 2 von 3")
                    self.KiHFragenLabel.configure(text="Die Trainingsdaten wurden ausschließlich im Winter gesammelt,\nals die Halle kalt war. Im Sommer steigt die Hallentemperatur auf 35 °C.\nWie zuverlässig ist die KI-Vorhersage dann noch?")
                    self.KiHQuestionBtn1.configure(text="Genauso zuverlässig")
                    self.KiHQuestionBtn2.configure(text="Etwas weniger zuverlässig, aber die KI erkennt die Veränderung selbst")
                    self.KiHQuestionBtn3.configure(text="Unzuverlässig – das Modell hat diese Bedingungen nie gesehen")
                
            self.KiHFrame4 = customtkinter.CTkFrame(self.KiHFrame, width=200, height=200, fg_color=BACKGROUND_COLOR)
            self.KiHFrame4.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
            
            self.KiHSubmitBtn = customtkinter.CTkButton(self.KiHFrame4, text="Antwort prüfen", command=submit_answer, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", state="disabled", text_color_disabled="#585858", font=FONT_MEDIUM)
            self.KiHSubmitBtn.grid(row=3, column=1, padx=10, pady=10, sticky="ns")
            self.KiHContinueBtn = customtkinter.CTkButton(self.KiHFrame4, text="Weiter", command=next_question, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", state="disabled", text_color_disabled="#585858", font=FONT_MEDIUM)
            self.KiHContinueBtn.grid(row=4, column=1, padx=10, pady=10, sticky="ns")
        
        self.KiH_headline_Label = customtkinter.CTkLabel(self.KiHFrame, text="KI hinterfragen", fg_color="transparent", font=FONT_EXTRALARGE)
        self.KiH_headline_Label.grid(row=0, column=0, padx=20, pady=10)
        self.KiH_text_Label = customtkinter.CTkLabel(self.KiHFrame, text="Du hast die KI trainiert und getestet – aber weißt du auch, wo ihre Grenzen liegen? Es folgen drei kurze Fragen aus der Produktionspraxis.", fg_color="transparent")
        self.KiH_text_Label.grid(row=1, column=0, padx=20, pady=10)
        self.KiH_spawn_question_button = customtkinter.CTkButton(self.KiHFrame, text="Jetzt starten →", command=createKIHinterfragen, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.KiH_spawn_question_button.grid(row=2, column=0, padx=20)
        
        
        # Maschinensimulation - Tab
        # Widgets:
        self.einstellParam_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
        self.einstellParam_frame.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.einstellParam_label = customtkinter.CTkLabel(self.einstellParam_frame, text="Maschineneinstellungen", font=FONT_EXTRALARGE, anchor="w",justify="left")
        self.einstellParam_label.grid(row=0, column=0, columnspan=3, sticky="we", padx=10, pady=10)
        self.parameterLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Parameter", fg_color="transparent", font=FONT_LARGE)
        self.parameterLabel.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        parameter_tooltip_string = "Diese Werte stellst du vor der Produktion an der Maschine ein."
        self.parameter_tooltip = CTkToolTip(self.parameterLabel, message=parameter_tooltip_string)
        self.parameterAmountLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Aktueller Wert", fg_color="transparent", font=FONT_LARGE)
        self.parameterAmountLabel.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.sliderLabel = customtkinter.CTkLabel(self.einstellParam_frame, text="Regler", fg_color="transparent", font=FONT_LARGE)
        self.sliderLabel.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        slider_label_tooltip_string = "Die Regler lassen sich etwas weiter einstellen als die Werte, die in den bisherigen Produktionsdaten vorkommen. In diesen Bereichen kennt die KI die Bedingungen jedoch nur eingeschränkt."
        self.slider_label_tooltip = CTkToolTip(self.sliderLabel, message=slider_label_tooltip_string)
        self.slider1var = tk.DoubleVar(value=((155.032-81.747)/2)+81.747)
        self.slider1 = customtkinter.CTkSlider(self.einstellParam_frame, from_=81.747/1.5, to=155.032*1.5, variable=self.slider1var, command=update_label1, button_color=TURQUOISE, hover=False)
        self.slider1.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.slider1label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schmelztemperatur:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider1label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.amount1label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider1var.get()):.1f}", CELSIUS), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount1label.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.slider2var = tk.DoubleVar(value=((82.159-78.409)/2)+78.409)
        self.slider2 = customtkinter.CTkSlider(self.einstellParam_frame, from_=78.409/1.5, to=82.159*1.5, variable=self.slider2var, command=update_label2, button_color=TURQUOISE, hover=False)
        self.slider2.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.slider2label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Werkzeugtemperatur:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider2label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.amount2label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider2var.get()):.1f}", CELSIUS), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount2label.grid(row=3, column=2, padx=10, pady=5, sticky="w")
        self.slider6var = tk.DoubleVar(value=((930.6-876.7)/2)+876.7)
        self.slider6 = customtkinter.CTkSlider(self.einstellParam_frame, from_=876.7/1.5, to=930.6*1.5, variable=self.slider6var, command=update_label6, button_color=TURQUOISE, hover=False)
        self.slider6.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.slider6label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schließkraft:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider6label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.amount6label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider6var.get()):.1f}", NEWTON), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount6label.grid(row=4, column=2, padx=10, pady=5, sticky="w")
        self.slider10var = tk.DoubleVar(value=((155.5-144.8)/2)+144.8)
        self.slider10 = customtkinter.CTkSlider(self.einstellParam_frame, from_=144.8/1.5, to=155.5*1.5, variable=self.slider10var, command=update_label10, button_color=TURQUOISE, hover=False)
        self.slider10.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        self.slider10label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Gegendruck:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider10label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.amount10label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider10var.get()):.1f}", BAR), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount10label.grid(row=5, column=2, padx=10, pady=5, sticky="w")
        self.slider11var = tk.DoubleVar(value=((943-780.5)/2)+780.5)
        self.slider11 = customtkinter.CTkSlider(self.einstellParam_frame, from_=780.5/1.5, to=943*1.5, variable=self.slider11var, command=update_label11, button_color=TURQUOISE, hover=False)
        self.slider11.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        self.slider11label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Einspritzdruck:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider11label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.amount11label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider11var.get()):.1f}", BAR), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount11label.grid(row=6, column=2, padx=10, pady=5, sticky="w")
        self.slider13var = tk.DoubleVar(value=((19.23-18.51)/2)+18.51)
        self.slider13 = customtkinter.CTkSlider(self.einstellParam_frame, from_=18.51/1.5, to=19.23*1.5, variable=self.slider13var, command=update_label13, button_color=TURQUOISE, hover=False)
        self.slider13.grid(row=7, column=1, padx=10, pady=5, sticky="w")
        self.slider13label = customtkinter.CTkLabel(self.einstellParam_frame, text=f"Schussvolumen:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider13label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.amount13label = customtkinter.CTkLabel(self.einstellParam_frame, text=(f"{(self.slider13var.get()):.1f}", CM3), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount13label.grid(row=7, column=2, padx=10, pady=5, sticky="w")
        
        self.paramPopup_frame = customtkinter.CTkFrame(self.einstellParam_frame,border_width=0,border_color=BACKGROUND_COLOR, fg_color=BACKGROUND_COLOR)
        self.paramPopup_frame.grid(row=8, column=0, columnspan=3, rowspan=2, padx=10, pady=(10, 10), sticky="nsw")
        self.paramPopup_label = customtkinter.CTkLabel(self.paramPopup_frame, text="", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.paramPopup_label.grid(row=7, column=2, padx=10, pady=10, sticky="w")
        
        self.generated_paremeters = self.get_kn_vals()
        
        self.prozessParam_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
        self.prozessParam_frame.grid(row=1, column=1, padx=10, pady=(10, 0), sticky="nsew")
        self.prozessParam_label = customtkinter.CTkLabel(self.prozessParam_frame, text="Prozessparameter (im Prozess gemessene Werte)", font=FONT_EXTRALARGE)
        self.prozessParam_label.grid(row=0, column=0, columnspan=2, sticky="we", padx=10, pady=10)
        self.parameter2Label = customtkinter.CTkLabel(self.prozessParam_frame, text="Parameter", fg_color="transparent", font=FONT_LARGE)
        self.parameter2Label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.dummy_label = customtkinter.CTkLabel(self.prozessParam_frame, text="", font=FONT_EXTRALARGE)
        self.dummy_label.grid(row=1, column=1, sticky="we", padx=10, pady=10)
        self.value2Label = customtkinter.CTkLabel(self.prozessParam_frame, text="Wert", fg_color="transparent", font=FONT_LARGE)
        self.value2Label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.slider3var = tk.DoubleVar(value=self.generated_paremeters[0][6])
        self.slider3label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Füllzeit:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider3label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.amount3label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider3var.get()):.1f}", SECONDS), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount3label.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.slider4var = tk.DoubleVar(value=self.generated_paremeters[0][7])
        self.slider4label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Plastizierzeit:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider4label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.amount4label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider4var.get()):.1f}", SECONDS), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount4label.grid(row=3, column=2, padx=10, pady=5, sticky="w")
        self.slider5var = tk.DoubleVar(value=self.generated_paremeters[0][8])
        self.slider5label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Zykluszeit:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider5label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.amount5label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider5var.get()):.1f}", SECONDS), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount5label.grid(row=4, column=2, padx=10, pady=5, sticky="w")
        self.slider7var = tk.DoubleVar(value=self.generated_paremeters[0][9])
        self.slider7label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Maximale Schließkraft:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider7label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.amount7label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider7var.get()):.1f}", NEWTON), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount7label.grid(row=5, column=2, padx=10, pady=5, sticky="w")
        self.slider8var = tk.DoubleVar(value=self.generated_paremeters[0][10])
        self.slider8label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Maximales Schneckendrehmoment:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider8label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.amount8label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider8var.get()):.1f}", NEWTONMETER), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount8label.grid(row=6, column=2, padx=10, pady=5, sticky="w")
        self.slider9var = tk.DoubleVar(value=self.generated_paremeters[0][11])
        self.slider9label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Mittleres Schneckendrehmoment:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider9label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.amount9label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider9var.get()):.1f}", NEWTONMETER), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount9label.grid(row=7, column=2, padx=10, pady=5, sticky="w")
        self.slider12var = tk.DoubleVar(value=self.generated_paremeters[0][12])
        self.slider12label = customtkinter.CTkLabel(self.prozessParam_frame, text=f"Finale Schneckenposition:", fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.slider12label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.amount12label = customtkinter.CTkLabel(self.prozessParam_frame, text=(f"{(self.slider12var.get()):.1f}", CM), fg_color="transparent", font=FONT_SMALL_LIGHT)
        self.amount12label.grid(row=8, column=2, padx=10, pady=5, sticky="w")
        
        #Quality Widgets
        def use_shap_button():
            shap_explainer(self.model)
        
        self.qual_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
        self.qual_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.qual_label = customtkinter.CTkLabel(self.qual_frame, text="Qualitätsergebnis:", fg_color="transparent", font=FONT_MEDIUM)
        self.qual_label.grid(row=0, column=0, padx=10, pady=10)
        self.border_frame = customtkinter.CTkFrame(self.qual_frame, fg_color=BACKGROUND_COLOR)
        self.border_frame.grid(row=1, column=0, padx=10, pady=10)
        
        #SHAP vorerst entfernt
        #self.shap_button = customtkinter.CTkButton(self.qual_frame, text="i", width=40, height=40, corner_radius=20, fg_color="#d0d0d0", hover_color="#989898", text_color="black", font=customtkinter.CTkFont(size=16, weight="bold"), border_width=3, border_color="black", command=use_shap_button)
        #self.shap_button.grid(row=1, column=1, padx=10, pady=10)
        
        self.quality_category_label = customtkinter.CTkLabel(self.border_frame, text="", fg_color="transparent")
        self.quality_category_label.grid(row=0, column=0, padx=3, pady=3)
        self.qual_value_label = customtkinter.CTkLabel(self.qual_frame, text="U\u2080-Wert:", fg_color="transparent", font=FONT_MEDIUM)
        self.qual_value_label.grid(row=0, column=2, padx=10, pady=10)
        self.produce_label = customtkinter.CTkLabel(self.qual_frame, text="", fg_color="transparent", font=FONT_LARGE)
        self.produce_label.grid(row=1, column=2, padx=10, pady=10)
        
        segmente = [
            ("Ausschuss\nU₀ < 0,4",           RED, 100),
            ("Akzeptabel\n0,4 ≤ U₀ < 0,45",  YELLOW, 100),
            ("Sollbereich\n0,45 ≤ U₀ ≤ 0,5", GREEN, 100),
            ("Ineffizient\nU₀ > 0,5",         ORANGE, 100),
        ]

        self.qualitySkala_label = customtkinter.CTkLabel(self.qual_frame, text="Qualitätsskala:", font=FONT_MEDIUM)
        self.qualitySkala_label.grid(row=0, column=3, columnspan=len(segmente), sticky="we", padx=10, pady=10)

        for col, (text, farbe, breite) in enumerate(segmente):
            seg = customtkinter.CTkFrame(self.qual_frame, fg_color=farbe, corner_radius=4, width=breite, height=50)
            seg.grid(row=1, column=col+3, padx=5, pady=(0, 10), sticky="nsew")
            seg.grid_propagate(False)
            customtkinter.CTkLabel(seg, text=text, font=customtkinter.CTkFont(size=12), text_color="black",justify="center").grid(row=0, column=0, sticky="nsew", padx=5, pady=4)
            seg.grid_rowconfigure(0, weight=1)
            seg.grid_columnconfigure(0, weight=1)
        
        
        # Production Widgets
        self.production_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
        self.production_frame.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nsw")
        self.production_frame.grid_rowconfigure((0), weight=1)
        self.producing_button = customtkinter.CTkButton(self.production_frame, text="1.\nProduktion starten", command=self.set_kn_vals, corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.producing_button.grid(row=0, column=1, padx=10, pady=10, sticky="ns")
        self.produce_button = customtkinter.CTkButton(self.production_frame, text="2.\nQualität bestimmen", command=self.produce_func, state="disabled", corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.produce_button.grid(row=0, column=2, padx=10, pady=10, sticky="ns")
        self.algoOptionVar = customtkinter.StringVar(value="Genetischer Algorithmus")
        #self.algoOption = customtkinter.CTkOptionMenu(self.production_frame, values=["Partikelschwarmoptimierung", "Genetischer Algorithmus", "Simulierte Abkühlung"],variable=self.algoOptionVar, corner_radius=12,fg_color=TURQUOISE_HELL,text_color="#1a1a1a")
        #self.algoOption.set("Partikelschwarmoptimierung")
        #self.algoOption.grid(row=0, column=3, padx=10, pady=10, sticky="ns")
        self.ai_button = customtkinter.CTkButton(self.production_frame, text="3.\nEinstellempfehlungen berechnen", command=self.use_algo, state="disabled", corner_radius=12,border_width=2,border_color="#1a1a1a",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.ai_button.grid(row=0, column=3, padx=10, pady=10, sticky="ns")
        ai_tooltip_string = "Das mathematische Verfahren probiert verschiedene Einstellungskombinationen aus. Deshalb kann es bei jedem Durchlauf etwas andere Empfehlungen geben."
        self.ai_tooltip = CTkToolTip(self.ai_button, message=ai_tooltip_string)
        
        # Linse fertigung:
        self.lense_frame = customtkinter.CTkFrame(self.tab1,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
        self.lense_frame.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #self.lense_frame.grid_forget()
        self.lens_pic = customtkinter.CTkImage(light_image=Image.open('graphics/lens.png'), dark_image=Image.open('graphics/lens.png'), size=(150,150))
        self.lens_image_label = customtkinter.CTkLabel(self.lense_frame, text="", image=self.lens_pic)
        self.lens_image_label.grid(row=0, column=0, rowspan=6, padx=20, pady=10)
        self.lens_produced_label1 = customtkinter.CTkLabel(self.lense_frame, text="Status:", fg_color="transparent", font=FONT_MEDIUM )
        self.lens_produced_label1.grid(row=0, column=1, padx=10, pady=0, sticky="w")
        self.lens_produced_label2 = customtkinter.CTkLabel(self.lense_frame, text="Noch nicht produziert", fg_color="transparent",font=FONT_MEDIUM)
        self.lens_produced_label2.grid(row=0, column=2, padx=10, pady=0, sticky="w")
        self.article_produced_label1 = customtkinter.CTkLabel(self.lense_frame, text="Artikelnummer:", fg_color="transparent",font=FONT_SMALL)
        self.article_produced_label1.grid(row=1, column=1, padx=10, pady=0, sticky="w")
        self.article_produced_label2 = customtkinter.CTkLabel(self.lense_frame, text="-", fg_color="transparent")
        self.article_produced_label2.grid(row=1, column=2, padx=10, pady=0, sticky="w")
        self.material_label1 = customtkinter.CTkLabel(self.lense_frame, text="Material:", fg_color="transparent", font=FONT_SMALL)
        self.material_label1.grid(row=2, column=1, padx=10, pady=0, sticky="w")
        self.material_label2 = customtkinter.CTkLabel(self.lense_frame, text="-", fg_color="transparent")
        self.material_label2.grid(row=2, column=2, padx=10, pady=0, sticky="w")
        self.current_iter = "000"
        self.current_charge = f"{DATE}-M1-{self.current_iter}"
        self.charge_label1 = customtkinter.CTkLabel(self.lense_frame, text="Charge: ", fg_color="transparent", font=FONT_SMALL)
        self.charge_label1.grid(row=3, column=1, padx=10, pady=0, sticky="w")
        self.charge_label2 = customtkinter.CTkLabel(self.lense_frame, text="-", fg_color="transparent")
        self.charge_label2.grid(row=3, column=2, padx=10, pady=0, sticky="w")
        
        # Wirksam Logo
        self.wirksam_frame1 = customtkinter.CTkFrame(self.tab1,border_width=2,border_color=BACKGROUND_COLOR, fg_color=BACKGROUND_COLOR)
        self.wirksam_frame1.grid(row=2, column=1, padx=10, pady=(10, 0), sticky="e")
        #self.wirksam_logo1 = customtkinter.CTkImage(light_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), dark_image=Image.open('graphics/logos/WIRKsam-Logo-Wortmarke-RGB_farbe.jpg'), size=(320, 174))
        self.wirksam_logo_image_label1 = customtkinter.CTkLabel(self.wirksam_frame1, text="", image=self.wirksam_logo)
        self.wirksam_logo_image_label1.grid(row=0, column=0, rowspan=6, padx=20, pady=10, sticky="e")
        
        # Funktionsweise verstehen
        width = 1120#self.winfo_screenwidth()#- 173 # 70x16
        height = 630#self.winfo_screenheight()#-136 # 70x9

        self.presentationList = [] # Alle Folien automatisch und numerisch sortiert laden
        slides_dir = Path("graphics/slides")
        slide_files = sorted(
            (path for path in slides_dir.iterdir()
             if path.is_file()
             and path.suffix.lower() == ".png"
             and path.stem.lower().startswith("folie")),
            key=lambda path: int(path.stem[5:]) if path.stem[5:].isdigit() else float("inf")
        )
        if not slide_files:
            raise FileNotFoundError(
                "Keine Folien im Ordner graphics/slides gefunden. Erwartet werden Dateien wie Folie1.PNG."
            )
        for slide_path in slide_files:
            slide_image = Image.open(slide_path)
            self.presentationList.append(
                customtkinter.CTkImage(
                    light_image=slide_image,
                    dark_image=slide_image,
                    size=(width, height)
                )
            )

        self.FunkVFrame = customtkinter.CTkFrame(self.tab6,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR, width=width, height=height)
        self.FunkVFrame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nswe")
        
        self.FunkVFrame1 = customtkinter.CTkFrame(self.FunkVFrame,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
        self.FunkVFrame1.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nswe")
        self.slideCounter = 0
        self.slideMax = len(self.presentationList) - 1
        self.slide_label = customtkinter.CTkLabel(self.FunkVFrame1, text="", image=self.presentationList[0])
        self.slide_label.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        
        self.FunkVFrame2 = customtkinter.CTkFrame(self.FunkVFrame,border_width=2,border_color="gray", fg_color=BACKGROUND_COLOR)
        self.FunkVFrame2.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nswe")
        
        def updateSlideNavigation():
            self.slide_label.configure(image=self.presentationList[self.slideCounter])
            self.slideStatusLabel.configure(
                text=f"Folie {self.slideCounter + 1} von {len(self.presentationList)}"
            )
            self.FunkVReturnBtn.configure(
                state="disabled" if self.slideCounter == 0 else "normal"
            )
            self.FunkVContinueBtn.configure(
                state="disabled" if self.slideCounter == self.slideMax else "normal"
            )

        def prevSlide():
            if self.slideCounter > 0:
                self.slideCounter -= 1
                updateSlideNavigation()

        def nextSlide():
            if self.slideCounter < self.slideMax:
                self.slideCounter += 1
                updateSlideNavigation()

        self.FunkVReturnBtn = customtkinter.CTkButton(
            self.FunkVFrame2, text="Zurück", corner_radius=12, border_width=2,
            border_color="#1a1a1a", fg_color=TURQUOISE_HELL, hover_color=TURQUOISE,
            text_color="#1a1a1a", command=prevSlide, state="disabled",
            text_color_disabled="#585858"
        )
        self.FunkVReturnBtn.grid(row=0, column=0, padx=20, pady=10, sticky="nswe")
        self.slideStatusLabel = customtkinter.CTkLabel(
            self.FunkVFrame2,
            text=f"Folie 1 von {len(self.presentationList)}",
            font=FONT_MEDIUM
        )
        self.slideStatusLabel.grid(row=0, column=1, padx=20, pady=10, sticky="nswe")
        self.FunkVContinueBtn = customtkinter.CTkButton(
            self.FunkVFrame2, text="Weiter", corner_radius=12, border_width=2,
            border_color="#1a1a1a", fg_color=TURQUOISE_HELL, hover_color=TURQUOISE,
            text_color="#1a1a1a", command=nextSlide, state="normal",
            text_color_disabled="#585858"
        )
        self.FunkVContinueBtn.grid(row=0, column=2, padx=20, pady=10, sticky="nswe")
        updateSlideNavigation()
        
        
        def resetEverything():
            #KI Live testen:
            self.current_iter = "000"
            self.slider1var = tk.DoubleVar(value=((155.032-81.747)/2)+81.747)
            self.slider1.configure(variable=self.slider1var)
            self.slider2var = tk.DoubleVar(value=((82.159-78.409)/2)+78.409)
            self.slider2.configure(variable=self.slider2var)
            self.slider6var = tk.DoubleVar(value=((930.6-876.7)/2)+876.7)
            self.slider6.configure(variable=self.slider6var)
            self.slider10var = tk.DoubleVar(value=((155.5-144.8)/2)+144.8)
            self.slider10.configure(variable=self.slider10var)
            self.slider11var = tk.DoubleVar(value=((943-780.5)/2)+780.5)
            self.slider11.configure(variable=self.slider11var)
            self.slider13var = tk.DoubleVar(value=((19.23-18.51)/2)+18.51)
            self.slider13.configure(variable=self.slider13var)
            self.slider_change()
            self.removeOptimized()
            self.lens_produced_label2.configure(text="Unproduziert")
            self.article_produced_label2.configure(text="-")
            self.material_label2.configure(text="-")
            self.charge_label2.configure(text="-")
            #KI Hinterfragen:
            KiHReset()
            #Einflussfaktoren untersuchen:
            if hasattr(self, "prodLaufFrame") and self.prodLaufFrame.winfo_exists():
                self.prodLaufFrame.destroy()
            if hasattr(self, "scoresFrame") and self.scoresFrame.winfo_exists():
                self.scoresFrame.destroy()
            self.eAeFrame.grid()
            #KI selbst trainieren:
            #clear_built_nns()
            #Funktionsweise verstehen:
            self.slideCounter = 0
            self.slide_label.configure(image=self.presentationList[self.slideCounter])
        self.tabview.configure(command=resetEverything)
        #Mögliche optimierung:
        self.update_idletasks()
        self.deiconify()
        
    def slider_change(self):
        self.produce_button.configure(state="disabled")
        self.quality_category_label.configure(text="")
        self.quality_category_label.configure(fg_color="transparent")
        self.produce_label.configure(text="")
        self.paramPopup_label.configure(text="")
        
        self.border_frame.configure(fg_color=BACKGROUND_COLOR, border_width=0,border_color=BACKGROUND_COLOR)
        
    def update_charge_init(self):
        self.lens_produced_label2.configure(text="Linse wird produziert")
        self.article_produced_label2.configure(text="-")
        self.material_label2.configure(text="-")
        self.charge_label2.configure(text="-")
        #Ändere Bild kurzzeitig zu Ladebild:
        
        self.after(2000, self.update_charge_finish)
        
    def update_charge_finish(self):
        
        
        self.current_iter = str(int(self.current_iter) + 1).zfill(len(self.current_iter))
        self.current_charge = f"{DATE}-M1-{self.current_iter}"
        self.charge_label2.configure(text=self.current_charge)
        self.lens_produced_label2.configure(text="Linse produziert")
        self.article_produced_label2.configure(text="LNS-4712")
        self.material_label2.configure(text="PMMA")
        
        self.produce_button.configure(state="normal")
        
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
        #self.lense_frame.grid()
        #self.lense_frame.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nsw")
        
        self.update_charge_init()
        self.slider_change()
        
        
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
            self.border_frame.configure(fg_color=RED, border_width=2,border_color="black")
            self.paramPopup_label.configure(text="Schade, diese Einstellung führt zu Ausschuss.\nProbiere eine andere Kombination aus!")
        if(quality_cat == "Akzeptabel"):
            self.border_frame.configure(fg_color=YELLOW, border_width=2,border_color="black")
            self.paramPopup_label.configure(text="Knapp daneben! Die Linse ist verkäuflich, jedoch nur als B-Ware.\nKannst du die Qualität weiter verbessern?")
        if(quality_cat == "Sollbereich"):
            self.border_frame.configure(fg_color=GREEN, border_width=2,border_color="black")
            self.paramPopup_label.configure(text="Perfekt! Mit dieser Einstellung erhältst du eine\n Linse in der gewünschten Qualität. Gut gemacht!")
        if(quality_cat == "Ineffizient"):
            self.border_frame.configure(fg_color=ORANGE, border_width=2,border_color="black")
            self.paramPopup_label.configure(text="Interessant! Die Qualität ist sogar besser als erforderlich. Die Herausforderung ist,\n die Qualitätsanforderungen zu erfüllen und gleichzeitig\n wirtschaftlich zu produzieren. Versuche es noch einmal.")
        produce_tooltip_string = "Ein KI-Modell schätzt die Qualität anhand der eingestellten Maschinenwerte. Es hat diese Zusammenhänge zuvor aus Produktionsdaten gelernt."
        self.produce_tooltip = CTkToolTip(self.produce_label, message=produce_tooltip_string)
        self.ai_button.configure(state="normal")
    
    def use_algo(self):
        if(self.algoOptionVar=="Genetischer Algorithmus"):
            solution_std, fitness, self.scores, iterations = ga.ga(self.model, get_X(self.min_max_scaler), self.kn, 5.0, 30, 200, 0.2, "tournament", "blend")
        elif(self.algoOptionVar=="Simulierte Abkühlung"):
            solution_std, fitness, self.scores, iterations = sa.simulated_annealing(self.model, get_X(self.min_max_scaler), self.kn, 5.0, 100, 200, 0.1, False, "exponential")
        else:
            solution_std, fitness, self.scores, it = pso.pso(self.model, get_X(self.min_max_scaler), self.kn, 5.0, pop_size=30, iterations=200, w=0.6, c1=1, c2=2)

        #self.lense_frame.grid_forget()
        
        solution = self.scaler2.inverse_transform(solution_std)
        self.transformed_solution = solution.squeeze()
        self.transformed_solution = self.transformed_solution[0:6]
        self.Optparamterlabel0 = customtkinter.CTkLabel(self.einstellParam_frame, text="KI-Empfehlung", fg_color="transparent", font=FONT_LARGE)
        self.Optparamterlabel0.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.optParameterLabels = []
        for i, value in enumerate(self.transformed_solution):
            optParameterLabel = customtkinter.CTkLabel(self.einstellParam_frame, text=f"{value:.1f}", fg_color="transparent", font=FONT_SMALL_LIGHT)
            optParameterLabel.grid(row=i + 2, column=3, padx=10, pady=5, sticky="w")
            self.optParameterLabels.append(optParameterLabel)
        self.useOptimizedButton = customtkinter.CTkButton(self.einstellParam_frame, text="4.\nVorschlag übernehmen", command=self.useOptimizedFunc, corner_radius=12,border_width=1,border_color="#a0b4c8",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.useOptimizedButton.grid(row=8, column=3, padx=10, pady=5)
        self.removeOptimizedButton = customtkinter.CTkButton(self.einstellParam_frame, text="Vorschlag zurücksetzen", command=self.removeOptimized, corner_radius=12,border_width=1,border_color="#a0b4c8",fg_color=TURQUOISE_HELL,hover_color=TURQUOISE,text_color="#1a1a1a", text_color_disabled="#585858")
        self.removeOptimizedButton.grid(row=9, column=3, padx=10, pady=5)
        
    def removeOptimized(self):
        if hasattr(self, "Optparamterlabel0") and self.Optparamterlabel0.winfo_exists():
            self.Optparamterlabel0.destroy()
        for i in range(0, len(self.optParameterLabels)):
            self.optParameterLabels[i].destroy()
        self.useOptimizedButton.destroy()
        self.removeOptimizedButton.destroy()
        
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
