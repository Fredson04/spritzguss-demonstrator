import customtkinter as ctk
from PIL import Image

class GifLabel(ctk.CTkLabel):
    def __init__(self, parent, gif_path, width=100, height=100, **kwargs):
        super().__init__(parent, text="", **kwargs)
        self.frames = []
        self.current_frame = 0
        
        # Alle Frames aus GIF laden
        gif = Image.open(gif_path)
        try:
            while True:
                frame = ctk.CTkImage(gif.copy().convert("RGBA"), size=(width, height))
                self.frames.append(frame)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        
        self.running = True
        self.animate()

    def animate(self):
        if self.running and self.frames:
            self.configure(image=self.frames[self.current_frame])
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.after(150, self.animate)

    def stop(self):
        self.running = False
        

    def start(self):
        self.running = True
        self.animate()
       