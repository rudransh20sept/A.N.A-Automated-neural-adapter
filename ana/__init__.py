from .train import LoRATrainerApp
import tkinter as tk

def main():
    root = tk.Tk()
    app = LoRATrainerApp(root)
    root.mainloop()
