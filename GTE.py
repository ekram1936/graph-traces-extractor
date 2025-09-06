import os
import platform
import subprocess
import logging
import matplotlib.pyplot as plt
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from graph_processor import process_entire_image_folder

ctk.set_appearance_mode("System") 
ctk.set_default_color_theme("blue")


class TkinterLogHandler(logging.Handler):
    def __init__(self, textbox_widget):
        super().__init__()
        self.textbox = textbox_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.textbox.configure(state="normal")
            self.textbox.insert("end", msg + "\n")
            self.textbox.see("end")
            self.textbox.configure(state="disabled")

        self.textbox.after(0, append)


class GraphTraceExtractorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Graph Trace Extractor")
        self.geometry("1000x750")
        self.minsize(800, 600)

        self.label = ctk.CTkLabel(
            self, text="Select a Graph folder to extract traces", font=("Segoe UI", 18))
        self.label.pack(pady=20)

        self.select_button = ctk.CTkButton(
            self, text="📂 Select Folder", command=self.select_folder)
        self.select_button.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            self, text="", text_color="green", font=("Segoe UI", 14))
        self.status_label.pack(pady=10)

        self.scrollable_frame = ctk.CTkScrollableFrame(
            self, width=900, height=400)
        self.scrollable_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.log_box = ctk.CTkTextbox(self, height=150)
        self.log_box.pack(padx=10, pady=(0, 10), fill="x")
        self.log_box.configure(state="disabled")

        log_handler = TkinterLogHandler(self.log_box)
        log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(log_handler)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        try:
            self.status_label.configure(text="Processing...")
            self.update()

            process_entire_image_folder(folder_path)

            self.status_label.configure(text="✅ Processing complete.")
            self.show_plots(folder_path)
            self.open_folder(folder_path)

        except Exception as error:
            messagebox.showerror("Error", f"Something went wrong:\n{str(error)}")

    def show_plots(self, folder_path):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        subfolders = [f for f in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, f))]
        for sub in sorted(subfolders):
            sub_path = os.path.join(folder_path, sub)
            for file in os.listdir(sub_path):
                if file.endswith("_trace.png"):
                    image_path = os.path.join(sub_path, file)
                    image = Image.open(image_path)
                    image = image.resize((800, 400), Image.Resampling.LANCZOS)
                    image_tk = ImageTk.PhotoImage(image)
                    label = ctk.CTkLabel(
                        self.scrollable_frame, image=image_tk, text="")
                    label.image = image_tk
                    label.pack(pady=10)

    def open_folder(self, path):
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", path])
            else:  
                subprocess.call(["xdg-open", path])
        except Exception as e:
            print(f"Could not open folder: {e}")


if __name__ == "__main__":
    app = GraphTraceExtractorApp()
    app.mainloop()