# ui_vehicle_counter.py

import tkinter as tk
from tkinter import filedialog, messagebox
import vehicle_counter  # Your script from above (must be in same folder)

def browse_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        vehicle_count, result = vehicle_counter.count_vehicles(path)
        messagebox.showinfo("Result", f"Vehicles Detected: {vehicle_count}")
        result.show()

# Set up basic window
app = tk.Tk()
app.title("Hackathon Vehicle Counter")
app.geometry("400x200")

# Add button
button = tk.Button(app, text="Select Image", command=browse_image, font=("Arial", 14))
button.pack(expand=True)

# Run the UI
app.mainloop()
