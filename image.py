import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
model = tf.keras.applications.MobileNetV2(weights="imagenet")
def classify_image():
    try:
        upload_button.config(state="disabled")
        result_label.config(text="Processing...", fg="blue")
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
        if not image_path:
            upload_button.config(state="normal")
            result_label.config(text="No image selected", fg="red")
            return
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))  
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        class_label = decoded_predictions[0][1]
        class_confidence = decoded_predictions[0][2] * 100  # Convert to percentage
        result_label.config(text=f"Prediction: {class_label}\nConfidence: {class_confidence:.2f}%", fg="green")
        upload_button.config(state="normal")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        upload_button.config(state="normal")
window = tk.Tk()
window.title("Image Classifier")
window.geometry("800x600") 
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack(fill="both", expand=True)
background_image = Image.open("background.webp")  
background_image = background_image.resize((800, 600), Image.Resampling.LANCZOS)  
background_tk = ImageTk.PhotoImage(background_image)
canvas.create_image(0, 0, anchor="nw", image=background_tk)
title_label = tk.Label(window, text="Image Classifier", font=("Arial", 24, "bold"), fg="white", bg="#4CAF50")
title_label.place(relx=0.5, rely=0.05, anchor="center")  
image_label = tk.Label(window, bg="#f5f5f5")
image_label.place(relx=0.5, rely=0.3, anchor="center")  
def on_enter(event):
    upload_button.config(bg="#45a049")  
def on_leave(event):
    upload_button.config(bg="#4CAF50")  

upload_button = tk.Button(window, text="Upload Image", font=("Arial", 14), bg="#4CAF50", fg="white", command=classify_image)
upload_button.place(relx=0.5, rely=0.55, anchor="center")
upload_button.bind("<Enter>", on_enter)
upload_button.bind("<Leave>", on_leave)
result_label = tk.Label(window, text="Prediction: N/A\nConfidence: 0%", font=("Arial", 14), fg="black", bg="#f5f5f5")
result_label.place(relx=0.5, rely=0.75, anchor="center")  
window.mainloop()