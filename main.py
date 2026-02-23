import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

model_path = "A:/bloodclot/stroke_model.h5"
model = load_model(model_path)
img_size = (224, 224)

def predict_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Stroke Detected" if prediction > 0.5 else "No Stroke"

def grad_cam(image_path, model, layer_name='block5_conv3'):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)[0]
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        result = predict_image(file_path)
        label_result.config(text=result)
        grad_cam(file_path, model)

root = tk.Tk()
root.title("Blood Clot Diagnosis")
root.geometry("400x300")

btn_upload = tk.Button(root, text="Upload CT Scan", command=open_image)
btn_upload.pack(pady=20)
label_result = tk.Label(root, text="", font=("Arial", 14))
label_result.pack()

root.mainloop()