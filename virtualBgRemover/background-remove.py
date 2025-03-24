import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# GUI setup
def choose_background():
    global bg_image
    path = filedialog.askopenfilename()
    if path:
        bg_image = cv2.imread(path)
        bg_image = cv2.resize(bg_image, (640, 480))

# Tkinter window
gui = tk.Tk()
gui.title("Virtual Background Remover")
select_button = tk.Button(gui, text="Select Background Image", command=choose_background)
select_button.pack()

def start_remover():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Get segmentation mask
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb)
        mask = results.segmentation_mask
        condition = mask > 0.6

        # If no background is selected, just blur it
        if bg_image is None:
            blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)
            output = np.where(condition[..., None], frame, blurred_frame)
        else:
            output = np.where(condition[..., None], frame, bg_image)

        cv2.imshow('Virtual Background Remover', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

start_button = tk.Button(gui, text="Start Camera", command=start_remover)
start_button.pack()

gui.mainloop()
