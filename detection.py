import tensorflow as tf
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import smtplib

root = tkinter.Tk()
root.withdraw()

model = tf.keras.models.load_model('Face_mask_detection.h5')
face_det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid_source = cv2.VideoCapture(0)

text_dict = {0: 'NO MASK', 1: ' MASK'}
rect_color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}



Subject = "Subject"
TEXT = "A Person has been detected without mask"

while (True):
    ret, img = vid_source.read()
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_det.detectMultiScale(grayscale_img, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = grayscale_img[y:y + h, x:x + w]
        resized_img = cv2.resize(face_img, (112, 112))
        normalized_img = resized_img / 255.0
        reshaped_img = np.reshape(resized_img, ( 1, 112, 112, 1))
        result = model.predict(reshaped_img)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), rect_color_dict[label], -1)
        cv2.putText(img, text_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        if (label == 0):
            # messagebox.showwarning("Warning","Access Denied.Please wear a mask")
            print("no mask")
            
            

            # message = 'Subject : {}\n\n{}'. format(Subject, TEXT)
            # mail = smtplib.SMTP('smtp.gmail.com', 587)
            # mail.ehlo()
            # mail.starttls()
            # mail.login('shivamsinghal24@gmail.com', '') # add pass
            # mail.sendmail('shivamsinghal24@gmail.com','shivamsinghal24@gmail.com', message)
            # mail.close

        else:
            pass
            break


    cv2.imshow('LIVE VIDEO FEED', img)
    key = cv2.waitKey(1)
    if (key == 27 or key == ord('q')):
        break

vid_source.release()
cv2.destroyAllWindows()
