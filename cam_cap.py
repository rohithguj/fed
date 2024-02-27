import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emo_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

cap = cv2.VideoCapture(0)

model = load_model('fed_50epoch.h5')

while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    face_roi = gray_img
    for (x, y, w, h) in faces:
        face_roi = gray_img[y:y + h, x:x + w]

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    resized_face_img = cv2.resize(face_roi, (48, 48))

    model_input_img = resized_face_img.reshape((1, 48, 48, 1))
    model_input_img = model_input_img / 255.0

    prediction = model.predict(model_input_img)

    emo_idx = np.argmax(prediction)

    emotion = emo_list[int(emo_idx)]

    cv2.putText(img, emotion, (30, 30 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Emotion Detection', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    print(emotion)