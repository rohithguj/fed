import cv2
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors=5, minSize=(30,30))

    model = load_model(' .keras')

    face_roi = gray_img
    for (x, y, w, h) in faces:
        face_roi = gray_img[y:y+h, x:x+w]

    resized_face_img = cv2.resize(face_roi, (48,48))

    model_input_img = resized_face_img.reshape((1,48,48,1))
    model_input_img = model_input_img / 255.0

    prediction = model.predict(model_input_img)

    print(prediction)