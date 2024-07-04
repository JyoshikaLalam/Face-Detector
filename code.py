import cv2
from google.colab.patches import cv2_imshow
# Load the cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Function to detect faces and eyes
def detect_faces_and_eyes(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.3, 5)
   for (x, y, w, h) in faces:
      cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = image[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
      for (ex, ey, ew, eh) in eyes:
          cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,  255, 0), 2)
   return image
# Load the image from your local environment and perform detection
file_path = '\image.jpg'
image = cv2.imread(file_path)
output_image = detect_faces_and_eyes(image)
# Display the output
cv2_imshow(output_image)