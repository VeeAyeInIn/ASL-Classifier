import time
import training
import cv2
import pandas as pd

from processing import process_frame

inputs_path = f'resources/X_Training_Angles.csv'
labels_path = f'resources/Y_Training.csv'

inputs = pd.read_csv(inputs_path, header=None)
labels = pd.read_csv(labels_path, header=None)

vid = cv2.VideoCapture(0)
model = training.ASLModel()
model.train(inputs, labels)

while True:

    ret, frame = vid.read()
    angles, image = process_frame(frame)

    if angles is not None:
        prediction = model.predict([angles], k=5)
        print(f'Prediction: {prediction}')
    else:
        prediction = '?'

    image = cv2.putText(image, prediction[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow(f'ASL-Classifier', image)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'C:\\Users\\veeay\\PycharmProjects\\MachineLearningStuff\\Prediction {prediction} {int(time.time())}.jpg', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
