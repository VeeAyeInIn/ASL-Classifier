import training
import cv2
import pandas as pd

from processing import process_frame

inputs_path = f'resources/X_Training_Angles.csv'
labels_path = f'resources/Y_Training.csv'

inputs = pd.read_csv(inputs_path, header=None)
labels = pd.read_csv(labels_path, header=None)

# define a video capture object
vid = cv2.VideoCapture(0)

model = training.ASLModel()
model.train(inputs, labels)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    angles, image = process_frame(frame)

    if angles is not None:
        prediction = model.predict([angles], k=5)
        print(f'Prediction: {prediction}')
    else:
        prediction = '?'

    image = cv2.putText(image, prediction[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow(f'ASL-Classifier', image)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
