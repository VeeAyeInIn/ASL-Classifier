import training
import cv2
import pandas as pd

from processing import process_frame

# Read the training data and labels from file.
inputs_path = f'resources/X_Training_Angles.csv'
labels_path = f'resources/Y_Training.csv'
inputs = pd.read_csv(inputs_path, header=None)
labels = pd.read_csv(labels_path, header=None)

vid = cv2.VideoCapture(0)

# Instantiate and train the model using the provided training data and labels.
model = training.ASLModel()
model.train(inputs, labels)

# Loop until appropriate keyboard interrupt ('q').
while True:

    # Read in the frame from the webcam.
    ret, frame = vid.read()

    # Process the image. Returns the angles of the hand's landmarks and the annotated frame.
    angles, image = process_frame(frame)

    # Check if the frame included a (processed) hand.
    if angles is not None:
        prediction = model.predict([angles], k=5)
        # print(f'Prediction: {prediction}')
    else:
        prediction = '?'

    # Label the image with the prediction.
    image = cv2.putText(image, prediction[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the annotated image, and check if the window should close.
    cv2.imshow(f'ASL-Classifier', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Kill the window.
vid.release()
cv2.destroyAllWindows()
