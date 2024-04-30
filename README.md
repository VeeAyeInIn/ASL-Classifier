# CS 4824 ASL Classification via Hand Landmarks

| Name        | PID          |
| ----------- |--------------|
| Adam Hall   | adamjhall129 |
| Connor Vann | veeayeinin   |
| Jay Gardner | jayg0706     |

## Requirements

The model utilizes [MediaPipe]([https://developers.google.com/mediapipe]) in order to process OpenCV2 frames from the
webcam.

`python.exe -m pip install mediapipe`

Data is already processed in the /resources/ directory.

## Running the Demo

To run the demo, it needs access to your webcam. To prevent too much strain, it will only process one hand at a time, so
using only one is recommended. The top-left of the window should display the predicted label, with the placeholder '?'
when a hand is not detected, or cannot make a prediction.

### Unsupported Letters

The letters 'j' and 'z' in ASL contain motion, so they will not accurately be predicted, since the model uses static
images to make predictions.

## Performance

By default, the model is throttled to only use 1 out of every 20 data points (5%). Without GPU acceleration, it is not
recommended to run this using all data points.

## Colab

Python Notebooks that we used to process data and form the beginning of our project can be found in the /colab/
directory. They are used as standalone files on Google Colab.