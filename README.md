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

## pip list

The following tables shows all currently installed packages in the virtual environment. Not all packages are currently
used, as some are remnants of older versions of the demo.

| Package               | Version  |
|-----------------------|----------|
| absl-py               | 2.1.0    |
| attrs                 | 23.2.0   |
| cffi                  | 1.16.0   |
| contourpy             | 1.2.0    |
| cuda-python           | 12.4.0   |
| cupy-cuda12x          | 13.0.0   |
| cycler                | 0.12.1   |
| fastrlock             | 0.8.2    |
| flatbuffers           | 23.5.26  |
| fonttools             | 4.47.2   |
| importlib-resources   | 6.1.1    |
| kiwisolver            | 1.4.5    |
| matplotlib            | 3.8.2    |
| mediapipe             | 0.10.9   |
| numpy                 | 1.26.3   |
| opencv-contrib-python | 4.9.0.80 |
| packaging             | 23.2     |
| pandas                | 2.2.1    |
| pillow                | 10.2.0   |
| pip                   | 24.0     |
| protobuf              | 3.20.3   |
| pycparser             | 2.21     |
| pyparsing             | 3.1.1    |
| python-dateutil       | 2.8.2    |
| pytz                  | 2024.1   |
| setuptools            | 65.5.1   |
| six                   | 1.16.0   |
| sounddevice           | 0.4.6    |
| tzdata                | 2024.1   |
| wheel                 | 0.38.4   |
| zipp                  | 3.17.0   |
