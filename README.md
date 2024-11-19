# Real-Time Object Classification using TensorFlow

This project uses a TensorFlow SavedModel to perform real-time object classification with a webcam. It processes video feed, makes predictions, and displays the results on the live video feed.

---

## Features
- **Real-Time Predictions**: Uses webcam feed to classify objects in real-time.
- **Confidence Scores**: Displays the class name and confidence score for each prediction.
- **Interactive Visualization**: Annotates the webcam feed with predictions.

---

## Prerequisites
1. **Python 3.7+**
2. **Required Libraries**:
    - TensorFlow
    - OpenCV
    - NumPy

3. **Trained Model**: Ensure you have a TensorFlow SavedModel at the specified path.
4. **Labels File**: A `labels.txt` file with class names, one per line.

---

## Installation

1. Clone this repository or download the project files.
2. Install the required Python libraries:
    ```bash
    pip install tensorflow opencv-python-headless numpy
    ```
3. Place your trained model in the appropriate directory and update the `model_path` variable in the script with its location.
4. Ensure the `labels.txt` file is in the project root, with each line representing a class.

---

## Usage

1. Run the script:
    ```bash
    python filename.py
    ```
2. Allow access to your webcam when prompted.
3. Observe real-time predictions displayed on the webcam feed.

---

## Key Files

- **Script**: `real_time_classification.py`
- **Model**: Place your SavedModel under the specified path.
- **Labels File**: `labels.txt` containing the class names.

---

## Troubleshooting

- **Error: Failed to capture image**
  Ensure your webcam is connected and accessible.

- **Error loading model**
  Double-check the path to your model and ensure it's a valid TensorFlow SavedModel.

- **Incorrect predictions**
  Ensure your `labels.txt` file matches the model's class output.

---

## Customization

- To adjust the input image size, modify this line:
  ```python
  image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
