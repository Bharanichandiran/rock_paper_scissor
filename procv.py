import tensorflow as tf
import cv2
import numpy as np

model_path = r'C:\Users\admin\Downloads\converted_savedmodel (4)\model.savedmodel'
try:
    # Load the model
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize the webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, image = camera.read()

    if not ret:
        print("Failed to capture image")
        break

    # Resize the image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize the image
    image_normalized = (image_resized / 127.5) - 1

    # Expand dimensions to match model input shape
    input_tensor = np.expand_dims(image_normalized, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    # Make a prediction
    try:
        prediction = infer(tf.constant(input_tensor))
        prediction = prediction['sequential_3'].numpy()  # Use the correct output key
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Remove any extra spaces or newline characters
        confidence_score = prediction[0][index]

        # Print all confidence scores for better insight
        print("Predictions:")
        for i, score in enumerate(prediction[0]):
            print(f"  {class_names[i].strip()}: {score * 100:.2f}%")

        # Display prediction and confidence score on the frame
        cv2.putText(image, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Confidence Score: {str(np.round(confidence_score * 100, 2))}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error during prediction: {e}")

    # Display the frame
    cv2.imshow('Webcam Feed', image)

    # Listen for keyboard input
    keyboard_input = cv2.waitKey(1)

    # Exit when the ESC key is pressed
    if keyboard_input == 27:
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()
