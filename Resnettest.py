import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load the ResNet model
model = ResNet50(weights='imagenet')

# Function to preprocess the frame for ResNet input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # ResNet input size
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = preprocess_input(frame)        # Apply ResNet-specific preprocessing
    return frame

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame and make predictions
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)

    # Decode the top prediction
    decoded_preds = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
    label, description, confidence = decoded_preds[0]

    # Filter based on specific classes (example: car, person, etc.)
    # Adjust `description` values to your needs
    if description in ["phone", "sunglasses", "speaker"]:
        print(f"Detected: {description} ({confidence*100:.2f}%)")

        # Show the frame and overlay text for detected object
        cv2.putText(frame, f"{description}: {confidence*100:.2f}%", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame in real-time
    cv2.imshow('Real-Time ResNet Classification', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
