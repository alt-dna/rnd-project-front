import cv2


def test_opencv_encoding():
    # Open the video file
    cap = cv2.VideoCapture('static/files/accident.mp4')

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame from the video")
        return

    # Encode the frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        # Save the encoded frame to disk for inspection
        with open('test_encoded_image.jpg', 'wb') as f:
            f.write(buffer)
        print("Encoding successful, saved as test_encoded_image.jpg")
    else:
        print("Failed to encode image")

    # When everything done, release the video capture object
    cap.release()


if __name__ == '__main__':
    test_opencv_encoding()
