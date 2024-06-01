import imagezmq
import cv2


def test_imagezmq_receiver():
    receiver = imagezmq.ImageHub(open_port='tcp://*:5555')
    print('ImageZMQ Test Receiver started')
    while True:
        camera_name, frame = receiver.recv_image()
        print(f'Received frame from {camera_name}')
        # Display the frame for testing purposes
        cv2.imshow(camera_name, frame)
        cv2.waitKey(1)  # Display the frame briefly
        receiver.send_reply(b'OK')  # Acknowledge receipt


if __name__ == '__main__':
    test_imagezmq_receiver()