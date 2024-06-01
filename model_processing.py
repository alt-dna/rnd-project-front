from ultralytics import YOLO
import cv2
import base64
import math
import json

model = YOLO("YOLO-Weights/best.pt")
classNames = ["accident", "non-accident"]


def process_frame(frame, model, classNames):
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def video_detection(path_x):
    cap = None  # Initialize cap to None
    try:
        cap = cv2.VideoCapture(path_x)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        frame_skip = 3
        frame_count = 0

        while True:
            ret, img = cap.read()
            if not ret:
                break  # Exit loop if no frames are left

            frame_count += 1
            if frame_count % frame_skip == 0:
                processed_frame = process_frame(img, model, classNames)  # Use the refactored function
                try:
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                    else:
                        print(f"Failed to encode frame {frame_count}")
                except Exception as e:
                    print(f"Exception while encoding frame {frame_count}: {e}")

    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        if cap:
            cap.release()
            cv2.destroyAllWindows()


def process_camera_stream(stream_url, camera_id):
    print(f"Processing stream for camera {camera_id} from URL: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    frame_skip = 3
    frame_count = 0

    if not cap.isOpened():
        print(f"Error: Could not open stream for camera {camera_id}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame for camera {camera_id}")
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            print(f"Processing frame {frame_count} for camera {camera_id}")
            processed_frame = process_frame(frame, model, classNames)
            try:
                processed_frame = cv2.resize(processed_frame, (640, 480))
                ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    json_data = json.dumps({'camera_id': camera_id, 'frame': frame_data})
                    yield f"data: {json_data}\n\n"
                else:
                    print(f"Failed to encode frame {frame_count} for camera {camera_id}")
            except Exception as e:
                print(f"Exception while encoding frame {frame_count} for camera {camera_id}: {e}")

    cap.release()
    print(f"Stopped processing stream for camera {camera_id}")
