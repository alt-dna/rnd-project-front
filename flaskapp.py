import os
import cv2
import boto3

from flask import Flask, render_template, request, session, redirect, url_for, Response
from dotenv import load_dotenv
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from botocore.exceptions import NoCredentialsError
from ultralytics import YOLO

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'static/files'

# AWS S3 Configuration
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_REGION = os.getenv('S3_REGION')
S3_FOLDER_NAME = 'front-video/'

# Boto3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

# YOLO Model
model = YOLO("YOLO-Weights/best.pt")
classNames = ["accident", "non-accident"]


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload")


def process_frame(frame, yolo_model, class_names, confidence_threshold=0.7):
    results = yolo_model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = class_names[cls]

            if confidence >= confidence_threshold and class_name == "accident":
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def video_detection(path_x):
    cap = None
    try:
        cap = cv2.VideoCapture(path_x)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        frame_skip = 3
        frame_count = 0

        while True:
            ret, img = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip == 0:
                processed_frame = process_frame(img, model, classNames)
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


def upload_to_s3(file_path, file_name):
    try:
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            f"{S3_FOLDER_NAME}{file_name}",
            ExtraArgs={"ACL": "public-read"}
        )
        print(f"Upload Successful: {S3_FOLDER_NAME}{file_name}")
        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{S3_FOLDER_NAME}{file_name}"
    except FileNotFoundError:
        print("File not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except Exception as e:
        print(f"Upload failed: {str(e)}")
        return None


@app.route('/', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        session['video_path'] = save_path
    return render_template('upload.html', form=form)


@app.route('/video')
def video():
    video_path = session.get('video_path')
    if not video_path:
        return "No video path found in session", 404
    return Response(video_detection(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
