from datetime import datetime
import time
import cv2
import numpy as np
import imagezmq
import threading
import os

from flask import Flask, render_template, Response, jsonify, url_for, session
from flask_cors import CORS
from flask import abort

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from threading import Lock
from model_processing import video_detection, process_camera_stream

app = Flask(__name__)

CORS(app)

app.config['SECRET_KEY'] = 'bechimcut'
app.config['UPLOAD_FOLDER'] = 'static/files'


frame_dict = {}
frame_dict_lock = Lock()


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def frame_receiver():
    global frame_dict
    while True:
        try:
            receiver = imagezmq.ImageHub(open_port='tcp://*:5555')
            print('ImageZMQ receiver started')
            while True:
                print("Frame receiver thread is running...")
                camera_name, frame = receiver.recv_image()
                print(f'ImgZMQ receiving frames from {camera_name} at {datetime.now()}')
                print(f"Updated frame_dict with {camera_name}. Current keys: {list(frame_dict.keys())}")
                with frame_dict_lock:
                    print(f"Type of frame data: {type(frame)}")
                    frame_dict[camera_name] = frame
                    print(f"Received and stored frame for {camera_name}. Total cameras now: {len(frame_dict)}")
                receiver.send_reply(b'OK')
        except Exception as e:
            print(f"General error processing video: {e}")
            time.sleep(5)


threading.Thread(target=frame_receiver, daemon=True).start()


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    def generate():
        while True:
            frame = frame_dict.get(camera_name)
            if frame is not None:
                # Check if frame is a NumPy array. No need to encode or reshape here.
                if isinstance(frame, np.ndarray):
                    print(f"Frame dimensions: {frame.shape}")
                    # Directly encode the NumPy array to JPEG format
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        print("Error encoding frame")
                        continue

                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print("Frame is not a NumPy array. Check the received frame format.")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_urls')
def camera_urls():
    with frame_dict_lock:
        # frame_dict["debug_camera"] = "debug_frame"
        print("Complete frame_dict contents:", frame_dict)
        camera_names = list(frame_dict.keys())
        # print("Sending camera names:", camera_names)
    return jsonify(camera_names)


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(save_path)
            session['video_path'] = save_path
            print(f"File saved at {save_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    return render_template('videoprojectnew.html', form=form)


@app.route('/all-cameras', methods=['GET', 'POST'])
def see_all_cams():
    try:
        session.clear()
        return render_template('showallcams.html')
    except Exception as e:
        print(e)
        abort(5000)


# @app.route('/trigger_update')
# def trigger_update():
#     frame_dict['manual_test_camera'] = 'manual_test_frame'
#     return jsonify(success=True)


@app.route('/debug/frame_dict')
def debug_frame_dict():
    with frame_dict_lock:
        # current_frame_dict = dict(frame_dict)
        # return jsonify({camera: "Frame data" for camera in current_frame_dict.keys()})
        return jsonify(list(frame_dict.keys()))


# with frame_dict_lock:
#     frame_dict['dummy_camera'] = 'dummy_data'


if __name__ == "__main__":
    print("Starting frame receiver thread...")
    threading.Thread(target=frame_receiver, daemon=True).start()
    print("Frame receiver thread started.")
    app.run(debug=True, threaded=True)
