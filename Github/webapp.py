import io
import argparse
import os
import subprocess
import re
import time

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename,send_from_directory

from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv5 model
model = YOLO('best_246.pt')


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            print("Upload Folder is ", filepath)
            f.save(filepath)
            global imgpath
            imgpath = f.filename
            print("Printing imgpath", imgpath)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()

                image = Image.open(io.BytesIO(frame))

                yolo = YOLO('best_246.pt')
                detections = yolo.predict(image, save=True)
                return display(f.filename)
            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mpv4') #file not found status here 
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                model = YOLO('best_246.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, save=True)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)

                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break

                return video_feed()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

    image_path = folder_path+'/'+latest_subfolder+'/'+imgpath
    return render_template('index.html', image_path=image_path)


@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    print("Printing the directory", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ

    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file, environ)
    else:
        return "Invalid File Format"


def get_frame():
    folder_path = os.getcwd()
    mp4_file = 'output.mp4'
    video = cv2.VideoCapture(mp4_file)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n\r\n' + jpeg.tobytes() + b'r\n\r\n')
        time.sleep(0.1)


@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(), mimetype='multipart/x-mixed-replace;boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask App")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
