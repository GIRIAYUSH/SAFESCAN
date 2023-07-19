import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO('best_246.pt')


def detect_objects():
    # Open the camera
    video = cv2.VideoCapture(0)

    while True:
        success, frame = video.read()

        # Perform object detection on the frame
        results = model.predict(frame)

        # Draw bounding boxes and labels on the frame
        results.render()

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame as a byte array
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    video.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
