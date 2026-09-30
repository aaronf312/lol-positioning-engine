import cv2
from flask import Flask, Response
from mss import mss
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load trained weights
model = YOLO("./weights/best.pt")

# Capture region (adjust to match your game window)
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}


def generate_frames():
    with mss() as sct:
        while True:
            # Capture screen
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Run YOLO tracking
            results = model.track(frame, conf=0.5, tracker="botsort.yaml", persist=True)
            annotated_frame = results[0].plot()

            # Encode frame to JPEG
            success, buffer = cv2.imencode(".jpg", annotated_frame)
            if not success:
                continue
            frame_bytes = buffer.tobytes()

            # Yield frame for MJPEG stream
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )


@app.route("/")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
