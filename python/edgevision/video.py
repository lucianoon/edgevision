import cv2


class VideoSource:
    def __init__(self, source: int | str):
        self.source = source
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        return self.capture.read()

    def release(self):
        self.capture.release()
