import cv2
import time
backend = getattr(cv2, 'CAP_DSHOW', cv2.CAP_ANY)
capture = cv2.VideoCapture(0, backend)
print('opened', capture.isOpened(), flush=True)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print('requested', capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT), flush=True)
start = time.time()
ok, frame = capture.read()
print('read', ok, None if frame is None else frame.shape, 'seconds', time.time() - start, flush=True)
capture.release()
