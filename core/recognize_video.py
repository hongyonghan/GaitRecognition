from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import json
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import socket
import threading
import utils.imageEnhancer as image_enhancer

FD_FOLDER = 'face_detection_model'
EMBEDDINGS_MODEL = 'openface_nn4.small2.v1.t7'
RECOGNIZER = 'output/recognizer.pickle'

LABEL_ENCODER = 'output/le.pickle'

vs = None
outputFrame = None
lock = threading.Lock()
current = "Unkown"
users = {}

# sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
# 	"192.168.43.132"))

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup

def recognize(inp_confidence, vid_file):
    # rpiName = socket.gethostname()
    # print(rpiName + "*************")
    global vs,outputFrame, lock, current, users

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([FD_FOLDER, "deploy.prototxt"])
    modelPath = os.path.sep.join([FD_FOLDER,"res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(EMBEDDINGS_MODEL)

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(RECOGNIZER, "rb").read())
    le = pickle.loads(open(LABEL_ENCODER, "rb").read())

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    vs = FileVideoStream(vid_file).start()
    time.sleep(2.0)

    # start the FPS throughput estimator
    fps = FPS().start()
    users = {}
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        frame = image_enhancer.image_enhance(frame)
        frame =  image_enhancer.image_sharpen(frame)
        # sender.send_image(rpiName, cv2.resize(frame, (640,320)))
        
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame)
        (h, w) = frame.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (720, 1280)), 1.0, (800,800),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > float(inp_confidence):
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                
                # face = cv2.resize(face, (160,160), interpolation = cv2.INTER_AREA)
                # sample = np.expand_dims(face, axis=0)
                # vec = model.predict(sample)

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                current = name
                if not current in users:
                    users[current] = 1 * proba
                else:
                    users[current] = users[current] + 1 * proba
                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        fps.update()

        with lock:
            outputFrame = frame.copy()
            # cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

def current_identification():
    global current
    print(current)
    yield "data: " + current + "\n\n"

def all_count():
    global users
    print(users)
    yield "data: " + json.dumps(users) + "\n\n"