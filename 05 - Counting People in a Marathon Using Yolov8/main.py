from ultralytics import YOLO
import numpy as np
from tracker import Tracker
import cv2
import random
import cvzone

model = YOLO('yolov8s.pt')
tracker = Tracker()
cap = cv2.VideoCapture('p3.mp4')

with open('coco.txt', 'r') as f:
    classes = f.readlines()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

cy1 = 450
offset = 4

count = 0
counter = []
while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame , (1020, 500))
    results = model.predict(frame)
    boxes = results[0].boxes
    detections = []

    for i in range(len(boxes)):
        item = boxes[i].data.cpu().numpy()[0]
        x1 = int(item[0])
        y1 = int(item[1])
        x2 = int(item[2])
        y2 = int(item[3])
        d = int(item[5])
        if "person" in classes[d]:
            detections.append([x1, y1, x2, y2])
    bboxes = tracker.update(detections)

    for bbox in bboxes:
        x1, y1, x2, y2, object_id = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[int(object_id) % 10], 2)
        cv2.putText(frame, str(object_id), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[int(object_id) % 10], 1)

        # if person's center matches line then we will increase counter by one
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            if counter.count(object_id) == 0:
                counter.append(object_id)
    cv2.line(frame, (355, cy1), (696, cy1), (0, 255, 0), 2)
    num_persons = len(counter)

    cvzone.putTextRect(frame, f"People Count : {num_persons}", (50, 50), 2, 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
