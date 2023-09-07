import cv2
import numpy as np
from object_detection import ObjectDetection
import math

od = ObjectDetection()
cap = cv2.VideoCapture('los_angeles.mp4')

center_points_previous = []
tracking_objects = {}
track_id = 0
count = 0
while True:
    # ret said if that is a frame or not
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    ## center points for current frame
    center_points_current = []

    ## detect object in the Frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        x, y, w, h = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_current.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # only at the beginning we compare previous frame and current
    if count <= 2:
        for pt in center_points_current:
            for pt2 in center_points_previous:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        for object_id, pt2 in tracking_objects.copy().items():
            object_exits = False
            center_points_current_copy = center_points_current.copy()
            for pt in center_points_current_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                # Update object position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exits = True
                    if pt in center_points_current:
                        center_points_current.remove(pt)
                    continue
            if not object_exits:
                tracking_objects.pop(object_id)

        # Add new Found IDS
        for pt in center_points_current:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("frame", frame)

    # make a copy of points
    center_points_previous = center_points_current.copy()

    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

## we will create an unique id to each box
