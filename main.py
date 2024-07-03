import torch
import cv2
import numpy as np
from sort.sort import Sort
from collections import deque

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/yolov5s.pt')
tracker = Sort()
classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
fps = 30
meters_per_pixel = 0.02
object_speeds = {}
confidence_threshold = 0.5


def detect_objects(frame):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    return detections


def calculate_speed(object_id, centroid, previous_positions):
    if object_id not in previous_positions:
        previous_positions[object_id] = deque(maxlen=2)
    previous_positions[object_id].append(centroid)

    if len(previous_positions[object_id]) == 2:
        delta_x = previous_positions[object_id][1][0] - previous_positions[object_id][0][0]
        delta_y = previous_positions[object_id][1][1] - previous_positions[object_id][0][1]
        distance = np.sqrt(delta_x ** 2 + delta_y ** 2) * meters_per_pixel
        speed = distance * fps * 3.6
        return speed
    return 0


def draw_results(frame, detections, tracker):
    valid_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf >= confidence_threshold and int(cls) in [classes.index(cls_name) for cls_name in classes]:
            valid_detections.append([x1, y1, x2, y2, conf, int(cls)])

    if valid_detections:
        valid_detections = np.array(valid_detections)
        tracked_objects = tracker.update(valid_detections[:, :5])

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            obj_centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            min_distance = float('inf')
            best_class_id = -1
            best_detection = None

            for det in valid_detections:
                det_centroid = np.array([(det[0] + det[2]) / 2, (det[1] + det[3]) / 2])
                distance = np.linalg.norm(obj_centroid - det_centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_class_id = int(det[5])
                    best_detection = det

            if best_class_id != -1:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                speed = calculate_speed(obj_id, obj_centroid, object_speeds)
                if best_detection is not None:
                    conf = best_detection[4]
                else:
                    conf = 0
                text = f"{classes[best_class_id]} Speed: {speed:.2f} km/h Conf: {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture('videos/traffic.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)
        result_frame = draw_results(frame, detections, tracker)

        cv2.imshow('YOLO Object Detection and Tracking', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
