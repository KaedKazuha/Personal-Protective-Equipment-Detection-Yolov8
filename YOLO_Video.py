from datetime import datetime
from ultralytics import YOLO
import cv2
import math


def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("YOLO-Weights/bestest.pt")
    classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-hardhat',
                  'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest',
                  'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
                  'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

    # Initialize variables
    start_time = datetime.now()
    detection_results = []

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                if class_name == 'Hardhat':
                    color = (0, 204, 255)
                elif class_name == "Gloves":
                    color = (222, 82, 175)
                elif class_name == "NO-hardhat":
                    color = (0, 100, 150)
                elif class_name == "Mask":
                    color = (0, 180, 255)
                elif class_name == "NO-Safety Vest":
                    color = (0, 230, 200)
                elif class_name == "Safety Vest":
                    color = (0, 266, 280)
                else:
                    color = (85, 45, 255)

                if conf > 0.6:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                    # Check if the class is NO-Mask, NO-Safety Vest, or NO-hardhat and confidence is above threshold
                    if class_name in ['NO-Mask', 'NO-Safety Vest', 'NO-hardhat']:
                        detection_results.append({
                            'class': class_name,
                            'confidence': conf,
                            'bounding_box': (x1, y1, x2, y2),
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })


        yield img

        if (datetime.now() - start_time).seconds >= 30:
            # Open the text file for appending detections
            with open('detection_results.txt', 'a') as file:
                # Write the detection results
                for detection in detection_results:

                    file.write(f"[ {detection['time']} ] {detection['class']} {detection['confidence']} {detection['bounding_box']} \n")
                file.write('\n')  # Add a newline to separate each 30-second interval

            # Reset the start time and clear the detection results list
            start_time = datetime.now()
            detection_results = []

        yield img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()