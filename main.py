import cv2
from tracker import *
cap=cv2.VideoCapture("highway.mp4")
tracker=EuclideanDistTracker()
# object detection from stable camera

object_detector=cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    _, frame=cap.read()

    # extract Region Of Interest
    roi=frame[340:720, 500:800]
    mask=object_detector.apply(roi)
    _, mask=cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours,_=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections=[]
    for cnt in contours:
        # calculate area and remove small elements
        area=cv2.contourArea(cnt)
        if area>100:
            x,y,w,h=cv2.boundingRect(cnt)
            detections.append([x,y,w,h])
    # object tracking
    vehicle_ids=tracker.update(detections)
    for vehicle_id in vehicle_ids:
        x,y,w,h,id=vehicle_id
        cv2.putText(roi, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key=cv2.waitKey(30)
    if key==27:
        break

cap.release()
cap.destroyAllWindows()
