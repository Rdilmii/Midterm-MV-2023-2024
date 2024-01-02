import cv2
from ultralytics import YOLO

model = YOLO('gestures.pt')  # load an official model 
# model = YOLO('path/to/best.pt')  # load a custom model

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for bbox in boxes:
            x1,y1,x2,y2 = bbox.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            cls_idx = int(bbox.cls[0])
            cls_name = model.names[cls_idx]

            conf = round(float(bbox.conf[0]),2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
            cv2.putText(frame,f'{cls_name} {conf}',(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # frame = cv2.resize(frame,(300,300))
    
    cv2.imshow('my pic', frame)
    cv2.waitKey(1)   

vid.release()
cv2.destroyAllWindows()