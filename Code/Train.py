from ultralytics import YOLO
import os
import torch
from sklearn.metrics import confusion_matrix, classification_report
import cv2 
import numpy as np


def is_correct(scooter_bbox, parking_bbox):
    x_min_s = scooter_bbox[0] - scooter_bbox[2] / 2
    x_max_s = scooter_bbox[0] + scooter_bbox[2] / 2
    y_min_s = scooter_bbox[1] - scooter_bbox[3] / 2
    y_max_s = scooter_bbox[1] + scooter_bbox[3] / 2

    x_min_p = parking_bbox[0] - parking_bbox[2] / 2
    x_max_p = parking_bbox[0] + parking_bbox[2] / 2
    y_min_p = parking_bbox[1] - parking_bbox[3] / 2
    y_max_p = parking_bbox[1] + parking_bbox[3] / 2

    return (x_min_s >= x_min_p) and (x_max_s <= x_max_p) and (y_min_s >= y_min_p) and (y_max_s <= y_max_p)


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model = YOLO('yolov8x.yaml')

    args = {
        'data': 'Config.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': '0',
        'workers': 0,
        'verbose': True
    }

    model.train(**args)

    model = YOLO('runs/detect/train/weights/best.pt')

    y_true = []
    y_pred = []
    test_images_dir = 'final_dataset/test/images'

    for img_name in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_name)
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

        true_parkings = []
        true_scooters = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    if int(cls) == 0:
                        true_scooters.append([x, y, w, h])
                    elif int(cls) == 1:
                        true_parkings.append([x, y, w, h])

        results = model.predict(img_path, conf=0.5)

        pred_scooters = []
        pred_parkings = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls.item())
                x, y, w, h = box.xywhn[0].tolist()
                if cls == 0:
                    pred_scooters.append([x, y, w, h])
                elif cls == 1:
                    pred_parkings.append([x, y, w, h])

        for true_scooter in true_scooters:
            correct = 0
            if true_parkings:
                correct = 1 if is_correct(true_scooter, true_parkings[0]) else 0
            y_true.append(correct)

        for pred_scooter in pred_scooters:
            correct = 0
            if pred_parkings:
                correct = 1 if is_correct(pred_scooter, pred_parkings[0]) else 0
            y_pred.append(correct)


    os.makedirs('parking_results', exist_ok=True)
    
    model = YOLO('runs/detect/train/weights/best.pt')
    
    test_images_dir = 'final_dataset/test/images'
    for img_name in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_name)
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        results = model.predict(img_path, conf=0.5, save=False)[0]
        
        scooters = []
        parkings = []
        for box in results.boxes:
            cls = int(box.cls.item())
            coords = box.xywhn[0].tolist()
            if cls == 0:  
                scooters.append(coords)
            elif cls == 1:  
                parkings.append(coords)
        
        for parking in parkings:
            x, y, w, h = parking
            x1 = int((x - w/2) * img_width)
            y1 = int((y - h/2) * img_height)
            x2 = int((x + w/2) * img_width)
            y2 = int((y + h/2) * img_height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) 
        
        for scooter in scooters:
            x, y, w, h = scooter
            x1 = int((x - w/2) * img_width)
            y1 = int((y - h/2) * img_height)
            x2 = int((x + w/2) * img_width)
            y2 = int((y + h/2) * img_height)
            
            correctly_parked = any(is_correct(scooter, p) for p in parkings)
            
            color = (0, 255, 0) if correctly_parked else (0, 0, 255) 
            label = "Correct" if correctly_parked else "Incorrect"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        output_path = os.path.join('parking_results', img_name)
        cv2.imwrite(output_path, img)
        print(f"Результат сохранен: {output_path}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    metrics = model.val()
    print("\nDetection Metrics:")
    print(f"mAP50: {metrics.box.map50:.2f}")
    print(f"Precision: {metrics.box.mp:.2f}")
    print(f"Recall: {metrics.box.mr:.2f}")

if __name__ == "__main__":
    main()
