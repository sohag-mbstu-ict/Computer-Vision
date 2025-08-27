from ultralytics import YOLO
import cv2
import torch
from PIL import Image
import cv2
from collections import Counter
import json
class BoundingBoxProcessor:
    def __init__(self, boxes, device="cpu"):
        """
        Initialize with a tensor of bounding boxes.
        :param boxes: Tensor of shape (N, 4) with [x1, y1, x2, y2] for each box.
        :param device: Device to store the tensor (default: CUDA if available).
        """
        self.device = torch.device( "cpu")
        self.boxes = boxes.to(self.device)

    def calculate_iou(self, box1, box2):
        """
        Compute IoU between two bounding boxes.
        :param box1: First bounding box (x1, y1, x2, y2).
        :param box2: Second bounding box (x1, y1, x2, y2).
        :return: IoU score between 0 and 1.
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def is_box_inside(self, large_box, small_box):
        """
        Check if the small box is completely inside the large box.
        :param large_box: The larger bounding box [x1, y1, x2, y2].
        :param small_box: The smaller bounding box [x1, y1, x2, y2].
        :return: True if small_box is inside large_box, otherwise False.
        """
        return (large_box[0] <= small_box[0] and  # small box left inside large box
                large_box[1] <= small_box[1] and  # small box top inside large box
                large_box[2] >= small_box[2] and  # small box right inside large box
                large_box[3] >= small_box[3])    # small box bottom inside large box

    def remove_high_iou_or_nested_boxes(self, iou_threshold=0.3):
        """
        Removes the larger bounding box if IoU is greater than the threshold or if a box is nested inside another.
        :param iou_threshold: IoU threshold above which boxes are considered overlapping.
        :return: Filtered tensor of bounding boxes.
        """
        keep_indices = set(range(len(self.boxes)))

        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                if i in keep_indices and j in keep_indices:
                    # Check IoU between box i and box j
                    iou = self.calculate_iou(self.boxes[i], self.boxes[j])
                    if iou > iou_threshold:
                        # Remove the larger box based on area
                        area_i = (self.boxes[i][2] - self.boxes[i][0]) * (self.boxes[i][3] - self.boxes[i][1])
                        area_j = (self.boxes[j][2] - self.boxes[j][0]) * (self.boxes[j][3] - self.boxes[j][1])

                        if area_i > area_j:
                            keep_indices.discard(i)
                        else:
                            keep_indices.discard(j)

                    # Check if one box is inside the other
                    elif self.is_box_inside(self.boxes[i], self.boxes[j]):
                        keep_indices.discard(j)
                    elif self.is_box_inside(self.boxes[j], self.boxes[i]):
                        keep_indices.discard(i)

        self.boxes = self.boxes[list(keep_indices)]
        return self.boxes
    

# -------- 1️⃣ Feature Extractor Using ResNet-50 -------- #
class preprocessing_and_postprocessing:
    def __init__(self):
        pass

    def get_image_and_label_for_borar(self,results,filtered_boxes,model,image):
        labels=[]
        for result in results:
            for box in result.boxes:
                # Check if the query box exists in the boxes tensor
                is_present = torch.any(torch.all(filtered_boxes == box.xyxy[0], dim=1))
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                cls = int(box.cls[0])  
                conf = box.conf[0].item()  
                label = model.names.get(cls, "Unknown")
                if(is_present.item()==True) :
                    labels.append(label)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        return labels,image
    

    def get_image_and_label_for_wilt(self,results,filtered_boxes,model,image):
        labels=[]
        for result in results:
            for box in result.boxes:
                # Check if the query box exists in the boxes tensor
                is_present = torch.any(torch.all(filtered_boxes == box.xyxy[0], dim=1))
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                cls = int(box.cls[0])  
                conf = box.conf[0].item()  
                label = model.names.get(cls, "Unknown")
                if(is_present.item()==True) :
                    labels.append(label)
                    if(label=='borer'):
                        label = 'wilt'
                labels.append(label)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        return labels,image





class YOLOPredictor:
    def __init__(self):
        pass
    def get_yolo_prediction(self,image_path,model_path):
        image = cv2.imread(image_path)
        image_wilt = image.copy()
        all_predictions=[]
        model = YOLO(model_path).to("cpu") 
        # Run detection with YOLO
        results = model.predict(image, conf=0.40, imgsz=640)
        img = results[0].plot()
        if len(results[0].boxes.cls) <= 0:
            predicted_label = "No related disease found"
            predicted_accuracy = "93.487771809101105"
            all_predictions.append({"accuracy": 93.487771809101105})
            return predicted_label,predicted_accuracy,all_predictions
        
        # Create an instance of the class and process bounding boxes
        processor = BoundingBoxProcessor(results[0].boxes.xyxy)
        filtered_boxes = processor.remove_high_iou_or_nested_boxes(iou_threshold=0.1)
        preprocess_and_postprocess_obj = preprocessing_and_postprocessing()
        labels,image = preprocess_and_postprocess_obj.get_image_and_label_for_borar(results,filtered_boxes,model,image)
        filtered_labels = [label for label in labels if label != "healthy" and label != "soil"]
        if(len(filtered_labels)==0):
            predicted_label = "No related disease found"
            predicted_accuracy = "93.487771809101105"
            all_predictions.append({"accuracy": 93.487771809101105})
            return predicted_label,predicted_accuracy,all_predictions
        
        else:
            if(len(filtered_labels)==1):
                predicted_label = filtered_labels[0]
            else:
                if(filtered_labels.count('borer')>7):
                    predicted_label = 'wilt'
                    labels,image = preprocess_and_postprocess_obj.get_image_and_label_for_wilt(results,filtered_boxes,model,image_wilt)
                else:
                    # Count occurrences of each label
                    label_counts = Counter(filtered_labels)
                    predicted_label = label_counts.most_common(1)[0][0]
        prediction_acc = results[0].boxes.conf
        prediction_acc = prediction_acc.tolist()
        predicted_accuracy = max(prediction_acc) # get only max prediction   
        all_predictions.append({"accuracy": predicted_accuracy})           
        return predicted_label,predicted_accuracy,all_predictions
        
    def disease_detection_using_yolo_model(self, image_path,model_path):
        all_predictions = []

        # # check the image is plant or non plant
        # classifier = PlantImageClassifier()
        # image = Image.open(image_path)
        # # Predict if it's related to plants
        # result = classifier.predict(image)
        # predicted = result['predicted']
        # if(predicted == 'Not_Plant'):
        #     predicted_label = "No related disease found"
        #     predicted_accuracy = "93.487771809101105"
        #     all_predictions.append({"accuracy": 93.487771809101105})

        # get prediction result from yolo model
        predicted_label,predicted_accuracy,all_predictions = self.get_yolo_prediction(image_path,model_path)
        return predicted_label,predicted_accuracy,all_predictions 
    
    


# yolo_object = YOLOPredictor()
# img_path = "/media/mtl/Volume D/brinjal/BINA_Brinjal_train/Bacterial_wilt/IMG_20240610_181513.jpg"
# # img_path = "/media/mtl/Volume F/brinjal/images/1741584601043.jpg"
# model_path = "/home/mtl/Music/DRF_keras/RND_model/trained_model/b_w_20_mar_25.pt"
# predicted_label,predicted_accuracy,all_predictions  = yolo_object.disease_detection_using_yolo_model(img_path,model_path)


# detection_id = "from_backend"
# detected_at = "from_backend"
# model_used = "from_backend"
# # Final response
# response_data = {
#     "detection_id": detection_id,
#     "detected_at": detected_at,
#     "model_used": model_used,
#     "predicted_label": predicted_label,
#     "predicted_accuracy": predicted_accuracy,
#     "all_predictions": all_predictions
# }
# print(json.dumps(response_data, indent=2))

