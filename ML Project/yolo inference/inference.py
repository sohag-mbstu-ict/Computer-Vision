from ultralytics import YOLO
import cv2
import os
from img_and_video import ImageAndVideoProcessing
model = YOLO("/home/mtl/Music/DRF_keras/detection_model/pretrained_model/brinjal/no_freze_240e.pt")
# model = YOLO("/home/mtl/Music/DRF_keras/detection_model/pretrained_model/brinjal/yolo_freeze_10.pt")

class get_Info:
    def __init__(self,model):
        self.model = model

    def write_frame_num_on_frame(self,image,frame_num,img_name):
        position = (30, 30)  # X, Y coordinates
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)  # Green color in BGR
        thickness = 2
        # Add text to image
        cv2.putText(image, str(frame_num), position, font, font_scale, color, thickness)
        cv2.putText(image, str(img_name), (100,30), font, font_scale, color, thickness)
        return image

    def get_prediction(self,image):
        result = model.predict(image,
               imgsz=640, 
               conf=0.25,
               # save=True,
               )
        image = result[0].plot()
        bboxes = result[0].boxes.data
        return image, bboxes

    def get_custom_prediction(self,image):
        image = cv2.resize(image,(640,640))
        result = model.predict(image,
               imgsz=640, 
               conf=0.25,
               # save=True,
               )
        image = result[0].plot()
        bboxes = self.remove_false_positive(result)
        return image, bboxes
    
    def remove_false_positive(self,result):
        bbox_conf_cls = result[0].boxes.data.tolist()
        class_names = result[0].boxes.cls.tolist()
        boxes = result[0].boxes.data
        # 0 ='kokrano'or borer    1 ='wilt'
        kokrano = 0
        wilt  = 1
        for cls in class_names:
            if(cls==0):
                kokrano+=1
            elif(cls==1):
                wilt+=1
        if(kokrano<4):
            bboxes = boxes
        else:
            bboxes=[]
            for box_conf_cls in bbox_conf_cls:
                if(box_conf_cls[5]==0):
                    bboxes.append(list(box_conf_cls))
        return bboxes
    
    def get_video_writer(self):
        frame_size = (1280, 720)  # Width x Height
        fps = 10  # Frames per second
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 Initialize video writer
        output_video = "/home/mtl/Music/DRF_keras/detection_model/output_video/no_freze_240e_vid.mp4"  # Output video file
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
        return video_writer
    
    def draw_bbox(self,bboxes,image):
        # bboxes = bboxes.cpu().numpy()
        for bbox in bboxes:
            x1, y1, x2, y2, conf, class_id = bbox  # Unpacking bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 0 ='kokrano'or borer    1 ='wilt'
            if(class_id)==0:
                # Add text to image
                cv2.putText(image, "bor", (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2)
            elif(class_id)==1:
                # Add text to image
                cv2.putText(image, "wilt", (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0) , 2)
        return image
    
    def get_output_detections_video(self,image_folder,img_video_obj,video_writer):
        images = img_video_obj.get_image_from_folder(image_folder)
        frame_num = 0
        for img_name in images:
            frame_num+=1
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping invalid image: {img_name}")
                continue
            img_resized = cv2.resize(img, (640, 640))
            org_image = img_resized.copy()
            pred_image, bboxes = self.get_prediction(org_image)
            output_image = self.draw_bbox(bboxes,org_image) 
            merged_image_resized = img_video_obj.get_merged_image(img_resized,output_image)
            merged_image_resized = self.write_frame_num_on_frame(merged_image_resized,frame_num,img_name)
            video_writer.write(merged_image_resized)
        video_writer.release()

    def get_output_detections_video_from_input_video(self,input_video_path,img_video_obj,video_writer):
        cap = cv2.VideoCapture(input_video_path)
        frame_count = 0  # Counter for frames
        frame_interval = 10  # Process every nth frame (adjust as needed)
        while cap.isOpened():
            frame_count+=1
            # if(frame_count>10):
            #     break
            ret, img = cap.read()
            if not ret:
                break  # Stop if video ends
            img_resized = cv2.resize(img, (640, 640))
            org_image = img_resized.copy()
            output_image, bboxes = self.get_prediction(org_image)
            # output_image = self.draw_bbox(bboxes,org_image) 
            merged_image_resized = img_video_obj.get_merged_image(img_resized,output_image)
            img_name = "" # It's blank for video
            merged_image_resized = self.write_frame_num_on_frame(merged_image_resized,frame_count,img_name)
            video_writer.write(merged_image_resized)
        video_writer.release()


infer_obj = get_Info(model)
img_video_obj = ImageAndVideoProcessing()

image_folder = "/home/mtl/Music/DRF_keras/detection_model/dataset/Brinjal_BINA/ডগা_ছির্দ্র"
image_folder = "/home/mtl/Music/DRF_keras/detection_model/dataset/Brinjal_BINA/ঢলে_পড়া"
video_writer = infer_obj.get_video_writer()
# infer_obj.get_output_detections_video(image_folder,img_video_obj,video_writer) # make output video from image folder
input_video_path = "/home/mtl/Music/DRF_keras/detection_model/dataset/Brinjal_BINA/input_video/dogha_chidro.mp4"
infer_obj.get_output_detections_video_from_input_video(input_video_path,img_video_obj,video_writer) # make output video from image folder


# # ----------------------------------------------------------------------------------------------
# image_path = "/home/mtl/Music/DRF_keras/detection_model/need_attention/spIMG_20240610_181924.jpg"
# org_image = cv2.imread(image_path)
# org_image = cv2.resize(org_image,(640,640))
# yolo_pred_image, bboxes = infer_obj.get_prediction(org_image)
# pred_image = infer_obj.draw_bbox(bboxes,org_image)
# cv2.imwrite("need_attention.jpeg", pred_image) # Save the result


# cv2.imwrite("/home/mtl/Music/DRF_keras/detection_model/dataset/saved_image/2.jpeg",output_image)
# cv2.imshow("Image Window", output_image)
# cv2.waitKey(100)
# cv2.destroyAllWindows()  
