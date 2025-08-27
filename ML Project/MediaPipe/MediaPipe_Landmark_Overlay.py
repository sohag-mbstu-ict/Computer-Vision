import cv2
import mediapipe as mp
from datetime import datetime
import argparse
import numpy as np
import math
import json

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.landmarks = None
        self.results = None
        self.image = None
        self.annotated_image = None
        self.black_img = None

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate angle ABC at point B (in degrees)"""
        ab = (a.x - b.x, a.y - b.y)
        cb = (c.x - b.x, c.y - b.y)
        dot = ab[0]*cb[0] + ab[1]*cb[1]
        mag = (math.sqrt(ab[0]**2 + ab[1]**2) * math.sqrt(cb[0]**2 + cb[1]**2)) + 1e-9
        return math.degrees(math.acos(max(-1, min(1, dot/mag))))

    def process_pose(self, image,frame_idx,static_image=True, min_conf=0.5, complexity=2):
        has_landmarks = True
        self.image = image.copy()
        landmarks_frame_data = None
        """Run MediaPipe Pose detection"""
        with self.mp_pose.Pose(static_image_mode=static_image,
                               min_detection_confidence=min_conf,
                               model_complexity=complexity) as pose:
            self.results = pose.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            if self.results.pose_landmarks:
                self.landmarks = self.results.pose_landmarks.landmark
                landmarks = self.results.pose_landmarks.landmark
                world_landmarks = self.results.pose_world_landmarks.landmark if self.results.pose_world_landmarks else []

                # Save JSON data
                landmarks_frame_data = {
                    "t_ms": int((frame_idx / fps) * 1000),
                    "landmarks": [
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                        for lm in landmarks
                    ],
                    "world_landmarks": [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in world_landmarks
                    ]
                }
                # pose_data.append(frame_data)
            else:
                print("No pose landmarks detected!")
                has_landmarks = False
            
            
        return self.image, has_landmarks, landmarks_frame_data
                

    def draw_black_rect(self, annotated_image,x1, y1, x2, y2):
        """Draw a black rectangle on black_img or copied image"""
        self.black_img = annotated_image.copy()
        cv2.rectangle(self.black_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    def draw_legs(self, left_ids=[23,25,27], right_ids=[24,26,28]):
        """Draw only left and right legs on black_img"""
        if not self.landmarks:
            print("No landmarks to draw.")
            return

        def draw_leg(leg_ids, color_point, color_line):
            points = []
            for idx in leg_ids:
                lm = self.landmarks[idx]
                x = 206 + int(lm.x * self.black_img.shape[1])
                y = int(lm.y * self.black_img.shape[0])
                points.append((x, y))
                cv2.circle(self.black_img, (x, y), 5, color_point, -1)
            for i in range(len(points)-1):
                cv2.line(self.black_img, points[i], points[i+1], color_line, 2)

        draw_leg(left_ids, (0,255,0), (0,255,255))
        draw_leg(right_ids, (255,0,0), (0,255,255))
        # return self.black_img

    def draw_full_pose(self,annotated_image):
        self.annotated_image = annotated_image.copy()
        """Draw full pose landmarks on annotated_image"""
        if not self.landmarks:
            print("No landmarks to draw.")
            return
        self.mp_drawing.draw_landmarks(
            self.annotated_image,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
        )
        return self.annotated_image

    def save_images(self, black_img_path="legs_only.png", annotated_img_path="annotated_image.jpg"):
        """Save images"""
        cv2.imwrite(black_img_path, self.black_img)
        cv2.imwrite(annotated_img_path, self.annotated_image)
        print(f"Saved {black_img_path} and {annotated_img_path}")

    def get_knee_angles(self):
        """Return left and right knee flexion angles"""
        if not self.landmarks:
            return None, None
        left_hip = self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = self.landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = self.landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_hip = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        return left_angle, right_angle
    
    def write_knee_angles_on_black(self, left_angle, right_angle):
        """Write left and right knee angles on black_img"""
        if not self.landmarks or self.black_img is None:
            print("No landmarks or black image to write on.")
            return
        # Timestamp at the top-left corner
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(self.black_img, f"Timestamp  : {timestamp}", (545, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        
        # Left knee position
        # 545, 222, 730, 466
        left_x, left_y = 545,175
        cv2.putText(self.black_img, f"Left Knee  : {left_angle}", (left_x, left_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)

        # Right knee position
        right_x, right_y = 545, 210
        cv2.putText(self.black_img, f"Right Knee : {right_angle}", (right_x, right_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
        return self.black_img
        
    def get_video_writer(self, video_name, fps=24, frame_size=(848, 478)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = f"{video_name}.mp4"
        return cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
        
    # -------------------- USAGE --------------------
    input_video_path = args.video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0  # Counter for frames
    pose_analyzer = PoseAnalyzer()
    video_writer = pose_analyzer.get_video_writer(args.output, fps)
    landmarks_json = "landmarks.json"
    angles_json = "angles.json"
    landmarks_data = []
    angles_data = []
    
    while cap.isOpened():
        frame_count+=1
        if frame_count%10 == 0:
            print("----------------- frame_count ---------------- : ",frame_count)
        # if(frame_count>50):
        #     break
        ret, img = cap.read()
        if not ret:
            break  # Stop if video ends 

        image, has_landmarks, landmarks_frame_data = pose_analyzer.process_pose(img,frame_count)
        if(not has_landmarks):
            continue
        annotated_image = pose_analyzer.draw_full_pose(image)
        pose_analyzer.draw_black_rect(annotated_image,545, 100, 820, 466)
        pose_analyzer.draw_legs()
        left_angle, right_angle = pose_analyzer.get_knee_angles()
        output_image = pose_analyzer.write_knee_angles_on_black(f"{left_angle:.2f}",f"{right_angle:.2f}")
        # pose_analyzer.save_images()
        video_writer.write(output_image)
        print(f"Left Knee Flexion: {left_angle:.2f}°")
        print(f"Right Knee Flexion: {right_angle:.2f}°")
        # Save JSON data
        angle_data = {
            "Frame ID": frame_count,
            "Left Knee Flexion":  left_angle,
            "Right Knee Flexion": right_angle
        }
        angles_data.append(angle_data)
        landmarks_data.append(landmarks_frame_data)
        
        
    with open(landmarks_json, "w") as f:
        json.dump(landmarks_data, f, indent=2)
    with open(angles_json, "w") as f:
        json.dump(angles_data, f, indent=2)
    video_writer.release()
