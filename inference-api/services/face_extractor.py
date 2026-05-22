import math
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceExtractor:
    def __init__(self, model_path: str = "assets/blaze_face_short_range.tflite"):
        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options, 
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.70
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self.offset_percentage = 0.30

    def extract_main_face(self, pil_image: Image.Image) -> Image.Image:
        """
        Detects faces in the given PIL Image, scores them to find the main face,
        and returns the cropped main face. Returns None if no face is detected.
        """
        # Convert PIL Image to numpy array (RGB)
        frame = np.array(pil_image)
        
        img_h, img_w, _ = frame.shape
        frame_cx, frame_cy = img_w / 2, img_h / 2
        
        # Mediapipe requires the image to be in ImageFormat.SRGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self.detector.detect(mp_image)
        
        if not results.detections:
            return None
            
        best_face_bbox = None
        highest_score = -float('inf')
        
        for detection in results.detections:
            bbox = detection.bounding_box
            confidence = detection.categories[0].score
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            face_cx, face_cy = x + (w / 2), y + (h / 2)
            
            area = w * h
            distance_to_center = math.sqrt((frame_cx - face_cx)**2 + (frame_cy - face_cy)**2)
            score = (area * confidence) - (distance_to_center * 50) 
            
            if score > highest_score:
                highest_score = score
                best_face_bbox = (x, y, w, h)
                
        if not best_face_bbox:
            return None
            
        # Crop with offset
        x, y, w, h = best_face_bbox
        offset_w = int(w * self.offset_percentage)
        offset_h = int(h * self.offset_percentage)
        
        new_x = max(0, x - offset_w)
        new_y = max(0, y - offset_h)
        new_w = min(img_w - new_x, w + (2 * offset_w))
        new_h = min(img_h - new_y, h + (2 * offset_h))
        
        cropped_face_np = frame[new_y:new_y+new_h, new_x:new_x+new_w]
        
        # Convert back to PIL Image
        if cropped_face_np.size > 0:
            return Image.fromarray(cropped_face_np)
            
        return None
