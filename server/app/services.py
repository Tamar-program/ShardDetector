from ultralytics import YOLO
import cv2
import numpy as np
import sys
sys.path.append("C:\\Users\\user\\segment-anything")
from segment_anything import SamPredictor, sam_model_registry
import torch
from google import genai
import tempfile
import os

# YOLO model
model = YOLO('C:/Users/user/runs/detect/run200imgsAnd100epochs2/weights/best.pt')

# SAM model
sam_checkpoint = r"C:\Users\user\Downloads\sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Gemini client
API_KEY = "AIzaSyCQMrxPRQcH64P7u9nPhY9koIqSiZdmzv0"
client = genai.Client(api_key=API_KEY)

# Global variable for storing an image
image = []

# Auxiliary functions
def masks_overlap(mask1, mask2):
    m1 = mask1.cpu().numpy() if hasattr(mask1, "cpu") else mask1
    m2 = mask2.cpu().numpy() if hasattr(mask2, "cpu") else mask2
    intersection = np.logical_and(m1, m2).sum()
    area1, area2 = m1.sum(), m2.sum()
    if intersection == 0: return 0
    return max(intersection / float(area1), intersection / float(area2))



def process_shard_with_gemini(crop_img):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, crop_img)
        file_path = tmp.name
    try:
        my_file = client.files.upload(file=file_path)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[my_file, """Extract the text from the image according to these rules:
1. If no text is detected, return an empty string.
2. Only the characters [A, B, L] (uppercase) and digits [0-9] are allowed.
3. Letters (A, B, L) may only appear at the beginning of a word.
4. The result must contain exactly 3 words.
5. Rarely, the text may include the characters '-', '\' or '/'.
Return only the cleaned text that follows these constraints."""]
        )
        return response.text
    finally:
        os.remove(file_path)
