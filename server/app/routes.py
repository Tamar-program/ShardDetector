from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from app.services import (
    model,
    predictor,
    process_shard_with_gemini,
    masks_overlap,
    image  
)
import cv2
import numpy as np
from io import BytesIO
import torch


router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    global image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return ("The image was uploaded successfully. You can now activate the other functions.")

@router.post("/find")
async def find_shards():
    results = model(image)
    annotated_frame = results[0].plot()
    _, img_encoded = cv2.imencode('.jpg', annotated_frame)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@router.post("/count")
async def count_shards_in_image():
    results = model(image)
    boxes = results[0].boxes
    count = len(boxes)
    predictor.set_image(image)
    sam_boxes = [np.array(box.xyxy[0].tolist()) for box in boxes]
    transformed_boxes = predictor.transform.apply_boxes_torch(
        torch.tensor(sam_boxes, dtype=torch.float32), image.shape[:2]
    )
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    overlap_count = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks_overlap(masks[i][0], masks[j][0]) > 0.3:
                overlap_count += 1



    response = {"shards_count": count}
    if overlap_count > 10:
        response["warning"] = "The value is only approximate. Many overlaps have been identified between the potsherds, so the count may be inaccurate."
    return response

@router.post("/crop")
async def crop_shards_and_process():
    results = model(image)
    boxes = results[0].boxes
    shard_texts = []

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        x_min, y_min, x_max, y_max = map(int, xyxy)
        crop = results[0].orig_img[y_min:y_max, x_min:x_max]
        text = process_shard_with_gemini(crop)
        shard_texts.append(text)

    return {"shards_texts": shard_texts}
