import io
from pathlib import Path
from typing import Optional, List
import base64
import imghdr

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

# Ensure folders exist
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Dog Breed Classifier")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Model & labels
MODEL_PATH = Path("./models/dogs_best_model.keras") # change path if using --reload
model: Optional[tf.keras.Model] = None

# TODO: load your real class names here (list of 120 breeds in the right order)
CLASS_PATH = Path("./api/data/label.labels.txt") # change path if using --reload
list_breed: List[str] = []  # e.g., ["chihuahua", "japanese_spaniel", ...]


def _safe_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _open_image_from_bytes(contents: bytes) -> Image.Image:
    """Open image from raw bytes and fix orientation via EXIF."""
    im = Image.open(io.BytesIO(contents))
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im.convert("RGB")


def _preprocess_image(image: Image.Image) -> tf.Tensor:
    """
    Apply same preprocessing as training:
    - Convert to tensor
    - Resize so shortest side = 256
    - Center crop to 224x224
    - Normalize to [-1, 1]
    - Add batch dimension
    """

    # PIL -> NumPy -> Tensor
    image_np = np.array(image)  # (H, W, 3)
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)

    # Resize so that the shortest side = 256, then center crop to target_size.
    h = tf.shape(image_tf)[0]
    w = tf.shape(image_tf)[1]

    # scale factor so min(h, w) → RESIZE_MIN
    resize_min = 256.0
    scale = resize_min / tf.cast(tf.minimum(h, w), tf.float32)
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    image_tf = tf.image.resize(image_tf, (new_h, new_w), method="bilinear")

    # Normalization: [0, 255] => [-1, 1] floats
    image_tf = image_tf / 127.5 - 1.0

    # Central crop
    image_tf = tf.image.resize_with_crop_or_pad(
            image_tf, 224, 224
        )
    
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    image_tf = tf.expand_dims(image_tf, axis=0)

    return image_tf


def _classify(image: Image.Image) -> int:
    """Preprocess image and run inference, returning the predicted class index."""

    if model is None:
        raise RuntimeError("Model is not loaded")

    input_tensor = _preprocess_image(image)

    # Inference
    y_prob = model(input_tensor, training=False)  # or model.predict(input_tensor)
    pred_idx = int(tf.argmax(y_prob, axis=-1)[0].numpy())
    
    return pred_idx

@app.on_event("startup")
def load_model():
    """Load TF model once at startup."""

    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    with open(CLASS_PATH, "r") as f:
        for line in f:
            breed = line.strip().split("-", 1)[1]  # split only once
            list_breed.append(breed)
    print(f"Labels loaded from {CLASS_PATH}")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/classify")
async def classify_image(
    request: Request,
    image: UploadFile = File(..., description="Image file to classify"),
):
    ext = _safe_ext(image.filename)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Read file contents into memory (no saving to disk)
    content = await image.read()

    # Detect image type (fallback to "jpeg" if unsure)
    img_type = imghdr.what(None, h=content) or "jpeg"
    b64 = base64.b64encode(content).decode("utf-8")
    image_data_url = f"data:image/{img_type};base64,{b64}"

    # Classify
    try:
        im = _open_image_from_bytes(content)
        pred_idx = _classify(im)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    # Map index to breed name if you have CLASS_NAMES
    if list_breed and 0 <= pred_idx < len(list_breed):
        prediction_label = list_breed[pred_idx]
    else:
        prediction_label = str(pred_idx)

    ctx = {
        "request": request,
        "prediction": prediction_label,
        "prediction_index": pred_idx,
        "filename": image.filename,
        "image_data_url": image_data_url,     # data URL for inline display

    }
    return templates.TemplateResponse("result.html", ctx)
