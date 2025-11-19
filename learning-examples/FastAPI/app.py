import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"

# Ensure folders exist
for d in (UPLOAD_DIR, OUTPUT_DIR, TEMPLATE_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Image Rotator")

# Static mounts so images can be viewed in the browser
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _safe_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _open_image(file_path: Path) -> Image.Image:
    # Open and auto-orient based on EXIF
    im = Image.open(file_path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/rotate")
async def rotate_image(
    request: Request,
    angle: float = Form(..., description="Rotation angle in degrees (clockwise)"),
    image: UploadFile = File(..., description="Image file to rotate"),
):
    ext = _safe_ext(image.filename)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Persist upload
    upload_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = UPLOAD_DIR / upload_name

    with upload_path.open("wb") as f:
        f.write(await image.read())

    # Rotate
    try:
        im = _open_image(upload_path)
        # Pillow rotates counter-clockwise for positive angles; invert to match clockwise UI
        rotated = im.rotate(-angle, expand=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    # Always save output as PNG to avoid EXIF issues
    output_name = f"{Path(image.filename).stem}-rotated-{int(angle)}.png"
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}-{output_name}"
    rotated.save(output_path, format="PNG")

    ctx = {
        "request": request,
        "angle": angle,
        "uploaded_rel": f"/uploads/{upload_path.name}",
        "output_rel": f"/outputs/{output_path.name}",
        "download_filename": output_path.name,
    }
    return templates.TemplateResponse("result.html", ctx)


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # Force a nice download name
    return FileResponse(
        path,
        media_type="image/png",
        filename=Path(filename).name,
    )
