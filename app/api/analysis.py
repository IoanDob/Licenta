import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request
from fastapi.responses import JSONResponse

# Path to your uploads directory (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # Validate extension
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type: only .csv allowed"
            )

        # Validate file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large (max 500MB)"
            )

        # Generate a unique and secure filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)

        # Save file
        with open(save_path, "wb") as buffer:
            buffer.write(contents)

        # TODO: Implement your analysis logic here using save_path

        return {"result": "File uploaded successfully", "filename": unique_name}
    except HTTPException as he:
        # Log error if desired: request.app.logger.error(f"Upload error: {he.detail}")
        raise he
    except Exception as e:
        # Log error if desired: request.app.logger.error(f"Unhandled upload error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error"}
        )