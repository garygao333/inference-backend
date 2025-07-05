from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from process_image import process_image
import shutil
import uuid
import os

app = FastAPI()

os.makedirs("uploads", exist_ok=True)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...), 
    weight: float = Form(...),
    material_type: str = Form("fine-ware")
):
    try:
        print(f"Received image: {image.filename}, weight: {weight}")
        
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_filename = f"uploads/{uuid.uuid4()}.jpg"
        with open(image_filename, "wb") as f:
            shutil.copyfileobj(image.file, f)
        
        print(f"Image saved to: {image_filename}")
        
        results = process_image(image_filename, total_weight=weight, material_type=material_type)
        
        try:
            os.remove(image_filename)
        except:
            pass
            
        print(f"Analysis complete. Found {len(results)} sherds")
        return results
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)