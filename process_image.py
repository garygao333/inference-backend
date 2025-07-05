import os, json, cv2, numpy as np, base64
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

OUTPUT_DIR = "output"
API_KEY = os.getenv("API_KEY")
WORKSPACE = os.getenv("WORKSPACE")
WORKFLOW_ID_FINE = os.getenv("WORKFLOW_ID_FINE")
WORKFLOW_ID_COARSE = os.getenv("WORKFLOW_ID_COARSE")
SERVERLESS_URL = 'https://serverless.roboflow.com'

def process_image(image_path: str, total_weight: float = 100, material_type: str = "fine-ware"):
    """
    Process an image to detect and classify sherds.
    
    Args:
        image_path: Path to the image file
        total_weight: Total weight of all sherds in grams
        material_type: Type of material ('fine-ware' or 'coarse-ware')
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Select workflow ID based on material type
    if material_type == "coarse-ware":
        workflow_id = WORKFLOW_ID_COARSE
    else:  # Default to fine-ware
        workflow_id = WORKFLOW_ID_FINE
        
    if not workflow_id:
        raise ValueError(f"No workflow ID configured for material type: {material_type}")
    
    client = InferenceHTTPClient(api_url=SERVERLESS_URL, api_key=API_KEY)
    result = client.run_workflow(WORKSPACE, workflow_id, images={"image": image_path}, use_cache=False)
    result = result[0]
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detections = result["detection_predictions"]["predictions"]
    type_preds = result["classification_predictions"]
    qual_preds = result["model_predictions"]
    
    type_map = {p["predictions"]["parent_id"]: p["predictions"].get("top", "unknown") for p in type_preds}
    qual_map = {p["parent_id"]: p.get("top", "unknown") for p in qual_preds}
    
    total_area = sum(d["width"] * d["height"] for d in detections)
    
    annotated_image = image_rgb.copy()
    
    sherds = []
    for i, d in enumerate(detections):
        area = d["width"] * d["height"]
        weight = (area / total_area) * total_weight if total_area > 0 else 0
        
        x, y, w, h = int(d["x"] - d["width"]/2), int(d["y"] - d["height"]/2), int(d["width"]), int(d["height"])
        
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        label = f"Sherd {i + 1}"
        type_pred = type_map.get(d["detection_id"], "unknown")
        qual_pred = qual_map.get(d["detection_id"], "unknown")
        
        cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(annotated_image, f"{type_pred}/{qual_pred}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        sherds.append({
            "sherd_id": f"Sherd {i + 1}",
            "weight": round(weight, 2),
            "type_prediction": type_pred,
            "qualification_prediction": qual_pred
        })
    
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "sherds": sherds,
        "annotated_image": annotated_image_base64
    }

# import os, json, cv2, numpy as np, base64
# import requests
# from dotenv import load_dotenv
# from typing import Optional

# load_dotenv()

# OUTPUT_DIR = "output"
# API_KEY = os.getenv("API_KEY")
# WORKSPACE = os.getenv("WORKSPACE")
# WORKFLOW_ID_FINE = os.getenv("WORKFLOW_ID_FINE")
# WORKFLOW_ID_COARSE = os.getenv("WORKFLOW_ID_COARSE")
# SERVERLESS_URL = os.getenv("SERVERLESS_URL")

# def process_image(image_path: str, total_weight: float = 100, material_type: str = "fine-ware"):
#     """
#     Process an image to detect and classify sherds.
    
#     Args:
#         image_path: Path to the image file
#         total_weight: Total weight of all sherds in grams
#         material_type: Type of material ('fine-ware' or 'coarse-ware')
#     """
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Select workflow ID based on material type
#     if material_type == "coarse-ware":
#         workflow_id = WORKFLOW_ID_COARSE
#     else:  # Default to fine-ware
#         workflow_id = WORKFLOW_ID_FINE
        
#     if not workflow_id:
#         raise ValueError(f"No workflow ID configured for material type: {material_type}")
    
#     url = f"https://serverless.roboflow.com/{WORKSPACE}/{workflow_id}?api_key={API_KEY}&use_cache=false"

#     with open(image_path, "rb") as image_file:
#         response = requests.post(
#             url,
#             files={"image": image_file}
#         )

#     response.raise_for_status()
#     result = response.json()


#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     detections = result["detection_predictions"]["predictions"]
#     type_preds = result["classification_predictions"]
#     qual_preds = result["model_predictions"]
    
#     type_map = {p["predictions"]["parent_id"]: p["predictions"].get("top", "unknown") for p in type_preds}
#     qual_map = {p["parent_id"]: p.get("top", "unknown") for p in qual_preds}
    
#     total_area = sum(d["width"] * d["height"] for d in detections)
    
#     annotated_image = image_rgb.copy()
    
#     sherds = []
#     for i, d in enumerate(detections):
#         area = d["width"] * d["height"]
#         weight = (area / total_area) * total_weight if total_area > 0 else 0
        
#         x, y, w, h = int(d["x"] - d["width"]/2), int(d["y"] - d["height"]/2), int(d["width"]), int(d["height"])
        
#         cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
#         label = f"Sherd {i + 1}"
#         type_pred = type_map.get(d["detection_id"], "unknown")
#         qual_pred = qual_map.get(d["detection_id"], "unknown")
        
#         cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#         cv2.putText(annotated_image, f"{type_pred}/{qual_pred}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
#         sherds.append({
#             "sherd_id": f"Sherd {i + 1}",
#             "weight": round(weight, 2),
#             "type_prediction": type_pred,
#             "qualification_prediction": qual_pred
#         })
    
#     _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#     annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
#     return {
#         "sherds": sherds,
#         "annotated_image": annotated_image_base64
#     }
