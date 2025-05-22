import os
import cv2
from inference import get_model
import supervision as sv
from dotenv import load_dotenv

# Configuración segura
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Carga de modelo seguro
modelo = get_model(
    model_id="meddia/1",
    api_key=API_KEY  # Ahora seguro
)

# Ruta relativa para imágenes
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "assets", "test_image.jpg")

try:
    imagen = cv2.imread(image_path)
    if imagen is None:
        raise FileNotFoundError(f"Imagen no encontrada en {image_path}")
    
    # Procesamiento...
except Exception as e:
    print(f"Error: {e}")
