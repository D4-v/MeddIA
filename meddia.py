
#imagen = cv2.imread("I:/OscarDRoblesB/py/Fotos/1108.jpg")
import os
import cv2
from inference import get_model
import supervision as sv
import warnings
import requests
# === API Key ===
#os.environ["ROBOFLOW_API_KEY"] = "rf_ErVwPbll04evlDgUJ6s7wDHHeq63"

warnings.filterwarnings("ignore", category=UserWarning)  # Silencia todos los UserWarnings

PRIVATE_KEY = "J1IL5DzExCs8tSMRGi4u"  # La que revelaste

url = f"https://api.roboflow.com/ort/meddia/1?api_key={PRIVATE_KEY}&device=cpu"
response = requests.get(url)
print(response.status_code, response.json())  # Debería devolver 200

modelo = get_model(model_id="meddia/1", api_key="J1IL5DzExCs8tSMRGi4u")  # <-- Usa la PRIVATE key


# === Cargar imagen ===
#""C:\Users\funky\OneDrive\Documentos\Test.jpg""
ruta = "C:/Users/funky/OneDrive/Documentos/test6.jpg"  
imagen = cv2.imread(ruta)

if imagen is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")

# === Cargar modelo ===
modelo = get_model(
    model_id="meddia/1",
    api_key="J1IL5DzExCs8tSMRGi4u",
)
#modelo = get_model(model_id="meddia/1")

# === Inferencia ===
resultado = modelo.infer(imagen)[0]
detecciones = sv.Detections.from_inference(resultado)
try:
    resultado = modelo.infer(imagen)[0]
except Exception as e:
    print(f"Error en la inferencia: {e}")

# === Dibujar cajas ===meddia/1
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

imagen_anotada = box_annotator.annotate(scene=imagen, detections=detecciones)
imagen_anotada = label_annotator.annotate(scene=imagen_anotada, detections=detecciones)


# === Mostrar resultado con OpenCV ===
cv2.imshow("Resultado", imagen_anotada)
cv2.waitKey(0)
cv2.destroyAllWindows()