import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------
# 1. Configuração
# ---------------------------------------------------------

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_PATH = "samples/images/imagem_teste.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dispositivo utilizado: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------
# 2. Carregar DINOv3
# ---------------------------------------------------------

print("\nCarregando DINOv3...")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

model = model.to(device)
model.eval()

print("DINOv3 carregado.")


# ---------------------------------------------------------
# 3. Abrir nossa imagem
# ---------------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"\nImagem original: {image.size}")


# ---------------------------------------------------------
# 4. Preparar a imagem para o DINOv3
# ---------------------------------------------------------

inputs = processor(images=image, return_tensors="pt")

pixel_values = inputs["pixel_values"].to(device)

print(f"Tensor de entrada: {pixel_values.shape}")


# ---------------------------------------------------------
# 5. Executar DINOv3
# ---------------------------------------------------------

print("\nExecutando inferência...")

with torch.inference_mode():
    outputs = model(pixel_values=pixel_values)


# ---------------------------------------------------------
# 6. Examinar resultado
# ---------------------------------------------------------

tokens = outputs.last_hidden_state

print("\n--- RESULTADO ---")

print(f"Formato dos tokens: {tokens.shape}")

print(f"Número total de tokens: {tokens.shape[1]}")
print(f"Dimensão de cada embedding: {tokens.shape[2]}")

print("\nExperimento concluído.")