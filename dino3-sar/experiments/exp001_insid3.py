import sys
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModel



# =========================================================
# 1. CAMINHOS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSID3_ROOT = PROJECT_ROOT / "external" / "INSID3"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INSID3_ROOT))

from models.insid3 import INSID3
from src.models.dinov3_hf_adapter import DINOv3HFAdapter


REFERENCE_IMAGE = (
    PROJECT_ROOT / "samples" / "images" / "imagem_teste.jpg"
)

REFERENCE_MASK = (
    PROJECT_ROOT / "samples" / "images" / "mascara_teste.png"
)

TARGET_IMAGE = (
    PROJECT_ROOT / "samples" / "images" / "imagem_alvo.jpg"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "images"
)

OUTPUT_MASK = (
    OUTPUT_DIR / "exp001_insid3_mask.png"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. VERIFICAR ARQUIVOS
# =========================================================

for path in [
    REFERENCE_IMAGE,
    REFERENCE_MASK,
    TARGET_IMAGE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}"
        )


# =========================================================
# 3. GPU
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA não está disponível."
    )

device = "cuda"

print("=" * 60)
print("EXP001 - DINOv3 + INSID3")
print("=" * 60)

print("\nGPU:")
print(torch.cuda.get_device_name(0))

print(
    "CUDA PyTorch:",
    torch.version.cuda
)

torch.cuda.reset_peak_memory_stats()


# =========================================================
# 4. CARREGAR DINOv3 PELO HUGGING FACE
# =========================================================

MODEL_ID = (
    "facebook/"
    "dinov3-vits16-pretrain-lvd1689m"
)

print("\nCarregando DINOv3 ViT-S/16...")

dinov3_hf = AutoModel.from_pretrained(
    MODEL_ID
)

dinov3_hf = dinov3_hf.to(device)
dinov3_hf.eval()

print("DINOv3 carregado.")


# =========================================================
# 5. ADAPTADOR DINOv3 -> INSID3
# =========================================================

print("\nCriando adaptador DINOv3 -> INSID3...")

encoder = DINOv3HFAdapter(
    dinov3_hf
)

encoder = encoder.to(device)
encoder.eval()

print("Adaptador criado.")


# =========================================================
# 6. CRIAR INSID3
# =========================================================

print("\nCriando INSID3...")

model = INSID3(
    encoder=encoder,

    # Primeiro teste conservador para a RTX 4060 8 GB
    image_size=768,

    svd_components=500,
    tau=0.90,
    merge_threshold=0.2,

    # Sem CRF por enquanto
    mask_refiner="bilinear",

    resize_to_orig_size=True,

    device=device,
)

print("INSID3 criado.")


# =========================================================
# 7. REFERÊNCIA
# =========================================================

print("\nImagem de referência:")
print(REFERENCE_IMAGE)

print("\nMáscara:")
print(REFERENCE_MASK)

print("\nProcessando referência...")

model.set_reference(
    str(REFERENCE_IMAGE),
    str(REFERENCE_MASK),
)

print("Referência processada.")

# =========================================================
# DIAGNÓSTICO DA REFERÊNCIA DENTRO DO INSID3
# =========================================================

from utils.data import downsample_mask

print("\n--- DIAGNÓSTICO DA REFERÊNCIA ---")

print("Shape imagem de referência:")
print(model._ref_images.shape)

print("Shape máscara de referência:")
print(model._ref_masks.shape)

print(
    "Valores únicos da máscara:",
    torch.unique(model._ref_masks)
)

print(
    "Pixels positivos na máscara interna:",
    model._ref_masks.sum().item()
)

print(
    "Percentual positivo na máscara interna:",
    100.0
    * model._ref_masks.sum().item()
    / model._ref_masks.numel()
)

# ViT-S/16: para image_size=768, esperamos grade 48 x 48
h = model._ref_images.shape[-2] // 16
w = model._ref_images.shape[-1] // 16

mask_patch = downsample_mask(
    model._ref_masks.unsqueeze(1),
    h,
    w
)

print("\nGrade de patches:")
print(h, "x", w)

print(
    "Patches positivos:",
    mask_patch.sum().item()
)

print(
    "Total de patches:",
    mask_patch.numel()
)

print(
    "Percentual de patches positivos:",
    100.0
    * mask_patch.sum().item()
    / mask_patch.numel()
)

print(
    "Valores únicos após downsample:",
    torch.unique(mask_patch)
)
# =========================================================
# 8. IMAGEM-ALVO
# =========================================================

print("\nImagem-alvo:")
print(TARGET_IMAGE)

print("\nProcessando imagem-alvo...")

model.set_target(
    str(TARGET_IMAGE)
)

print("Imagem-alvo processada.")


# =========================================================
# 9. SEGMENTAÇÃO
# =========================================================

print("\nExecutando INSID3...")

with torch.inference_mode():
    prediction = model.segment()

# =========================================================
# DIAGNÓSTICO DA PREDIÇÃO
# =========================================================

total_pixels = prediction.numel()
pixels_true = prediction.sum().item()
pixels_false = total_pixels - pixels_true

percent_true = 100.0 * pixels_true / total_pixels

print("\n--- DIAGNÓSTICO DA PREDIÇÃO ---")
print("Pixels totais:", total_pixels)
print("Pixels TRUE:", pixels_true)
print("Pixels FALSE:", pixels_false)
print(f"Área marcada como alvo: {percent_true:.4f}%")
print("Valores únicos:", torch.unique(prediction))

# =========================================================
# 10. RESULTADO
# =========================================================

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)

print(
    "Tipo:",
    type(prediction)
)

print(
    "Shape:",
    prediction.shape
)

print(
    "dtype:",
    prediction.dtype
)

peak_memory = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)

print(
    f"VRAM máxima: {peak_memory:.2f} GB"
)


# =========================================================
# 11. PREPARAR MÁSCARA
# =========================================================

mask = prediction.detach().cpu()

mask = mask.squeeze()

mask = mask.numpy()


# =========================================================
# 12. CONVERTER PARA IMAGEM BINÁRIA
# =========================================================

# Dependendo da saída do INSID3, teremos valores
# booleanos ou numéricos.
#
# Tudo que for maior que zero será considerado alvo.

mask_binary = (
    mask > 0
).astype(np.uint8) * 255


# =========================================================
# 13. SALVAR
# =========================================================

Image.fromarray(
    mask_binary
).save(
    OUTPUT_MASK
)

print("\nMáscara salva em:")

print(
    OUTPUT_MASK
)

print("\nExperimento concluído.")

print("=" * 60)

# =========================================================
# 14. GERAR IMAGEM COM BOUNDING BOXES
# =========================================================

print("\nGerando imagem com bounding boxes...")

# Abrir a imagem-alvo original
image_bgr = cv2.imread(str(TARGET_IMAGE))

if image_bgr is None:
    raise RuntimeError(
        f"Não foi possível abrir a imagem-alvo: {TARGET_IMAGE}"
    )

# A máscara já está em 0 e 255.
# Garantimos aqui que seja uint8 para o OpenCV.
mask_cv = mask_binary.astype(np.uint8)

# Identificar regiões conectadas da máscara.
num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        mask_cv,
        connectivity=8
    )
)

# Caminho da nova imagem
OUTPUT_BOXES = (
    OUTPUT_DIR / "exp001_insid3_boxes.jpg"
)

# Área mínima para considerar uma região.
# Por enquanto usamos 1 para NÃO eliminar alvos pequenos.
MIN_AREA = 1

detections = 0

# label 0 = fundo, portanto começamos em 1
for label_id in range(1, num_labels):

    x = stats[
        label_id,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        label_id,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        label_id,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        label_id,
        cv2.CC_STAT_HEIGHT
    ]

    area = stats[
        label_id,
        cv2.CC_STAT_AREA
    ]

    # Não descartar pequenos alvos neste estágio
    if area < MIN_AREA:
        continue

    detections += 1

    # Desenhar bounding box
    cv2.rectangle(
        image_bgr,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

    # Número da região
    cv2.putText(
        image_bgr,
        str(detections),
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    print(
        f"Região {detections}: "
        f"x={x}, y={y}, "
        f"w={w}, h={h}, "
        f"area={area}"
    )


# =========================================================
# 15. SALVAR IMAGEM COM BOXES
# =========================================================

cv2.imwrite(
    str(OUTPUT_BOXES),
    image_bgr
)

print("\nNúmero de regiões detectadas:")
print(detections)

print("\nImagem com bounding boxes salva em:")
print(OUTPUT_BOXES)