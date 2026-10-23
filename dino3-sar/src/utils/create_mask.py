import cv2
import numpy as np
from pathlib import Path

IMAGE_PATH = Path("samples/images/imagem_teste.jpg")
MASK_PATH = Path("samples/images/mascara_teste.png")

image_original = cv2.imread(str(IMAGE_PATH))

if image_original is None:
    raise FileNotFoundError(f"Não foi possível abrir: {IMAGE_PATH}")

height, width = image_original.shape[:2]

# Máscara final, na resolução ORIGINAL
mask = np.zeros((height, width), dtype=np.uint8)

# Redimensionamento apenas para visualização
MAX_WIDTH = 1400
scale = min(1.0, MAX_WIDTH / width)

display_width = int(width * scale)
display_height = int(height * scale)

display_image = cv2.resize(
    image_original,
    (display_width, display_height)
)

# Lista de objetos já confirmados
regions = []

# Região que está sendo desenhada
current_points = []
drawing = False


def mouse_callback(event, x, y, flags, param):
    global drawing, current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_points.append((x, y))


window_name = "Criar mascara - DINO3-SAR"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("\nINSTRUÇÕES")
print("--------------------------------------")
print("Mouse esquerdo = desenhar objeto")
print("ENTER           = confirmar objeto")
print("R               = apagar desenho atual")
print("Z               = desfazer último objeto")
print("S               = salvar máscara")
print("ESC             = sair")
print("--------------------------------------")


while True:

    preview = display_image.copy()

    # Mostrar regiões já confirmadas
    overlay = preview.copy()

    for region in regions:
        pts = np.array(region, dtype=np.int32)

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 255, 0)
        )

        cv2.polylines(
            preview,
            [pts],
            True,
            (0, 255, 0),
            2
        )

    # Transparência para enxergar a imagem embaixo
    preview = cv2.addWeighted(
        overlay,
        0.25,
        preview,
        0.75,
        0
    )

    # Mostrar região sendo desenhada
    if len(current_points) > 1:

        pts = np.array(current_points, dtype=np.int32)

        cv2.polylines(
            preview,
            [pts],
            False,
            (0, 0, 255),
            2
        )

    cv2.imshow(window_name, preview)

    key = cv2.waitKey(20) & 0xFF

    # ENTER = confirmar objeto
    if key == 13:

        if len(current_points) >= 3:
            regions.append(current_points.copy())
            current_points = []

            print(
                f"Objeto confirmado. Total: {len(regions)}"
            )

    # R = apagar desenho atual
    elif key == ord("r"):

        current_points = []
        print("Desenho atual apagado.")

    # Z = desfazer último objeto
    elif key == ord("z"):

        if regions:
            regions.pop()

            print(
                f"Último objeto removido. Total: {len(regions)}"
            )

    # S = salvar
    elif key == ord("s"):

        if not regions:
            print("Nenhum objeto confirmado.")
            continue

        mask[:] = 0

        for region in regions:

            pts_display = np.array(
                region,
                dtype=np.float32
            )

            # Voltar para coordenadas da imagem original
            pts_original = pts_display / scale
            pts_original = pts_original.astype(np.int32)

            cv2.fillPoly(
                mask,
                [pts_original],
                255
            )

        cv2.imwrite(
            str(MASK_PATH),
            mask
        )

        print("\nMáscara salva!")
        print(f"Arquivo: {MASK_PATH}")
        print(f"Resolução: {width} x {height}")
        print(f"Objetos marcados: {len(regions)}")

        break

    elif key == 27:
        print("Operação cancelada.")
        break


cv2.destroyAllWindows()