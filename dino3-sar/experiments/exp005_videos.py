import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


# =========================================================
# 1. CAMINHOS DO PROJETO
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSID3_ROOT = PROJECT_ROOT / "external" / "INSID3"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INSID3_ROOT))

from models.insid3 import INSID3
from src.models.dinov3_hf_adapter import DINOv3HFAdapter


# =========================================================
# 2. ARQUIVOS
# =========================================================

REFERENCE_IMAGE = (
    PROJECT_ROOT
    / "samples"
    / "images"
    / "imagem_teste.jpg"
)

REFERENCE_MASK = (
    PROJECT_ROOT
    / "samples"
    / "images"
    / "mascara_teste.png"
)

INPUT_VIDEO = (
    PROJECT_ROOT
    / "samples"
    / "videos"
    / "video_teste.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "videos"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "exp005_video_real_boxes.mp4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 3. VERIFICAR ARQUIVOS
# =========================================================

for path in [
    REFERENCE_IMAGE,
    REFERENCE_MASK,
    INPUT_VIDEO,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {path}"
        )


# =========================================================
# 4. GPU
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA nao esta disponivel."
    )

device = "cuda"

print("=" * 70)
print("EXP005 - VIDEO - DINOv3 + INSID3")
print("=" * 70)

print("\nGPU:")
print(torch.cuda.get_device_name(0))

print(
    "CUDA PyTorch:",
    torch.version.cuda
)


# =========================================================
# 5. CARREGAR DINOV3
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
# 6. ADAPTADOR DINOV3 -> INSID3
# =========================================================

print("\nCriando adaptador DINOv3 -> INSID3...")

encoder = DINOv3HFAdapter(
    dinov3_hf
)

encoder = encoder.to(device)
encoder.eval()

print("Adaptador criado.")


# =========================================================
# 7. CRIAR INSID3
# =========================================================

print("\nCriando INSID3...")

model = INSID3(
    encoder=encoder,

    # Mesmo valor do experimento de imagem
    image_size=768,

    svd_components=500,

    # Valor que evitou o colapso K=1
    tau=0.90,

    merge_threshold=0.20,

    # Continuamos sem CRF
    mask_refiner="bilinear",

    resize_to_orig_size=True,

    device=device,
)

print("INSID3 criado.")


# =========================================================
# 8. CONFIGURAR REFERENCIA
# =========================================================

print("\nConfigurando referencia...")

print("Imagem:")
print(REFERENCE_IMAGE)

print("Mascara:")
print(REFERENCE_MASK)

model.set_reference(
    str(REFERENCE_IMAGE),
    str(REFERENCE_MASK),
)

print("Referencia processada.")

# Guardar a referencia preparada pelo INSID3.
# segment() chama reset_state() depois de cada inferencia,
# portanto preservamos estes tensores para reutiliza-los
# em todos os frames do video.

ref_images_video = model._ref_images
ref_masks_video = model._ref_masks

# =========================================================
# 9. ABRIR VIDEO
# =========================================================

print("\nAbrindo video:")

print(INPUT_VIDEO)

cap = cv2.VideoCapture(
    str(INPUT_VIDEO)
)

if not cap.isOpened():
    raise RuntimeError(
        f"Nao foi possivel abrir o video: {INPUT_VIDEO}"
    )


# =========================================================
# 10. INFORMACOES DO VIDEO
# =========================================================

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

if video_fps <= 0:
    video_fps = 30.0


print("\n--- INFORMACOES DO VIDEO ---")

print(
    "Resolucao:",
    f"{frame_width}x{frame_height}"
)

print(
    f"FPS original: {video_fps:.2f}"
)

print(
    "Total de frames:",
    total_frames
)

if total_frames > 0:
    duration = (
        total_frames
        / video_fps
    )

    print(
        f"Duracao aproximada: {duration:.2f} s"
    )


# =========================================================
# 11. VIDEO DE SAIDA
# =========================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    video_fps,
    (
        frame_width,
        frame_height
    ),
)

if not writer.isOpened():
    cap.release()

    raise RuntimeError(
        "Nao foi possivel criar o video de saida."
    )


# =========================================================
# 12. CONFIGURACOES DAS BOXES
# =========================================================

# Neste primeiro teste NÃO vamos eliminar regiões pequenas.
#
# Em SAR isso é importante porque um alvo pequeno pode ser
# justamente uma pessoa distante ocupando poucos pixels.

MIN_AREA = 1


# =========================================================
# 13. CONTADORES
# =========================================================

frame_index = 0

total_processing_time = 0.0

print("\n" + "=" * 70)
print("INICIANDO PROCESSAMENTO")
print("=" * 70)

print("\nPressione Q na janela para interromper.\n")


# =========================================================
# 14. LOOP DO VIDEO
# =========================================================

try:

    while True:

        success, frame_bgr = cap.read()

        if not success:
            break

        frame_index += 1

        start_time = time.perf_counter()


        # =================================================
        # 14.1 BGR -> RGB
        # =================================================

        # OpenCV lê imagens como BGR.
        # PIL / modelos de visão normalmente trabalham
        # com RGB.

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # 14.2 NUMPY -> PIL
        # =================================================

        frame_pil = Image.fromarray(
            frame_rgb
        )


        # =================================================
        # 14.3 DEFINIR FRAME COMO ALVO
        # =================================================

        model.set_target(
            frame_pil
        )

        # segment() apagaria a referencia ao final de cada frame.
        # Para video, chamamos diretamente predict_mask(),
        # reutilizando a referencia que guardamos anteriormente.

        with torch.inference_mode():
            
            prediction = model.predict_mask(
                ref_images_video,
                ref_masks_video,
                model._tgt_image
            )

        # =================================================
        # 14.5 CONVERTER MASCARA
        # =================================================

        mask = (
            prediction
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        mask_binary = (
            mask > 0
        ).astype(
            np.uint8
        ) * 255


        # =================================================
        # 14.6 GARANTIR MESMO TAMANHO DO FRAME
        # =================================================

        if (
            mask_binary.shape[0] != frame_height
            or
            mask_binary.shape[1] != frame_width
        ):

            mask_binary = cv2.resize(
                mask_binary,
                (
                    frame_width,
                    frame_height
                ),
                interpolation=cv2.INTER_NEAREST
            )


        # =================================================
        # 14.7 COMPONENTES CONECTADOS
        # =================================================

        (
            num_labels,
            labels,
            stats,
            centroids
        ) = cv2.connectedComponentsWithStats(
            mask_binary,
            connectivity=8
        )


        # =================================================
        # 14.8 DESENHAR BOXES
        # =================================================

        detections = 0

        for label_id in range(
            1,
            num_labels
        ):

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

            if area < MIN_AREA:
                continue

            detections += 1


            # Caixa ao redor da região
            cv2.rectangle(
                frame_bgr,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 0, 255),
                2
            )


            # Identificação simples da detecção
            label_text = (
                f"TARGET {detections}"
            )

            cv2.putText(
                frame_bgr,
                label_text,
                (
                    x,
                    max(
                        y - 7,
                        18
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )


        # =================================================
        # 14.9 TEMPO / FPS
        # =================================================

        processing_time = (
            time.perf_counter()
            - start_time
        )

        total_processing_time += (
            processing_time
        )

        if processing_time > 0:

            processing_fps = (
                1.0
                / processing_time
            )

        else:

            processing_fps = 0.0


        # =================================================
        # 14.10 INFORMACOES NA TELA
        # =================================================

        cv2.putText(
            frame_bgr,
            (
                f"Frame: "
                f"{frame_index}"
            ),
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame_bgr,
            (
                f"Deteccoes: "
                f"{detections}"
            ),
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame_bgr,
            (
                f"IA: "
                f"{processing_fps:.2f} FPS"
            ),
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame_bgr,
            (
                f"Latencia: "
                f"{processing_time * 1000:.0f} ms"
            ),
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # =================================================
        # 14.11 SALVAR FRAME NO VIDEO
        # =================================================

        writer.write(
            frame_bgr
        )

        # =================================================
        # 14.12 MOSTRAR AO VIVO
        # =================================================

        # Tempo atual do video, em segundos.
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Duracao total aproximada.
        total_time_sec = (
            total_frames / video_fps
            if video_fps > 0
            else 0
        )

        # Converter para MM:SS
        current_min = int(current_time_sec // 60)
        current_sec = int(current_time_sec % 60)

        total_min = int(total_time_sec // 60)
        total_sec = int(total_time_sec % 60)

        # Mostrar tempo atual no frame
        cv2.putText(
            frame_bgr,
            (
                f"Tempo: "
                f"{current_min:02d}:{current_sec:02d} / "
                f"{total_min:02d}:{total_sec:02d}"
            ),
            (15, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Mostrar controles
        cv2.putText(
            frame_bgr,
            "A:-30s  S:-10s  D:+10s  F:+30s  P:Pausa  Q:Sair",
            (15, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(
            "EXP005 - DINOv3 + INSID3",
            frame_bgr
        )


        # =================================================
        # 14.13 PROGRESSO NO TERMINAL
        # =================================================

        print(
            f"\rFrame "
            f"{frame_index}/{total_frames} | "
            f"Tempo: "
            f"{current_min:02d}:{current_sec:02d} | "
            f"Deteccoes: {detections} | "
            f"{processing_fps:.2f} FPS | "
            f"{processing_time * 1000:.0f} ms",
            end="",
            flush=True
        )


        # =================================================
        # 14.14 CONTROLES DO VIDEO
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        # -------------------------------------------------
        # Q = SAIR
        # -------------------------------------------------

        if key == ord("q"):

            print(
                "\n\nProcessamento interrompido pelo usuario."
            )

            break


        # -------------------------------------------------
        # A = VOLTAR 30 SEGUNDOS
        # -------------------------------------------------

        elif key == ord("a"):

            new_time = max(
                0.0,
                current_time_sec - 30.0
            )

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                new_time * 1000.0
            )

            frame_index = int(
                cap.get(cv2.CAP_PROP_POS_FRAMES)
            )

            print(
                f"\nVoltando para "
                f"{int(new_time // 60):02d}:"
                f"{int(new_time % 60):02d}"
            )


        # -------------------------------------------------
        # S = VOLTAR 10 SEGUNDOS
        # -------------------------------------------------

        elif key == ord("s"):

            new_time = max(
                0.0,
                current_time_sec - 10.0
            )

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                new_time * 1000.0
            )

            frame_index = int(
                cap.get(cv2.CAP_PROP_POS_FRAMES)
            )

            print(
                f"\nVoltando para "
                f"{int(new_time // 60):02d}:"
                f"{int(new_time % 60):02d}"
            )


        # -------------------------------------------------
        # D = AVANCAR 10 SEGUNDOS
        # -------------------------------------------------

        elif key == ord("d"):

            new_time = min(
                total_time_sec,
                current_time_sec + 10.0
            )

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                new_time * 1000.0
            )

            frame_index = int(
                cap.get(cv2.CAP_PROP_POS_FRAMES)
            )

            print(
                f"\nAvancando para "
                f"{int(new_time // 60):02d}:"
                f"{int(new_time % 60):02d}"
            )


        # -------------------------------------------------
        # F = AVANCAR 30 SEGUNDOS
        # -------------------------------------------------

        elif key == ord("f"):

            new_time = min(
                total_time_sec,
                current_time_sec + 30.0
            )

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                new_time * 1000.0
            )

            frame_index = int(
                cap.get(cv2.CAP_PROP_POS_FRAMES)
            )

            print(
                f"\nAvancando para "
                f"{int(new_time // 60):02d}:"
                f"{int(new_time % 60):02d}"
            )


        # -------------------------------------------------
        # P = PAUSAR / CONTINUAR
        # -------------------------------------------------

        elif key == ord("p"):

            print(
                "\nVideo pausado. "
                "P = continuar | "
                "A/S/D/F = navegar | "
                "Q = sair"
            )

            while True:

                pause_key = cv2.waitKey(0) & 0xFF

                # P = continuar
                if pause_key == ord("p"):
                    print("Continuando...")
                    break

                # Q = sair
                elif pause_key == ord("q"):
                    print(
                        "\nProcessamento interrompido "
                        "pelo usuario."
                    )

                    raise KeyboardInterrupt

                # A = -30 s
                elif pause_key == ord("a"):

                    current_time_sec = (
                        cap.get(
                            cv2.CAP_PROP_POS_MSEC
                        ) / 1000.0
                    )

                    new_time = max(
                        0.0,
                        current_time_sec - 30.0
                    )

                    cap.set(
                        cv2.CAP_PROP_POS_MSEC,
                        new_time * 1000.0
                    )

                    frame_index = int(
                        cap.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    )

                    print(
                        f"Posicao: "
                        f"{int(new_time // 60):02d}:"
                        f"{int(new_time % 60):02d}"
                    )

                    break

                # S = -10 s
                elif pause_key == ord("s"):

                    current_time_sec = (
                        cap.get(
                            cv2.CAP_PROP_POS_MSEC
                        ) / 1000.0
                    )

                    new_time = max(
                        0.0,
                        current_time_sec - 10.0
                    )

                    cap.set(
                        cv2.CAP_PROP_POS_MSEC,
                        new_time * 1000.0
                    )

                    frame_index = int(
                        cap.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    )

                    print(
                        f"Posicao: "
                        f"{int(new_time // 60):02d}:"
                        f"{int(new_time % 60):02d}"
                    )

                    break

                # D = +10 s
                elif pause_key == ord("d"):

                    current_time_sec = (
                        cap.get(
                            cv2.CAP_PROP_POS_MSEC
                        ) / 1000.0
                    )

                    new_time = min(
                        total_time_sec,
                        current_time_sec + 10.0
                    )

                    cap.set(
                        cv2.CAP_PROP_POS_MSEC,
                        new_time * 1000.0
                    )

                    frame_index = int(
                        cap.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    )

                    print(
                        f"Posicao: "
                        f"{int(new_time // 60):02d}:"
                        f"{int(new_time % 60):02d}"
                    )

                    break

                # F = +30 s
                elif pause_key == ord("f"):

                    current_time_sec = (
                        cap.get(
                            cv2.CAP_PROP_POS_MSEC
                        ) / 1000.0
                    )

                    new_time = min(
                        total_time_sec,
                        current_time_sec + 30.0
                    )

                    cap.set(
                        cv2.CAP_PROP_POS_MSEC,
                        new_time * 1000.0
                    )

                    frame_index = int(
                        cap.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    )

                    print(
                        f"Posicao: "
                        f"{int(new_time // 60):02d}:"
                        f"{int(new_time % 60):02d}"
                    )

                    break


# =========================================================
# 15. LIBERAR RECURSOS
# =========================================================

finally:

    cap.release()

    writer.release()

    cv2.destroyAllWindows()


# =========================================================
# 16. ESTATISTICAS FINAIS
# =========================================================

print("\n")

print("=" * 70)
print("RESULTADO EXP005")
print("=" * 70)

print(
    "Frames processados:",
    frame_index
)

print(
    f"FPS original do video: "
    f"{video_fps:.2f}"
)

if frame_index > 0:

    mean_time = (
        total_processing_time
        / frame_index
    )

    mean_fps = (
        1.0
        / mean_time
        if mean_time > 0
        else 0.0
    )

    print(
        f"Latencia media: "
        f"{mean_time * 1000:.2f} ms/frame"
    )

    print(
        f"FPS medio da IA: "
        f"{mean_fps:.2f}"
    )

    realtime_ratio = (
        mean_fps
        / video_fps
    )

    print(
        f"Razao IA/video: "
        f"{realtime_ratio:.2f}x"
    )

    if mean_fps >= video_fps:

        print(
            "Desempenho: processamento >= tempo real."
        )

    else:

        print(
            "Desempenho: processamento abaixo do FPS original."
        )


print("\nVideo processado salvo em:")

print(
    OUTPUT_VIDEO
)

print("=" * 70)