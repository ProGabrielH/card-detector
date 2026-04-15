import cv2
import numpy as np


# Paleta com 25 cores distintas (uma por classe)
COLORS = [
    ( 68,148,228), (228, 68, 68), ( 68,228,148), (228,168, 68), (148, 68,228),
    ( 96,202,231), (231, 96,137), ( 96,231,162), (231,196, 96), (162, 96,231),
    (164,120, 87), ( 87,164,120), (120, 87,164), (164,164, 87), ( 87,120,164),
    (178,182,133), (133,178,182), (182,133,178), (178,133,182), (133,182,178),
    ( 40,180,100), (180,100, 40), (100, 40,180), (180,180, 40), ( 40,100,180),
]


def get_color(class_id: int) -> tuple:
    """Retorna uma cor para o class_id dado."""
    return COLORS[class_id % len(COLORS)]


def draw_detection(frame, xmin: int, ymin: int, xmax: int, ymax: int,
                   class_name: str, conf: float, color: tuple) -> None:
    """
    Desenha a bounding box e o label (nome + confiança) no frame.
    Modifica o frame in-place.
    """
    # Bounding box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    # Texto: "NomeDaClasse  xx.x%"
    text = f'{class_name}  {conf * 100:.1f}%'

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Garante que o label não saia pelo topo da janela
    label_y = max(ymin, th + 10)

    # Fundo colorido para o texto
    cv2.rectangle(frame,
                  (xmin, label_y - th - 8),
                  (xmin + tw, label_y + baseline - 8),
                  color, cv2.FILLED)

    # Texto em preto sobre o fundo
    cv2.putText(frame, text, (xmin, label_y - 5),
                font, font_scale, (0, 0, 0), thickness)


def draw_fps(frame, fps: float) -> None:
    """Exibe o FPS médio no canto superior esquerdo."""
    cv2.putText(frame, f'FPS: {fps:.1f}',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
