import numpy as np
from ultralytics import YOLO

from draw import draw_detection, draw_fps, get_color


class Detector:
    """
    Encapsula o modelo YOLO e a lógica de inferência/desenho.

    Uso:
        detector = Detector('best.pt', thresh=0.5)
        detector.process(frame, show_fps=True, fps=42.0)
    """

    def __init__(self, model_path: str, thresh: float = 0.5):
        self.thresh = thresh

        self.model  = YOLO(model_path, task='detect')
        self.labels = self.model.names   # {0: 'Pikachu', 1: 'Charizard', ...}

        print(f'Modelo carregado. Classes: {list(self.labels.values())}')

    def process(self, frame, show_fps: bool = False, fps: float = 0.0) -> list[dict]:
        """
        Roda a inferência no frame, desenha os resultados e retorna as detecções.

        Retorna uma lista de dicts com:
            - class_id, class_name, conf, xmin, ymin, xmax, ymax

        Isso facilita usar os dados futuramente (ex: chamar uma API).
        """
        results    = self.model(frame, verbose=False)
        detections = results[0].boxes

        found = []

        for det in detections:
            conf = det.conf.item()
            if conf < self.thresh:
                continue

            xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().squeeze().astype(int)
            class_id   = int(det.cls.item())
            class_name = self.labels[class_id]
            color      = get_color(class_id)

            # Desenha no frame
            draw_detection(frame, xmin, ymin, xmax, ymax, class_name, conf, color)

            # Salva para retornar
            found.append({
                'class_id':   class_id,
                'class_name': class_name,
                'conf':       conf,
                'xmin': xmin, 'ymin': ymin,
                'xmax': xmax, 'ymax': ymax,
            })

            # Ponto de extensão: chamada de API
            # Exemplo futuro:
            # info = api.buscar_carta(class_name)
            # draw_card_info(frame, info, xmin, ymin)

        # HUD de FPS (somente para streams)
        if show_fps:
            draw_fps(frame, fps)

        return found
