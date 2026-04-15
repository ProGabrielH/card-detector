import os
import sys
import argparse
import time

import cv2
import numpy as np

from source   import ImageSource
from detector import Detector



# Argumentos de linha de comando
parser = argparse.ArgumentParser(description='Detecção de cartas Pokémon com YOLO')
parser.add_argument('--model',      required=True,  help='Caminho para o modelo .pt')
parser.add_argument('--source',     required=True,  help='Imagem, pasta, vídeo ou "usb0"')
parser.add_argument('--thresh',     default=0.5,    type=float, help='Confiança mínima (padrão: 0.5)')
parser.add_argument('--resolution', default=None,   help='Resolução LARGURAxALTURA (ex: 1280x720)')
parser.add_argument('--record',     action='store_true', help='Grava resultado em demo.avi (requer --resolution)')
args = parser.parse_args()


# Validações
if not os.path.exists(args.model):
    print(f'ERRO: Modelo não encontrado em "{args.model}".')
    sys.exit(1)

if args.record and args.resolution is None:
    print('ERRO: --record exige --resolution.')
    sys.exit(1)


# Resolução
resW = resH = None
if args.resolution:
    resW, resH = map(int, args.resolution.split('x'))


# Gravação
recorder = None
if args.record:
    recorder = cv2.VideoWriter(
        'demo.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH),
    )


# Inicializa fonte e detector
detector = Detector(args.model, thresh=args.thresh)

fps_buffer      = []
FPS_BUFFER_SIZE = 60
avg_fps         = 0.0


# Loop principal
WINDOW_NAME = 'Detecção de Cartas Pokémon'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
if resW and resH:
    cv2.resizeWindow(WINDOW_NAME, resW, resH)
    
with ImageSource(args.source, width=resW, height=resH) as src:

    for frame in src:

        t_start = time.perf_counter()

        # Redimensiona se necessário
        if resW and resH:
            frame = cv2.resize(frame, (resW, resH))

        # Inferência + desenho
        detections = detector.process(frame,
                                      show_fps=src.is_stream,
                                      fps=avg_fps)

        # Exibe o frame
        cv2.imshow('Detecção de Cartas Pokémon', frame)

        if recorder:
            recorder.write(frame)

        # Teclas: imagens esperam input; streams passam a cada 1 ms
        wait_ms = 0 if not src.is_stream else 1
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):       # Sair
            break
        elif key == ord('s'):     # Pausar
            cv2.waitKey(0)
        elif key == ord('p'):     # Salvar
            cv2.imwrite('captura.png', frame)
            print('Captura salva em captura.png')

        # FPS médio
        elapsed = time.perf_counter() - t_start
        fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
        if len(fps_buffer) > FPS_BUFFER_SIZE:
            fps_buffer.pop(0)
        avg_fps = float(np.mean(fps_buffer))


# Limpeza
print(f'FPS médio: {avg_fps:.2f}')

if recorder:
    recorder.release()

cv2.destroyAllWindows()
