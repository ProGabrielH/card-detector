import os
import sys
import glob

import cv2


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
VID_EXTS = {'.avi', '.mov', '.mp4', '.mkv', '.wmv'}


class ImageSource:
    """
    Abstrai qualquer fonte de frames: imagem, pasta, vídeo ou webcam USB.

    Uso:
        with ImageSource(args.source, width=1280, height=720) as src:
            for frame in src:
                ...
    """

    def __init__(self, source: str, width: int = None, height: int = None):
        self.source = source
        self.width  = width
        self.height = height

        self._type      = None
        self._cap       = None       # cv2.VideoCapture para vídeo/usb
        self._imgs      = []         # lista de paths para imagem/pasta
        self._img_index = 0

        self._detect_type()
        self._init_source()

    # Detecção do tipo de fonte
    def _detect_type(self):
        s = self.source

        if os.path.isdir(s):
            self._type = 'folder'

        elif os.path.isfile(s):
            ext = os.path.splitext(s)[1]
            if ext in IMG_EXTS:
                self._type = 'image'
            elif ext in VID_EXTS:
                self._type = 'video'
            else:
                print(f'ERRO: Extensão "{ext}" não suportada.')
                sys.exit(1)

        elif s.startswith('usb'):
            self._type = 'usb'

        else:
            print(f'ERRO: Fonte "{s}" não reconhecida.')
            sys.exit(1)

    # Inicialização
    def _init_source(self):
        if self._type == 'image':
            self._imgs = [self.source]

        elif self._type == 'folder':
            self._imgs = [f for f in glob.glob(self.source + '/*')
                          if os.path.splitext(f)[1] in IMG_EXTS]
            if not self._imgs:
                print('Nenhuma imagem encontrada na pasta.')
                sys.exit(1)
            print(f'{len(self._imgs)} imagem(ns) encontrada(s).')

        elif self._type in ('video', 'usb'):
            arg = self.source if self._type == 'video' else int(self.source[3:])
            self._cap = cv2.VideoCapture(arg)

            if self.width and self.height:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    # ── Propriedades públicas ───────────────────────────────────────────────
    @property
    def source_type(self) -> str:
        return self._type

    @property
    def is_stream(self) -> bool:
        """True quando a fonte é contínua (vídeo ou câmera)."""
        return self._type in ('video', 'usb')

    # ── Iterador ────────────────────────────────────────────────────────────
    def __iter__(self):
        return self

    def __next__(self):
        if self._type in ('image', 'folder'):
            if self._img_index >= len(self._imgs):
                raise StopIteration
            frame = cv2.imread(self._imgs[self._img_index])
            self._img_index += 1
            return frame

        elif self._type == 'video':
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
            return frame

        elif self._type == 'usb':
            ret, frame = self._cap.read()
            if not ret or frame is None:
                raise StopIteration
            return frame

    # ── Context manager (garante liberação do cap) ──────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def release(self):
        if self._cap:
            self._cap.release()
