import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading
from PIL import Image, ImageDraw, ImageFont

# =========================
# CONFIGURACIÓN
# =========================
FRASES = [
    "Me duele",
    "¿Qué está pasando?",
    "Quiero ver a mi familia",
    "Estoy bien",
    "Tengo miedo",
    "Gracias por estar aquí"
]

COOLDOWN_CLIC    = 1.5   # segundos mínimos entre selecciones
SEGUNDOS_HOVER   = 3.0   # segundos que hay que mantener el cursor para seleccionar

# ── Sensibilidad del cursor ─────────────────────────────────────────────────
ZONA_ACTIVA = 0.5
# ────────────────────────────────────────────────────────────────────────────

FONT_PATH = "arial.ttf"

# ── Zona de detección de la mano en la cámara ───────────────────────────────
# Fracción vertical de la cámara que se usa para detectar la mano.
# 0.0 = arriba de la imagen,  1.0 = abajo de la imagen.
# Rango recomendado: 0.4–1.0  (parte inferior = pecho/abdomen del usuario)
ZONA_CAM_Y_MIN = 0.4   # la mano debe estar por debajo del 40% superior
ZONA_CAM_Y_MAX = 1.0   # hasta el borde inferior de la cámara
# ────────────────────────────────────────────────────────────────────────────

# ── Suavizado del cursor ─────────────────────────────────────────────────────
# EMA (media exponencial): cuánto "peso" tiene la posición nueva cada frame.
# Más bajo = más suave y lento  |  Más alto = más rápido pero más temblor
# Rango recomendado: 0.10 (muy suave) – 0.30 (ágil)
CURSOR_SUAVIZADO = 0.20

# Umbral mínimo de movimiento en píxeles.
# Si la mano se mueve menos de esto el cursor no se actualiza
# (filtra el temblor fino del pulso y del sensor).
CURSOR_UMBRAL_PX = 3
# ────────────────────────────────────────────────────────────────────────────

# Descripciones de la escala de dolor
DESC_DOLOR = [
    "Sin dolor",
    "Muy leve",
    "Leve",
    "Moderado",
    "Moderado+",
    "Moderado++",
    "Fuerte",
    "Fuerte+",
    "Muy fuerte",
    "Insoportable",
]

# =========================
# MEDIAPIPE
# =========================
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# =========================
# VOZ (hilo separado)
# =========================
_voz_lock = threading.Lock()

def _hablar(texto):
    with _voz_lock:
        engine = pyttsx3.init()
        engine.setProperty('rate', 100)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        for v in voices:
            if 'spanish' in v.name.lower() or 'es' in v.id.lower():
                engine.setProperty('voice', v.id)
                break
        engine.say(texto)
        engine.runAndWait()

def texto_a_voz(texto):
    threading.Thread(target=_hablar, args=(texto,), daemon=True).start()

# =========================
# FUENTE PIL
# =========================
def fuente(size):
    intentos = [
        FONT_PATH,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for ruta in intentos:
        try:
            return ImageFont.truetype(ruta, size)
        except Exception:
            pass
    return ImageFont.load_default()

def texto_centrado(draw, texto, cx, cy, size, color):
    f = fuente(size)
    lineas = texto.split("\n")
    total_h = sum(f.getbbox(l)[3] - f.getbbox(l)[1] + 4 for l in lineas)
    y = cy - total_h // 2
    for linea in lineas:
        bbox = f.getbbox(linea)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, y), linea, font=f, fill=color)
        y += bbox[3] - bbox[1] + 4

def texto_simple(draw, texto, x, y, size, color):
    f = fuente(size)
    draw.text((x, y), texto, font=f, fill=color)

# =========================
# SILUETA DEL CUERPO
# =========================
def crear_silueta(w=130, h=385):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    cx  = w // 2
    fill   = (190, 210, 235, 230)
    borde  = (100, 140, 200, 255)

    d.ellipse([cx-26, 5, cx+26, 63],            fill=fill, outline=borde, width=2)  # Cabeza
    d.rectangle([cx-11, 62, cx+11, 82],          fill=fill, outline=borde, width=2)  # Cuello
    d.rounded_rectangle([cx-44, 80, cx+44, 198], radius=10, fill=fill, outline=borde, width=2)  # Torso
    d.rounded_rectangle([cx-40, 194, cx+40, 228],radius=8,  fill=fill, outline=borde, width=2)  # Cadera
    d.rounded_rectangle([cx-66, 86, cx-44, 192], radius=9,  fill=fill, outline=borde, width=2)  # Brazo izq
    d.rounded_rectangle([cx+44, 86, cx+66, 192], radius=9,  fill=fill, outline=borde, width=2)  # Brazo der
    d.rounded_rectangle([cx-38, 226, cx-8, 380], radius=9,  fill=fill, outline=borde, width=2)  # Pierna izq
    d.rounded_rectangle([cx+8,  226, cx+38, 380],radius=9,  fill=fill, outline=borde, width=2)  # Pierna der

    return np.array(img)  # RGBA

SILUETA_BASE = crear_silueta()
_SIL_H, _SIL_W = SILUETA_BASE.shape[:2]

# =========================
# COLORES (RGB para PIL)
# =========================
COLORES_DOLOR_RGB = [
    (50,  210,  80),
    (90,  220,  55),
    (130, 225,  35),
    (170, 225,  25),
    (205, 215,  20),
    (225, 175,  20),
    (225, 125,  20),
    (215,  75,  20),
    (200,  38,  20),
    (175,   0,   0),
]

# =========================
# DIMENSIONES DE VENTANA
# =========================
PANEL_FRASES_W = 750
PANEL_FRASES_H = 620
PANEL_CUERPO_W = 220
PANEL_CUERPO_H = 620
PANEL_DOLOR_W  = 280
PANEL_DOLOR_H  = 620
HEADER_H       = 55
FOOTER_H       = 60
TOTAL_W = PANEL_FRASES_W + PANEL_CUERPO_W + PANEL_DOLOR_W
TOTAL_H = PANEL_FRASES_H

# =========================
# ESTADO GLOBAL
# =========================
class Estado:
    def __init__(self):
        self.historial         = []
        self.ultimo_clic       = 0.0
        self.nivel_dolor       = None

        # hover con tiempo
        self.hover_frase       = -1
        self.hover_frase_ini   = 0.0
        self.hover_dolor       = -1
        self.hover_dolor_ini   = 0.0

        self.cursor_x          = PANEL_FRASES_W // 2
        self.cursor_y          = PANEL_FRASES_H // 2
        # posición suavizada (float para mayor precisión en la interpolación)
        self.cursor_sx         = float(PANEL_FRASES_W // 2)
        self.cursor_sy         = float(PANEL_FRASES_H // 2)

E = Estado()

# =========================
# HELPERS DE HOVER
# =========================
def actualizar_hover_frase(idx_nuevo):
    ahora = time.time()
    if idx_nuevo != E.hover_frase:
        E.hover_frase     = idx_nuevo
        E.hover_frase_ini = ahora if idx_nuevo >= 0 else 0.0

def actualizar_hover_dolor(idx_nuevo):
    ahora = time.time()
    if idx_nuevo != E.hover_dolor:
        E.hover_dolor     = idx_nuevo
        E.hover_dolor_ini = ahora if idx_nuevo >= 0 else 0.0

def progreso_hover(ini):
    """0.0 → 1.0 según cuánto tiempo lleva en hover."""
    if ini == 0.0:
        return 0.0
    return min((time.time() - ini) / SEGUNDOS_HOVER, 1.0)

def check_seleccion():
    """Dispara selecciones cuando el progreso llega a 1.0."""
    ahora = time.time()
    if ahora - E.ultimo_clic < COOLDOWN_CLIC:
        return

    # ── Frase
    if E.hover_frase >= 0 and progreso_hover(E.hover_frase_ini) >= 1.0:
        frase = FRASES[E.hover_frase]
        extras = []
        if E.nivel_dolor is not None:
            extras.append(f"con dolor nivel {E.nivel_dolor}")
        msg = frase + (" " + ", ".join(extras) if extras else "")
        E.historial.append(frase)
        E.ultimo_clic  = ahora
        E.hover_frase_ini = ahora
        texto_a_voz(msg)
        return

    # ── Dolor
    if E.hover_dolor >= 0 and progreso_hover(E.hover_dolor_ini) >= 1.0:
        E.nivel_dolor    = E.hover_dolor + 1
        E.ultimo_clic    = ahora
        E.hover_dolor_ini = ahora
        texto_a_voz(f"Dolor nivel {E.nivel_dolor}")
        return

# =========================
# PANEL DE FRASES
# =========================
def construir_panel_frases(cursor_x, cursor_y):
    COLS, ROWS = 3, 2
    PAD = 14
    cell_w = (PANEL_FRASES_W - PAD * (COLS + 1)) // COLS
    cell_h = (PANEL_FRASES_H - HEADER_H - FOOTER_H - PAD * (ROWS + 1)) // ROWS

    img  = Image.new("RGB", (PANEL_FRASES_W, PANEL_FRASES_H), (238, 240, 245))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (PANEL_FRASES_W, HEADER_H)], fill=(30, 30, 50))
    texto_centrado(draw, "Mantén el cursor 3 seg para seleccionar",
                   PANEL_FRASES_W // 2, HEADER_H // 2, 20, (200, 220, 255))

    rects = []
    ahora = time.time()

    for i, frase in enumerate(FRASES):
        col = i % COLS
        row = i // COLS
        x1 = PAD + col * (cell_w + PAD)
        y1 = HEADER_H + PAD + row * (cell_h + PAD)
        x2, y2 = x1 + cell_w, y1 + cell_h
        rects.append((x1, y1, x2, y2))

        en_hover = (E.hover_frase == i)
        prog     = progreso_hover(E.hover_frase_ini) if en_hover else 0.0

        bg  = (0, 110, 210) if en_hover else (255, 255, 255)
        fg  = (255, 255, 255) if en_hover else (30, 30, 50)
        brd = (0, 80, 170)   if en_hover else (180, 185, 200)

        draw.rectangle([(x1+4, y1+4), (x2+4, y2+4)], fill=(200, 202, 210))
        draw.rectangle([(x1, y1), (x2, y2)], fill=bg, outline=brd, width=3)

        # Barra de progreso (borde inferior de la celda)
        if prog > 0:
            bw = int((x2 - x1) * prog)
            color_prog = (0, 230, 130) if prog < 1.0 else (0, 255, 80)
            draw.rectangle([(x1, y2-6), (x1+bw, y2)], fill=color_prog)

        texto_centrado(draw, frase, (x1+x2)//2, (y1+y2)//2, 26, fg)
        texto_simple(draw, str(i+1), x1+10, y1+8, 16,
                     (180, 200, 240) if en_hover else (160, 165, 175))

    # Cursor
    if 0 <= cursor_x < PANEL_FRASES_W and 0 <= cursor_y < PANEL_FRASES_H:
        draw.ellipse([(cursor_x-12, cursor_y-12), (cursor_x+12, cursor_y+12)],
                     fill=(0, 200, 100), outline=(0, 100, 50), width=2)

    # Footer
    footer_y = PANEL_FRASES_H - FOOTER_H
    draw.rectangle([(0, footer_y), (PANEL_FRASES_W, PANEL_FRASES_H)], fill=(215, 218, 225))
    hist_txt = "Seleccionadas: " + (" | ".join(E.historial[-4:]) if E.historial else "(ninguna)")
    texto_simple(draw, hist_txt, 14, footer_y+10, 22, (0, 100, 30))
    texto_simple(draw, "ESC: salir   C: limpiar", PANEL_FRASES_W-230, footer_y+36, 17, (120,120,130))

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr, rects

# =========================
# PANEL DEL CUERPO HUMANO (solo visual, decorativo)
# =========================
def construir_panel_cuerpo(cursor_x_abs, cursor_y_abs, offset_x):
    img  = Image.new("RGB", (PANEL_CUERPO_W, PANEL_CUERPO_H), (230, 235, 242))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (PANEL_CUERPO_W, HEADER_H)], fill=(30, 50, 30))
    texto_centrado(draw, "Cuerpo", PANEL_CUERPO_W//2, HEADER_H//2, 20, (200, 240, 200))

    # Posición de la silueta centrada
    sil_ox = (PANEL_CUERPO_W - _SIL_W) // 2
    sil_oy = HEADER_H + 15

    # Pegar silueta (solo la imagen base, sin zonas)
    pil_sil = Image.fromarray(SILUETA_BASE, "RGBA")
    fondo   = img.convert("RGBA")
    fondo.paste(pil_sil, (sil_ox, sil_oy), pil_sil)
    img = fondo.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Cursor local
    cur_local_x = cursor_x_abs - offset_x
    cur_local_y = cursor_y_abs
    if 0 <= cur_local_x < PANEL_CUERPO_W and 0 <= cur_local_y < PANEL_CUERPO_H:
        draw.ellipse([(cur_local_x-12, cur_local_y-12),
                      (cur_local_x+12, cur_local_y+12)],
                     fill=(0, 200, 100), outline=(0, 100, 50), width=2)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr

# =========================
# PANEL ESCALA DE DOLOR
# =========================
def construir_panel_dolor(cursor_x_abs, cursor_y_abs, offset_x):
    img  = Image.new("RGB", (PANEL_DOLOR_W, PANEL_DOLOR_H), (235, 230, 242))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (PANEL_DOLOR_W, HEADER_H)], fill=(50, 30, 50))
    texto_centrado(draw, "Dolor", PANEL_DOLOR_W//2, HEADER_H//2, 20, (240, 210, 255))

    PAD  = 10
    CH   = (PANEL_DOLOR_H - HEADER_H - FOOTER_H - PAD * 11) // 10
    CW   = PANEL_DOLOR_W - PAD * 2

    cur_local_x = cursor_x_abs - offset_x
    cur_local_y = cursor_y_abs

    rects_dolor = []
    for i in range(10):
        rx = PAD
        ry = HEADER_H + PAD + i * (CH + PAD)
        rects_dolor.append((rx, ry, CW, CH))

    # Detectar hover
    hover_nuevo = -1
    for i, (rx, ry, rw, rh) in enumerate(rects_dolor):
        if rx <= cur_local_x <= rx+rw and ry <= cur_local_y <= ry+rh:
            hover_nuevo = i
            break
    actualizar_hover_dolor(hover_nuevo)

    for i, (rx, ry, rw, rh) in enumerate(rects_dolor):
        nivel    = i + 1
        cr       = COLORES_DOLOR_RGB[i]
        en_hover = (E.hover_dolor == i)
        es_sel   = (E.nivel_dolor == nivel)
        prog     = progreso_hover(E.hover_dolor_ini) if en_hover else 0.0

        if es_sel:
            bg_cell = cr
            txt_col = (20, 20, 20)
        elif en_hover:
            bg_cell = tuple(int(c * 0.55) for c in cr)
            txt_col = (255, 255, 255)
        else:
            bg_cell = tuple(int(c * 0.22) for c in cr)
            txt_col = (220, 220, 220)

        draw.rectangle([(rx, ry), (rx+rw, ry+rh)],
                        fill=bg_cell, outline=cr, width=2)

        # Barra de progreso
        if prog > 0:
            bw = int(rw * prog)
            draw.rectangle([(rx, ry+rh-5), (rx+bw, ry+rh)], fill=(0, 240, 140))

        texto_centrado(draw, f"{nivel}  –  {DESC_DOLOR[i]}",
                       rx + rw//2, ry + rh//2, 15, txt_col)

    # Nivel seleccionado
    if E.nivel_dolor is not None:
        texto_centrado(draw, f"✓ Nivel {E.nivel_dolor}",
                       PANEL_DOLOR_W//2,
                       PANEL_DOLOR_H - FOOTER_H + 22, 16,
                       COLORES_DOLOR_RGB[E.nivel_dolor - 1])

    # Cursor local
    if 0 <= cur_local_x < PANEL_DOLOR_W and 0 <= cur_local_y < PANEL_DOLOR_H:
        draw.ellipse([(cur_local_x-10, cur_local_y-10),
                      (cur_local_x+10, cur_local_y+10)],
                     fill=(0, 200, 100), outline=(0, 100, 50), width=2)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr

# =========================
# BUCLE PRINCIPAL
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as hands:

    cursor_x = PANEL_FRASES_W // 2
    cursor_y = PANEL_FRASES_H // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w, _   = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)
        display   = frame.copy()

        gesto = "sin mano"

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark

            mp_drawing.draw_landmarks(
                display,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

            cam_x = lm[8].x
            cam_y = lm[8].y

            # ── Dibujar zona de detección en la cámara (franja inferior) ──
            zona_y1_px = int(ZONA_CAM_Y_MIN * h)
            cv2.rectangle(display, (0, zona_y1_px), (w, h), (0, 180, 255), 2)
            cv2.putText(display, "zona activa", (6, zona_y1_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

            # ── Mapear posición horizontal normal (ZONA_ACTIVA)
            margen  = (1.0 - ZONA_ACTIVA) / 2
            cx_norm = max(0.0, min(1.0, (cam_x - margen) / ZONA_ACTIVA))

            # ── Mapear posición vertical SOLO dentro de la franja inferior
            rango_y = ZONA_CAM_Y_MAX - ZONA_CAM_Y_MIN
            cy_norm = max(0.0, min(1.0, (cam_y - ZONA_CAM_Y_MIN) / rango_y))

            # Posición cruda (sin suavizar)
            raw_x = cx_norm * TOTAL_W
            raw_y = cy_norm * TOTAL_H

            # Umbral: ignorar movimientos menores a CURSOR_UMBRAL_PX píxeles
            dx = raw_x - E.cursor_sx
            dy = raw_y - E.cursor_sy
            if (dx*dx + dy*dy) > CURSOR_UMBRAL_PX ** 2:
                # EMA: mezcla gradual entre posición actual y nueva
                E.cursor_sx += CURSOR_SUAVIZADO * dx
                E.cursor_sy += CURSOR_SUAVIZADO * dy

            cursor_x = int(E.cursor_sx)
            cursor_y = int(E.cursor_sy)

            # Punto del índice en la cámara
            cx_cam = int(cam_x * w)
            cy_cam = int(cam_y * h)
            cv2.circle(display, (cx_cam, cy_cam), 12, (0, 200, 100), -1)
            cv2.circle(display, (cx_cam, cy_cam), 14, (0, 100, 50), 2)

            gesto = "apuntando"

            # Hover en panel de frases
            if cursor_x < PANEL_FRASES_W:
                _, rects = construir_panel_frases(cursor_x, cursor_y)
                hover_nuevo = -1
                for i, (x1, y1, x2, y2) in enumerate(rects):
                    if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                        hover_nuevo = i
                        break
                actualizar_hover_frase(hover_nuevo)
                actualizar_hover_dolor(-1)
            else:
                actualizar_hover_frase(-1)
                # El panel de dolor calcula su hover internamente

        else:
            actualizar_hover_frase(-1)
            actualizar_hover_dolor(-1)
            gesto = "sin mano"

        # Verificar si algún hover completó los 3 segundos
        check_seleccion()

        # ── Construir canvas total ──────────────────────────────────────────
        panel_frases, _ = construir_panel_frases(cursor_x, cursor_y)
        panel_cuerpo    = construir_panel_cuerpo(cursor_x, cursor_y,
                                                  offset_x=PANEL_FRASES_W)
        panel_dolor     = construir_panel_dolor(cursor_x, cursor_y,
                                                 offset_x=PANEL_FRASES_W + PANEL_CUERPO_W)

        canvas = np.hstack([panel_frases, panel_cuerpo, panel_dolor])

        # ── Info en cámara ──────────────────────────────────────────────────
        color_g = (0, 200, 0) if gesto == "apuntando" else (100, 100, 100)
        cv2.putText(display, f"Gesto: {gesto}", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_g, 2)

        # Mostrar progreso del hover activo en cámara
        prog_activo = max(
            progreso_hover(E.hover_frase_ini) if E.hover_frase >= 0 else 0.0,
            progreso_hover(E.hover_dolor_ini) if E.hover_dolor >= 0 else 0.0,
        )
        if prog_activo > 0:
            bar_w = int(200 * prog_activo)
            cv2.rectangle(display, (10, h-50), (210, h-35), (50,50,50), -1)
            cv2.rectangle(display, (10, h-50), (10+bar_w, h-35), (0,220,130), -1)
            cv2.putText(display, f"{int(prog_activo*100)}%", (215, h-37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,130), 2)

        cv2.imshow("Camara", display)
        cv2.imshow("Sistema de Comunicacion", canvas)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        elif tecla == ord('c'):
            E.historial.clear()
            E.nivel_dolor = None

cap.release()
cv2.destroyAllWindows()