


import mediapipe as mp  # Detección de rostro y ojos
import numpy as np      # Cálculos matemáticos
import time             # Control de tiempos

import pyttsx3

import cv2              # Cámara y ventanas gráficas
from PIL import Image, ImageDraw, ImageFont     #Acentos para cv2

font = ImageFont.truetype("arial.ttf", 24) # fuente para acentos

# =========================
# CONFIGURACIÓN AJUSTABLE
# =========================
UMBRAL_EAR = 0.20          # Si EAR < 0.20 => ojo cerrado
FRAMES_CERRADOS = 4        # Frames mínimos cerrados para contar parpadeo
VENTANA_DOBLE = 1.2        # Segundos entre 2 parpadeos para contar doble parpadeo
INTERVALO_ROTACION = 3.0   # Segundos entre frases del carrusel


# libreria no soporta acentos unicode
'''
FRASES = [
    "Necesito Ayuda", "Quiero hablar con mi familia", 
    "Siento Dolor", "Sí", "No"
]
'''


FRASES = [
    "Me duele", "¿Qué está pasando?", "Quiero ver a mi familia",
    "Estoy bien", "Tengo miedo", "Gracias por estar aquí, los quiero"
]


# =========================
# DETECCIÓN DE OJOS
# =========================
# Se crea un objeto de la malla completa de puntos en la cara pero solo se seleccionan los puntos para cada ojo
malla_cara = mp.solutions.face_mesh
OJO_IZQ = [33, 160, 158, 133, 153, 144]
OJO_DER = [362, 385, 387, 263, 373, 380]



def texto_a_voz(texto):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)   # Ajusta velocidad
    engine.setProperty('volume', 1.0)
    engine.say(texto)         
    engine.runAndWait()

#Calcula Relación de Aspecto del Ojo, tecnica ya documentada para saber si esta abierto o cerrado
#landmarks tiene el conjunto de todos los puntos
def calcular_EAR(landmarks, ojo, w, h):
    #EAR relación de aspecto del ojo / se normalizan las coordenadas entre 0 y 1
    #lista por comprensión
    puntos = [(int(landmarks.landmark[i].x * w),
               int(landmarks.landmark[i].y * h)) for i in ojo]
    A = np.linalg.norm(np.array(puntos[1]) - np.array(puntos[5])) #calcula distancia euclidiana
    B = np.linalg.norm(np.array(puntos[2]) - np.array(puntos[4]))
    C = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[3]))
    return (A + B) / (2.0 * C + 1e-6) #0.000001
#Si por alguna razón C fuera 0 (ejemplo: error en detección de puntos), se produciría una división por #cero, lo que genera un error en Python.

# se recibe la imagen donde se dibujará el texto
# img: imagen, texto: string a mostras, y: altura, color texto, negro RGB, escala: tamaño texto, grosor

def dibujar_texto_centrado(img, texto, y, color=(0,0,0), escala=1.2, grosor=2):
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor) #cv2.getTextSize devuelve el ancho y alto que ocupará el texto en píxeles.
    x = max(10, (img.shape[1] - w) // 2)

#img.shape[1] es el ancho total de la imagen.
#(img.shape[1] - w) // 2 calcula la posición X para que el texto quede centrado.
#mínimo 10 px de margen
    #cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)
    # cv2.FONT_HERSHEY_SIMPLEX  e sun tipo de fuente de la libreria  corresponde a sans-serif
    draw.text((x, y), texto, font=ImageFont.truetype("arial.ttf", 50), fill=color)

# =========================
# CÁMARA
# =========================
camara = cv2.VideoCapture(0) #abre la cámara si se esscribe 1 búsca otra 

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")


frase_actual = 0
ultimo_cambio = time.time() #guarda instante actual en segundos
historial = []

frames_cerrados = 0
parpadeos = 0
ultimo_parpadeo = 0.0





#faceMesh detecta 468 puntos de referencia
# bloque with en python asegura que cuando se abre un recurso  se libere 
with malla_cara.FaceMesh(
    static_image_mode=False,# indica se trabaja con video en teimpo real
    max_num_faces=1, # solo detecta una cara
    refine_landmarks=False, # puede reconocer puntos más precisos en ojos y labios
    min_detection_confidence=0.5, # confianza miniam 50% para considerar la detección de cara
    min_tracking_confidence=0.5 # confianza minima para seguir la cara entre frames , is baja de 50% se volvera a la detección
    
    ) as malla: # se crea el objeto de Mediapipe que detecta la malla facial 

    while True:
        ok, frame = camara.read() # lee un frame o una imagen desde la cámara, devuele un true, imagen capturada (matriz pixeles)
        if not ok: break 

        frame = cv2.flip(frame, 1)# invierte la imagen horizontalmente (efecto espejo)
        h, w = frame.shape[:2] # devuele: alto, ancho, canales,  toma solo alto y ancho [:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convierte formato BGR  a RGB mediapipe
        
       

        #conversión espacios de color
        resultados = malla.process(rgb) #pasa la imagen en RGB al modelo de Mediapipe (malla)

        display = frame.copy() # se crea una copia de la ventana original, se usa para dibujar encima los puntos de los ojos
        ear_prom = None # se inicializa la variable
        ahora = time.time() #guarda le timpo actual en segundos

        if resultados.multi_face_landmarks:
            rostro = resultados.multi_face_landmarks[0]
         #se verifica si al menos se detecto una cara y se guarda en rostro
            # EAR por ojo y promedio
            ear_izq = calcular_EAR(rostro, OJO_IZQ, w, h)
            ear_der = calcular_EAR(rostro, OJO_DER, w, h)
            ear_prom = (ear_izq + ear_der) / 2.0

            # Dibujar puntos de ojos
            for idx in OJO_IZQ + OJO_DER:# itera en los arreglos
                x = int(rostro.landmark[idx].x * w)
                y = int(rostro.landmark[idx].y * h)
                cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
                #dibuja el circulo verde en la imagen, radio 3 pixeles,  grosor -1, circulo de relleno

            #Crea la malla facial completa
            mp.solutions.drawing_utils.draw_landmarks(display, rostro, malla_cara.FACEMESH_TESSELATION,
                                                       mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1),
                                                       mp.solutions.drawing_utils.DrawingSpec(color=(255,100,0), thickness=1, circle_radius=1))

            # ---------- Detección de parpadeo ----------
            if ear_prom < UMBRAL_EAR:
                frames_cerrados += 1 # se incrememta para saber cuando frames los ojos están cerrados
            else: #ojo abierto

               # Cuando el ojo vuelve a abrirse, se revisa si estuvo cerrado al menos FRAMES_CERRADOS frames
               #  #(para evitar falsos positivos por ruido).
                if frames_cerrados >= FRAMES_CERRADOS:
                    '''if ahora - ultimo_parpadeo <= VENTANA_DOBLE: #detecta doble parpadeo
                        parpadeos += 1
                    else:
                        parpadeos = 1
                    ultimo_parpadeo = ahora
# acción al detectar parpadeo
                    if parpadeos >= 2:'''
                    historial.append(FRASES[frase_actual]) # se guarda frase actual
                    #parpadeos = 0 # reinicio contador
                    ultimo_cambio = ahora # se actualiza tiempo último
                        
                    #pyttsx3
                    texto_a_voz(FRASES[frase_actual])           
                        #ventana, texto, coordenadas, tipo letra, tamaño,  color, grosor letras
                    cv2.putText(display, "✔ Seleccionado", (30, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

                frames_cerrados = 0 #reinicio contador

        # Rotación automática de frases
        if time.time() - ultimo_cambio > INTERVALO_ROTACION: #guarda última frase
            frase_actual = (frase_actual + 1) % len(FRASES) # división entera, len número elementos
            #rotación: si frase_actual = 4 (última) → (4+1) % 5 = 0, vuelve a la primera.
            # el % devuelve el residuo
            ultimo_cambio = time.time()

        # ========== Ventana de frases ==========
        ancho, alto = 1000, 500
        img_frase = np.full((alto, ancho, 3), 255, np.uint8) # lienz RGB de color blanco

        



        
        
        
        
        cv2.putText(img_frase, "Manten los ojos cerrados 2 segundos para seleccionar",
                    (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.putText(img_frase, f"EAR promedio: {ear_prom:.2f}" if ear_prom is not None else "EAR promedio: --",
                    (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,60), 2)
        cv2.putText(img_frase, f"Parpadeos en ventana: {parpadeos}",
                    (30, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,60), 2)
        cv2.putText(img_frase, "ESC: salir, C: limpiar historial",
                    (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 2)
        

        


                    
       

        if ear_prom is not None: # muestra valor de EAR si ya se calculó
            cv2.putText(display, f"EAR: {ear_prom:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        

        rgb_pil = cv2.cvtColor(img_frase, cv2.COLOR_BGR2RGB) #convierte a RGB para PIL
        img_pil = Image.fromarray(rgb_pil) #convierte la imagen a formato PIL para dibujar texto con acentos
        draw = ImageDraw.Draw(img_pil)

        #draw.text((x, y), texto, font=ImageFont.truetype("arial.ttf", 35), fill=color)
        # se muestra historial de frases
        texto_historial = "Seleccionadas: " + (" | ".join(historial[-3:]) if historial else "(vacío)")
        
        #se usa pil en lugar del cv2
        #cv2.putText(img_frase, texto_historial[:80], (30, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,120,0), 2)
        draw.text((30, 375), texto_historial[:80], font=ImageFont.truetype("arial.ttf", 30), fill=(0,120,0), stroke_width=2, stroke_fill=(255,255,255))

        #llamado al método para dibujar 
        dibujar_texto_centrado(img_frase, "Frase actual", 60, color=(70,70,70), escala=1.0, grosor=2)
        dibujar_texto_centrado(img_frase, FRASES[frase_actual], 120, color=(0,0,200), escala=2.0, grosor=3)

        img_frase = np.array(img_pil)
        #img_frase = cv2.cvtColor(img_frase, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camara", display)
        cv2.imshow("Frases", img_frase)

#captura de teclas para salir
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == 27:  # ESC
            break
        elif tecla == ord('c'): #limpia historial
            historial = []

camara.release()
cv2.destroyAllWindows()

