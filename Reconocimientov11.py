#Librerias
import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from tkinter import *
from PIL import Image, ImageTk
import imutils
import math
#Profile
def Profile():
    global step, conteo, UserName, OutFolderPathUser
    #Reseteamos Variables

    step = 0
    conteo = 0

    # Window
    pantalla4 = Toplevel(pantalla)
    pantalla4.title("BIENVENIDO A TU EXAMEN")
    pantalla4.geometry("1280x720")

#Code Face Funcion
def Code_Face(images):
    listacod = []

    #Iteramos
    for img in images:
        #color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image code
        cod = fr.face_encodings(img)[0]
        #save lista
        listacod.append(cod)

    return listacod

#Close Window Funcion
def Close_Window2():
    global step, conteo
    #Reset
    conteo = 0
    step = 0
    pantalla3.destroy()
#Sign Biometric Funcion
def Sign_Biometric():
    global LogUser, LogPass, OutFolderPathFace, cap, lblVideo, pantalla3, FaceCode, clases, images,pantalla2,step,parpadeo,conteo,UserName

    # Leemos la videocaptura
    if cap is not None:
        ret, frame = cap.read()
        frameSave = frame.copy()
        # Redimencionamos
        frame = imutils.resize(frame, width=1280)
        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Frame show

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:

            #INTERFERENCIA FACE MESH
            res = FaceMesh.process(frameRGB)

            # lista de Resultados

            px = []
            py = []
            lista = []

            if res.multi_face_landmarks:
                #Extrac face
                for rostros in res.multi_face_landmarks:
                    # Draw
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACE_CONNECTIONS, ConfigDraw, ConfigDraw)

                    # Extraer lo puntos

                    for id, puntos in enumerate(rostros.landmark):
                        # Informacion de las imagenes
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 puntos

                        if len(lista) == 468:
                            # Ojos derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot(x2 - x1, y2 - y1)

                            # Ojos Izquierdo:

                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            # Parietal Derecho
                            x5, y5 = lista[139][1:]
                            # Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja Derecha
                            x7, y7 = lista[70][1:]
                            # Ceja Izquierda
                            x8, y8 = lista[300][1:]

                            # Face Detect

                            faces = detector.process(frameRGB)

                            if faces.detections is not None:

                                for face in faces.detections:
                                    # Recuadro de rostro
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # thesold
                                    if score > confThreshold:
                                        # Coordenates
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int(xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        # Ofset

                                        offsetan = (offsetx / 100) * anc
                                        xi = int(xi - int(offsetan / 2))
                                        anc = int(anc + offsetan)
                                        xf = xi + anc

                                        # Height
                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        # Error < 0
                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if anc < 0: an = 0
                                        if alt < 0: al = 0

                                        # Steps

                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)

                                            # image step0

                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step0

                                            # IMG Step1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1

                                            # IMG Step2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            # Condiciones
                                            if x7 > x5 and x8 < x6:
                                                # Imagen Check
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check

                                                # Cont Parpadeo

                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:  # Parpadeo
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:  # Seguridad parpadeo
                                                    parpadeo = False

                                                    # CONTEO DE PARPADEO

                                                cv2.putText(frame, f'Parpadeos:{int(conteo)}', (1070, 375),
                                                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                                                if conteo >= 3:
                                                    # IMG check
                                                    alch, anch, c = img_check.shape
                                                    frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 15 and longitud2 > 15:

                                                        # Sep 1

                                                        step = 1

                                            else:
                                                conteo = 0

                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            # IMG Chek livenes
                                            alli, anli, c = img_liche.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_liche

                                            # Encontramos los Nuevos rostros

                                            facess = fr.face_locations(frameRGB)
                                            facescod = fr.face_encodings(frameRGB, facess)

                                            # Iteramos

                                            for facecod, facesloc in zip(facescod, facess):
                                                # MATCHING
                                                Match = fr.compare_faces(FaceCode, facecod)

                                                # Sim
                                                simi = fr.face_distance(FaceCode, facecod)

                                                # min
                                                min = np.argmin(simi)

                                                if Match[min]:
                                                    #nombre de la persona
                                                    UserName = clases[min].upper()

                                                    Profile()

                                #close
                                close = pantalla3.protocol("WM_DELETE_WINDOW", Close_Window2)

                            # Circulos
                            cv2.circle(frame, (x7, y7), 2, (255, 0, 0), cv2.FILLED)
                            cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)

        # Convertir el Video

        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Mostrar el video

        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, Sign_Biometric)

    else:
        cap.release()

#Funcion Sign
def Sign ():
    global LogUser, LogPass, OutFolderPathFace, cap, lblVideo, pantalla3, FaceCode, clases, images
    # Extraemos nombres

    # Base de datos rostros
    images = []
    clases = []
    lista = os.listdir(OutFolderPathFace)

    # Leer imágenes de caras

    for lis in lista:
        # Read images
        imgdb = cv2.imread(f"{OutFolderPathFace}/{lis}")

        # Guardamos las imagenes
        images.append(imgdb)
        # Nombre de las imagenes
        clases.append(os.path.splitext(lis)[0])

    #Face Code
    FaceCode = Code_Face(images)

    # Window
    pantalla3 = Toplevel(pantalla)
    pantalla3.title("LoginBiometic")
    pantalla3.geometry("1280x720")

    # label video o captura de pantalla

    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    # Video Captura o cmamara

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Sign_Biometric()

#Funcion Log

#Path

OutFolderPathUser = 'C:/Users/John/PycharmProjects/ReconocimientoEva1/DataBase/Users'
PathUserCheck = 'C:/Users/John/PycharmProjects/ReconocimientoEva1/DataBase/Users/'
OutFolderPathFace = 'C:/Users/John/PycharmProjects/ReconocimientoEva1/DataBase/Faces'

# Leer imagenes

img_check = cv2.imread("C:/Users/John/PycharmProjects/ReconocimientoEva1/SetUp/check.png")
img_step0 = cv2.imread("C:/Users/John/PycharmProjects/ReconocimientoEva1/SetUp/Step0.png")
img_step1 = cv2.imread("C:/Users/John/PycharmProjects/ReconocimientoEva1/SetUp/Step1.png")
img_step2 = cv2.imread("C:/Users/John/PycharmProjects/ReconocimientoEva1/SetUp/Step2.png")
img_liche = cv2.imread("C:/Users/John/PycharmProjects/ReconocimientoEva1/SetUp/LivenessCheck.png")

#Variables

parpadeo = False
conteo = 0
muestra = 0
step = 0

# Margen
offsety = 30
offsetx = 20

# Umbral
confThreshold = 0.5
blurThreshold = 15

#tools draw

mpDraw =mp.solutions.drawing_utils
ConfigDraw =mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Object Face Mesh

FacemeshObject = mp.solutions.face_mesh
FaceMesh = FacemeshObject.FaceMesh(max_num_faces=1)

# Object Detect

FaceObject = mp.solutions.face_detection
detector = FaceObject.FaceDetection(min_detection_confidence=0.5 , model_selection=1)

#Lista

info = []

#ventana Principal
pantalla = Tk()
pantalla.title("SISTEMA DE RECONOCIMIENTO DE IDENTIDAD")
pantalla.geometry("1288x720")

#Botones
#SIG
imagenBL = PhotoImage(file="C:/Users/John/PycharmProjects/ReconocimientoEva1/SetUp/BtSign.png")
BtSign = Button(pantalla, text="Registro",image=imagenBL, height="40",width="200", command=Sign)
BtSign.place(x=900, y = 580)
pantalla.mainloop()