import cv2
import numpy as np
import mediapipe as mp
import face_recognition as fr
import math
import os
from django.shortcuts import render
from django.http import JsonResponse

mpDraw = mp.solutions.drawing_utils
FaceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

# Paths
OutFolderPathFace = 'C:/Users/John/PycharmProjects/ReconocimientoEva1/DataBase/Faces'
img_check = cv2.imread("static/images/check.png")
img_step0 = cv2.imread("static/images/Step0.png")
img_step1 = cv2.imread("static/images/Step1.png")
img_step2 = cv2.imread("static/images/Step2.png")
img_liche = cv2.imread("static/images/LivenessCheck.png")

# Variables globales
parpadeo = False
conteo = 0
step = 0
FaceCode = []
clases = []
UserName = ''
offsety = 30
offsetx = 20
confThreshold = 0.5


def home(request):
    return render(request, 'recognition/home.html')


def login_biometric(request):
    return render(request, 'recognition/login.html')


def process_video(request):
    global step, conteo, parpadeo, UserName, FaceCode, clases

    if request.method == 'POST':
        file = request.FILES['video']
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame_save = frame.copy()
        frame = cv2.resize(frame, (1280, 720))
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = FaceMesh.process(frameRGB)

        px = []
        py = []
        lista = []

        if res.multi_face_landmarks:
            for rostros in res.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, rostros, mp.solutions.face_mesh.FACE_CONNECTIONS, ConfigDraw, ConfigDraw)
                for id, puntos in enumerate(rostros.landmark):
                    al, an, c = frame.shape
                    x, y = int(puntos.x * an), int(puntos.y * al)
                    px.append(x)
                    py.append(y)
                    lista.append([id, x, y])

                    if len(lista) == 468:
                        x1, y1 = lista[145][1:]
                        x2, y2 = lista[159][1:]
                        longitud1 = math.hypot(x2 - x1, y2 - y1)

                        x3, y3 = lista[374][1:]
                        x4, y4 = lista[386][1:]
                        longitud2 = math.hypot(x4 - x3, y4 - y3)

                        x5, y5 = lista[139][1:]
                        x6, y6 = lista[368][1:]

                        x7, y7 = lista[70][1:]
                        x8, y8 = lista[300][1:]

                        faces = detector.process(frameRGB)

                        if faces.detections is not None:
                            for face in faces.detections:
                                score = face.score[0]
                                bbox = face.location_data.relative_bounding_box

                                if score > confThreshold:
                                    xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                    xi, yi, anc, alt = int(xi * an), int(yi * al), int(anc * an), int(alt * al)

                                    offsetan = (offsetx / 100) * anc
                                    xi = int(xi - int(offsetan / 2))
                                    anc = int(anc + offsetan)
                                    offsetal = (offsety / 100) * alt
                                    yi = int(yi - offsetal)
                                    alt = int(alt + offsetal)

                                    if xi < 0: xi = 0
                                    if yi < 0: yi = 0

                                    if step == 0:
                                        cv2.rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)
                                        frame[50:50 + img_step0.shape[0], 50:50 + img_step0.shape[1]] = img_step0
                                        frame[50:50 + img_step1.shape[0], 1030:1030 + img_step1.shape[1]] = img_step1
                                        frame[270:270 + img_step2.shape[0], 1030:1030 + img_step2.shape[1]] = img_step2

                                        if x7 > x5 and x8 < x6:
                                            frame[165:165 + img_check.shape[0],
                                            1105:1105 + img_check.shape[1]] = img_check

                                            if longitud1 <= 10 and longitud2 <= 10 and not parpadeo:
                                                conteo += 1
                                                parpadeo = True
                                            elif longitud1 > 10 and longitud2 > 10 and parpadeo:
                                                parpadeo = False

                                            cv2.putText(frame, f'Parpadeos:{int(conteo)}', (1070, 375),
                                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                                            if conteo >= 3:
                                                frame[385:385 + img_check.shape[0],
                                                1105:1105 + img_check.shape[1]] = img_check
                                                if longitud1 > 15 and longitud2 > 15:
                                                    step = 1
                                        else:
                                            conteo = 0

                                    if step == 1:
                                        cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                        frame[50:50 + img_liche.shape[0], 50:50 + img_liche.shape[1]] = img_liche

                                        facess = fr.face_locations(frameRGB)
                                        facescod = fr.face_encodings(frameRGB, facess)

                                        for facecod, facesloc in zip(facescod, facess):
                                            Match = fr.compare_faces(FaceCode, facecod)
                                            simi = fr.face_distance(FaceCode, facecod)
                                            min_idx = np.argmin(simi)

                                            if Match[min_idx]:
                                                UserName = clases[min_idx].upper()
                                                return JsonResponse({'result': 'success', 'username': UserName})

                            cv2.circle(frame, (x7, y7), 2, (255, 0, 0), cv2.FILLED)
                            cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = buffer.tobytes()
        return JsonResponse({'result': 'success', 'frame': frame_encoded})

    return JsonResponse({'result': 'error', 'message': 'Método no permitido'}, status=405)


def Code_Face(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = fr.face_locations(img)
        encodings.append(fr.face_encodings(img, boxes)[0])
    return encodings


def Sign(request):
    global FaceCode, clases, images

    images = []
    clases = []
    lista = os.listdir(OutFolderPathFace)

    for lis in lista:
        imgdb = cv2.imread(f"{OutFolderPathFace}/{lis}")
        images.append(imgdb)
        clases.append(os.path.splitext(lis)[0])

    FaceCode = Code_Face(images)

    return render(request, 'recognition/login.html')