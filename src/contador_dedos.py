import mediapipe as mp
import cv2

wcam = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = wcam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Hand.process(imgRGB)
    handPoints = result.multi_hand_landmarks
    h,w,_ = img.shape
    pontos = []
    if handPoints:
        for points in handPoints:
            # print(points)
            mpDraw.draw_landmarks(img,points,hand.HAND_CONNECTIONS)
            for id, coord in enumerate(points.landmark):
                cx, cy = int(coord.x*w), int(coord.y*h)
                # cv2.putText(img, str(id), (cx,cy+10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 2)
                pontos.append((cx,cy))
                # print(pontos)

        dedos = [8,12,16,20]
        contador = 0
        if points:
            if pontos[4][0] < pontos[2][0]:
                contador += 1
            for x in dedos:
                if pontos[x][1] < pontos[x-2][1]:
                    contador += 1
        
        cv2.putText(img,str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 5)
        # print(contador)

    cv2.imshow("Imagem", img)
    cv2.waitKey(1)
