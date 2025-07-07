import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe hand detector
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Colors
colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]  # Purple, Blue, Green, Eraser
colorIndex = 0
drawColor = colors[colorIndex]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), np.uint8)

def findPosition(img):
    lmList = []
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((id, cx, cy))
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return lmList

xp, yp = 0, 0  # Previous positions

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    lmList = findPosition(img)

    # Draw toolbar
    cv2.rectangle(img, (0, 0), (1280, 100), (255, 255, 255), -1)
    for i, color in enumerate(colors):
        cv2.rectangle(img, (i*100, 0), ((i+1)*100, 100), color, -1)

    if lmList:
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger

        # Check if fingers are up
        fingers = [lmList[8][2] < lmList[6][2], lmList[12][2] < lmList[10][2]]  # Index, Middle

        if fingers[0] and fingers[1]:
            # Selection Mode
            xp, yp = 0, 0
            if y1 < 100:
                colorIndex = x1 // 100
                drawColor = colors[colorIndex]
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

        elif fingers[0] and not fingers[1]:
            # Drawing Mode
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = 30 if drawColor == (0, 0, 0) else 10
            cv2.line(canvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1

    # Merge canvas and image
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("AI Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
