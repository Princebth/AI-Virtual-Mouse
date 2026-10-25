import cv2
import mediapipe as mp
import time
import math
import pyautogui
import pyperclip

# ================= SETUP =================
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# ================= VARIABLES =================
frameR = 100
pTime = 0

alpha = 0.2
deadZone = 5

smoothX, smoothY = screen_width // 2, screen_height // 2

clickThreshold = 40
doubleClickThreshold = 25

clicked = False
doubleClicked = False

prevIndexY = 0
scrollCooldown = 0
scrollDelay = 12
swipeThreshold = 25

pauseTriggered = False
likeTriggered = False
saveTriggered = False
copyTriggered = False

gestureText = ""

# ================= WINDOW =================
cv2.namedWindow("Hand Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Gesture Control", 1000, 700)

# ================= MAIN LOOP =================
while True:

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    hCam, wCam, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cv2.rectangle(img, (frameR, frameR),
                  (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    gestureText = ""

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * wCam), int(lm.y * hCam)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) > 12:

                _, ix, iy = lmList[8]
                _, i2x, i2y = lmList[6]
                _, mx, my = lmList[12]
                _, tx, ty = lmList[4]
                _, px, py = lmList[0]

                # ============ CURSOR MOVE ============
                indexUp = iy < i2y

                if indexUp:

                    x3 = int((ix - frameR) * screen_width / (wCam - 2 * frameR))
                    y3 = int((iy - frameR) * screen_height / (hCam - 2 * frameR))

                    x3 = max(0, min(screen_width, x3))
                    y3 = max(0, min(screen_height, y3))

                    dx = x3 - smoothX
                    dy = y3 - smoothY

                    if abs(dx) < deadZone: dx = 0
                    if abs(dy) < deadZone: dy = 0

                    smoothX += dx * alpha
                    smoothY += dy * alpha

                    pyautogui.moveTo(smoothX, smoothY)
                    gestureText = "MOVE CURSOR"

                # ============ CLICK ============
                clickDistance = math.hypot(tx - ix, ty - iy)

                if clickDistance < clickThreshold and not clicked:
                    pyautogui.click()
                    clicked = True
                    gestureText = "CLICK"

                if clickDistance > clickThreshold + 10:
                    clicked = False

                # ============ DOUBLE CLICK ============
                if clickDistance < doubleClickThreshold and not doubleClicked:
                    pyautogui.doubleClick()
                    doubleClicked = True
                    gestureText = "DOUBLE CLICK"

                if clickDistance > doubleClickThreshold + 10:
                    doubleClicked = False

                # ============ SCROLL ============
                if scrollCooldown == 0:

                    deltaY = iy - prevIndexY

                    if deltaY < -swipeThreshold:
                        pyautogui.scroll(-600)
                        scrollCooldown = scrollDelay
                        gestureText = "SCROLL DOWN"

                    elif deltaY > swipeThreshold:
                        pyautogui.scroll(600)
                        scrollCooldown = scrollDelay
                        gestureText = "SCROLL UP"

                else:
                    scrollCooldown -= 1

                prevIndexY = iy

                # ============ PLAY / PAUSE ============
                palmOpen = abs(lmList[8][2] - lmList[20][2]) > 120

                if palmOpen and not pauseTriggered:
                    pyautogui.press('space')
                    pauseTriggered = True
                    gestureText = "PLAY / PAUSE"

                if not palmOpen:
                    pauseTriggered = False

                # ============ LIKE ============
                indexMiddleDist = math.hypot(mx - ix, my - iy)

                if indexMiddleDist < 30 and not likeTriggered:
                    pyautogui.press('l')
                    likeTriggered = True
                    gestureText = "LIKE"

                if indexMiddleDist > 45:
                    likeTriggered = False

                # ============ SAVE ============
                thumbUp = ty < iy - 40

                if thumbUp and not saveTriggered:
                    pyautogui.press('s')
                    saveTriggered = True
                    gestureText = "SAVE"

                if not thumbUp:
                    saveTriggered = False

                # ============ COPY FEATURE (NEW) ============
                # pinch index + thumb → copy selected text
                copyDistance = math.hypot(ix - tx, iy - ty)

                if copyDistance < 25 and not copyTriggered:
                    pyautogui.hotkey('ctrl', 'c')
                    copyTriggered = True
                    gestureText = "COPY"

                if copyDistance > 40:
                    copyTriggered = False

    # ================= FPS =================
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',
                (20, 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255), 2)

    cv2.putText(img, f'Gesture: {gestureText}',
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()