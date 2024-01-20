import cv2
import mediapipe as mp
import autopy

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# maximum number of hands
hands = mp_hands.Hands(max_num_hands=1)

# screen setup
screen_width, screen_height = autopy.screen.size()

# hand  to screen coordinates mapping
def map_coordinates(x, y):
    mapped_x = int(x * screen_width)
    mapped_y = int(y * screen_height)
    return mapped_x, mapped_y

# left click
def perform_left_click():
    autopy.mouse.click()

# main
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # screen flip
    frame = cv2.flip(frame, 1)

    # image BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # hand landmark detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # index finger (8)
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = index_finger_tip.x, index_finger_tip.y

            # hand to screen mapping
            mapped_x, mapped_y = map_coordinates(x, y)

            # moving mouse pointer
            autopy.mouse.move(mapped_x, mapped_y)

            # check index finger
            is_index_finger_extended = hand_landmarks.landmark[7].y < hand_landmarks.landmark[6].y
            if is_index_finger_extended:
                perform_left_click()

            # draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # display
    cv2.imshow("AI Mouse Pointer", frame)

    # exit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# close camera and windows
cap.release()
cv2.destroyAllWindows()