# Connor Robinet 700735488
import cv2
import mediapipe
import pyautogui
import time

# --- Setup ---
face_mesh = mediapipe.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    static_image_mode=False
)

screen_w, screen_h = pyautogui.size()

sensitivity = 3.5
prev_mouse_x = prev_mouse_y = None

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 60)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Click cooldowns
last_left_click = 0
last_right_click = 0
click_cooldown = 1.0

# --- Main Loop ---
while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert once
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # --- Nose Tracking ---
        nose = lm[1]
        nx = int(nose.x * w)
        ny = int(nose.y * h)

        mouse_x = int((nx / w) * screen_w)
        mouse_y = int((ny / h) * screen_h)

        if prev_mouse_x is None:
            prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

        # Smoothing
        smooth_factor = 0.15
        sm_x = prev_mouse_x + (mouse_x - prev_mouse_x) * smooth_factor
        sm_y = prev_mouse_y + (mouse_y - prev_mouse_y) * smooth_factor

        # Delta
        dx_raw = sm_x - prev_mouse_x
        dy_raw = sm_y - prev_mouse_y

        # Dead zone
        if abs(dx_raw) < 2:
            dx_raw = 0
        if abs(dy_raw) < 2:
            dy_raw = 0

        # Acceleration
        accel = 0.02
        dx = dx_raw * (1 + accel * abs(dx_raw)) * sensitivity
        dy = dy_raw * (1 + accel * abs(dy_raw)) * sensitivity

        # Move mouse
        if dx != 0 or dy != 0:
            pyautogui.moveRel(dx, dy)

        prev_mouse_x, prev_mouse_y = sm_x, sm_y

        # --- Blink Detection ---
        left_top = lm[145].y
        left_bottom = lm[159].y
        right_top = lm[374].y
        right_bottom = lm[386].y

        left_eye_closed = (left_top - left_bottom) < 0.01
        right_eye_closed = (right_top - right_bottom) < 0.01

        now = time.time()

        # Left click
        if left_eye_closed and not right_eye_closed:
            if now - last_left_click > click_cooldown:
                pyautogui.moveRel(2, 0)
                pyautogui.moveRel(-2, 0)
                time.sleep(0.015)
                pyautogui.click()
                last_left_click = now
                print("Left Click")

        # Right click
        if right_eye_closed and not left_eye_closed:
            if now - last_right_click > click_cooldown:
                pyautogui.moveRel(2, 0)
                pyautogui.moveRel(-2, 0)
                time.sleep(0.015)
                pyautogui.rightClick()
                last_right_click = now
                print("Right Click")

    # Display
    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
