import cv2, pygame, sys, csv, time
import mediapipe as mp

# --- config
WIDTH, HEIGHT = 640, 480
CSV_PATH = "data/raw/hand_log.csv"

# --- setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # if you have multiple cams, try 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

ball_x, ball_y = WIDTH // 2, HEIGHT // 2
logging_on = False
csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["ts_ms", "wrist_x", "wrist_y"])  # normalized 0..1

def draw_text(s, x, y):
    img = font.render(s, True, (200, 200, 200))
    screen.blit(img, (x, y))

while True:
    # --- pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            csv_file.close()
            pygame.quit()
            cap.release()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                csv_file.close()
                pygame.quit()
                cap.release()
                sys.exit()
            if event.key == pygame.K_SPACE:
                logging_on = not logging_on

    # --- camera frame
    ok, frame = cap.read()
    if not ok:
        screen.fill((0, 0, 0))
        draw_text("Camera not available", 20, 20)
        pygame.display.flip()
        clock.tick(30)
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # default: keep previous ball pos
    info = "No hand"
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        wrist = hand.landmark[0]  # normalized 0..1
        # map to screen pixels
        ball_x = int(wrist.x * WIDTH)
        ball_y = int(wrist.y * HEIGHT)
        info = f"wrist ({wrist.x:.2f}, {wrist.y:.2f})"

        if logging_on:
            ts = int(time.time() * 1000)
            writer.writerow([ts, wrist.x, wrist.y])

    # --- draw
    screen.fill((25, 25, 30))
    pygame.draw.circle(screen, (0, 200, 120), (ball_x, ball_y), 20)
    draw_text("SPACE: toggle logging  |  ESC: quit", 10, 10)
    draw_text(f"Logging: {'ON' if logging_on else 'OFF'}", 10, 35)
    draw_text(info, 10, 60)
    pygame.display.flip()
    clock.tick(30)