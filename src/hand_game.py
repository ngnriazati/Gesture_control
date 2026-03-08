import cv2, pygame, sys, time, random, csv
import mediapipe as mp
import numpy as np

# --- CONFIG ---
WIDTH, HEIGHT = 1000, 1000
SPEED = 5
TURN_SPEED = 5
SHIP_IMG_PATH = "/Users/negin/Desktop/gusture_control/data/raw/tiefighter_black_bg.png"
TARGET_IMG_PATH = "/Users/negin/Desktop/gusture_control/data/raw/xwing_bg.png"
CSV_PATH = "/Users/negin/Desktop/gusture_control/data/raw/landmarks_log.csv"

# --- INIT ---
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TIE Fighter vs X-Wing Battle")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# --- LOAD IMAGES ---
ship_img = pygame.image.load(SHIP_IMG_PATH).convert_alpha()
ship_img = pygame.transform.scale(ship_img, (120, 120))
target_img = pygame.image.load(TARGET_IMG_PATH).convert_alpha()
target_img = pygame.transform.scale(target_img, (100, 100))

# --- GAME STATE ---
ship_angle = 0
ship_x, ship_y = WIDTH // 2, HEIGHT // 2
vx, vy = 0, 0
bullets, enemy_bullets = [], []
shoot_cooldown = 0
enemy_shoot_timer = 100
target_pos = [random.randint(200, WIDTH-200), random.randint(150, HEIGHT-200)]
target_vel = [4, 2]
target_health = 5
ship_health = 5
exploded = ship_destroyed = game_over = False
turn_velocity = 0

# --- RECORDING STATE ---
recording = False
csv_file = None
writer = None

# --- FUNCTIONS ---
def draw_text(s, x, y, c=(230,230,230)):
    img = font.render(s, True, c)
    screen.blit(img, (x, y))

def finger_is_extended(hand, tip_idx, pip_idx):
    return hand.landmark[tip_idx].y < hand.landmark[pip_idx].y

def count_extended_fingers(hand):
    fingers = [finger_is_extended(hand,8,6), finger_is_extended(hand,12,10),
               finger_is_extended(hand,16,14), finger_is_extended(hand,20,18)]
    return sum(fingers)

def fist_score(hand):
    pairs = [(8,5),(12,9),(16,13),(20,17)]
    return sum(np.linalg.norm(
        np.array([hand.landmark[a].x, hand.landmark[a].y]) -
        np.array([hand.landmark[b].x, hand.landmark[b].y])) for a,b in pairs)

def gesture_from_landmarks(hand):
    if fist_score(hand) < 0.25:
        return "FIST"
    ext = count_extended_fingers(hand)
    if ext >= 3:
        return "OPEN"
    if finger_is_extended(hand,8,6) and ext == 1:
        tip = hand.landmark[8].x
        base = hand.landmark[5].x
        return "POINT_RIGHT" if (tip > base) else "POINT_LEFT"
    if ext == 2:
        return "SHOOT"
    return "UNKNOWN"

def reset_game():
    global ship_x, ship_y, vx, vy, bullets, enemy_bullets, target_pos, target_vel
    global target_health, ship_health, exploded, ship_destroyed, game_over
    global enemy_shoot_timer, ship_angle, turn_velocity
    ship_x, ship_y = WIDTH // 2, HEIGHT // 2
    vx = vy = 0
    bullets.clear(); enemy_bullets.clear()
    target_pos[:] = [random.randint(200, WIDTH-200), random.randint(150, HEIGHT-200)]
    target_vel[:] = [4, 2]
    target_health = ship_health = 5
    exploded = ship_destroyed = game_over = False
    ship_angle = turn_velocity = 0
    enemy_shoot_timer = 100

# --- MAIN LOOP ---
gesture = "NO_HAND"
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            if csv_file: csv_file.close()
            cap.release()
            pygame.quit()
            sys.exit()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_e:
                if csv_file: csv_file.close()
                cap.release()
                pygame.quit()
                sys.exit()
            # SPACE toggles recording mode
            if e.key == pygame.K_SPACE:
                if not recording:
                    csv_file = open(CSV_PATH, "w", newline="")
                    writer = csv.writer(csv_file)
                    header = ["ts_ms"]
                    for i in range(21):
                        header += [f"x{i}", f"y{i}"]
                    header += ["gesture"]
                    writer.writerow(header)

                    recording = True
                    print("🟢 Started recording hand landmarks...")
                else:
                    recording = False
                    csv_file.close()
                    csv_file = None
                    print("🛑 Stopped recording.")
            # Restart game if over
            if e.key == pygame.K_r and game_over:
                reset_game()

    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # --- GAME LOGIC ---
    if not game_over:
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            gesture = gesture_from_landmarks(hand)

            # === LOG ONLY WHEN RECORDING ===
            if recording and writer:
                ts = int(time.time() * 1000)
                coords = []
                for i in range(21):
                    coords += [f"{hand.landmark[i].x:.5f}", f"{hand.landmark[i].y:.5f}"]
                writer.writerow([ts] + coords + [gesture])
            # ===============================

            wrist = hand.landmark[0]
            hand_x, hand_y = wrist.x, wrist.y
            target_x, target_y = int(hand_x * WIDTH), int(hand_y * HEIGHT)

            # Movement
            if gesture == "FIST":
                vx = (target_x - ship_x) * 0.15
                vy = (target_y - ship_y) * 0.15
            elif gesture == "OPEN":
                vx, vy = 0, 0

            # Smooth turning
            turn_direction = 0
            if gesture == "POINT_LEFT":
                turn_direction = +1
            elif gesture == "POINT_RIGHT":
                turn_direction = -1
            turn_velocity = 0.8 * turn_velocity + 0.2 * turn_direction * TURN_SPEED
            ship_angle += turn_velocity

            # Shooting
            if gesture == "SHOOT" and shoot_cooldown <= 0:
                firing_angle = ship_angle + 90
                nose_offset = 60
                rad = np.radians(firing_angle)
                nose_x = ship_x + np.cos(rad) * nose_offset
                nose_y = ship_y - np.sin(rad) * nose_offset
                bullets.append([nose_x, nose_y, firing_angle])
                shoot_cooldown = 20
        else:
            gesture = "NO_HAND"

        # Movement and battle logic (unchanged)
        ship_x = np.clip(ship_x + vx, 0, WIDTH)
        ship_y = np.clip(ship_y + vy, 0, HEIGHT)
        for b in bullets:
            b[0] += np.cos(np.radians(b[2])) * 25
            b[1] -= np.sin(np.radians(b[2])) * 25
        bullets = [b for b in bullets if 0 < b[0] < WIDTH and 0 < b[1] < HEIGHT]
        shoot_cooldown -= 1
        target_pos[0] += target_vel[0]
        target_pos[1] += target_vel[1]
        if target_pos[0] <= 80 or target_pos[0] >= WIDTH - 80:
            target_vel[0] *= -1
        if target_pos[1] <= 80 or target_pos[1] >= HEIGHT - 80:
            target_vel[1] *= -1

        enemy_shoot_timer -= 1
        if enemy_shoot_timer <= 0:
            dx, dy = ship_x - target_pos[0], ship_y - target_pos[1]
            angle = np.degrees(np.arctan2(-dy, dx))
            enemy_bullets.append([target_pos[0], target_pos[1], angle])
            enemy_shoot_timer = random.randint(80, 150)
        for eb in enemy_bullets:
            eb[0] += np.cos(np.radians(eb[2])) * 15
            eb[1] -= np.sin(np.radians(eb[2])) * 15
        enemy_bullets = [eb for eb in enemy_bullets if 0 < eb[0] < WIDTH and 0 < eb[1] < HEIGHT]

        if not exploded:
            for b in bullets[:]:
                if np.hypot(b[0]-target_pos[0], b[1]-target_pos[1]) < 60:
                    bullets.remove(b)
                    target_health -= 1
                    if target_health <= 0:
                        exploded = True
                        game_over = True
        if not ship_destroyed:
            for eb in enemy_bullets[:]:
                if np.hypot(eb[0]-ship_x, eb[1]-ship_y) < 40:
                    enemy_bullets.remove(eb)
                    ship_health -= 1
                    if ship_health <= 0:
                        ship_destroyed = True
                        game_over = True

    # --- DRAW ---
    screen.fill((0, 0, 0))
    if not exploded:
        rect = target_img.get_rect(center=target_pos)
        screen.blit(target_img, rect)
        bar_w, bar_h = 100, 10
        health_ratio = target_health / 5
        pygame.draw.rect(screen, (255,0,0), (target_pos[0]-50, target_pos[1]-80, bar_w, bar_h))
        pygame.draw.rect(screen, (0,255,0), (target_pos[0]-50, target_pos[1]-80, bar_w*health_ratio, bar_h))
    if not ship_destroyed:
        rotated_ship = pygame.transform.rotate(ship_img, ship_angle)
        rect = rotated_ship.get_rect(center=(ship_x, ship_y))
        screen.blit(rotated_ship, rect)
        bar_w, bar_h = 120, 10
        health_ratio = ship_health / 5
        pygame.draw.rect(screen, (255,0,0), (ship_x-60, ship_y+70, bar_w, bar_h))
        pygame.draw.rect(screen, (0,255,0), (ship_x-60, ship_y+70, bar_w*health_ratio, bar_h))
    for b in bullets:
        pygame.draw.line(screen, (255, 0, 0),
                         (int(b[0]), int(b[1])),
                         (int(b[0] + np.cos(np.radians(b[2])) * 20),
                          int(b[1] - np.sin(np.radians(b[2])) * 20)), 4)
    for eb in enemy_bullets:
        pygame.draw.line(screen, (50, 100, 255),
                         (int(eb[0]), int(eb[1])),
                         (int(eb[0] + np.cos(np.radians(eb[2])) * 15),
                          int(eb[1] - np.sin(np.radians(eb[2])) * 15)), 4)

    if exploded:
        draw_text("💥 X-WING DESTROYED 💥", WIDTH//2 - 150, HEIGHT//2 - 20, (255, 50, 50))
        draw_text("YOU WIN!", WIDTH//2 - 60, HEIGHT//2 + 20, (0,255,0))
        draw_text("Press R to Restart or E to Exit", WIDTH//2 - 160, HEIGHT//2 + 60, (200,200,200))
    elif ship_destroyed:
        draw_text("💥 TIE FIGHTER DESTROYED 💥", WIDTH//2 - 160, HEIGHT//2 - 20, (255, 50, 50))
        draw_text("X-WING WINS!", WIDTH//2 - 90, HEIGHT//2 + 20, (0,255,0))
        draw_text("Press R to Restart or E to Exit", WIDTH//2 - 160, HEIGHT//2 + 60, (200,200,200))

    draw_text(f"Gesture: {gesture}", 20, 20)
    draw_text(f"Pos: ({int(ship_x)}, {int(ship_y)})", 20, 50)
    if recording:
        draw_text("🔴 RECORDING", WIDTH - 180, 20, (255, 50, 50))  # indicator
    pygame.display.flip()
    clock.tick(30)
