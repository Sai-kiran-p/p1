import cv2
import mediapipe as mp
import numpy as np
import math
import os

# ============ Camera Setup ============
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows; remove if on Linux/Mac
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
screen_w, screen_h = 640, 480

# ============ Menu Items ============
menu_items = ["pizza", "burger", "spring_rolls", "chicken", "dosa", "idly"]
images = {}

# Load images or create placeholders
for item in menu_items:
    if os.path.exists(f"{item}.jpg"):
        img = cv2.imread(f"{item}.jpg")
    elif os.path.exists(f"{item}.png"):
        img = cv2.imread(f"{item}.png")
    else:
        img = None

    if img is not None:
        images[item] = cv2.resize(img, (150, 150))
    else:
        placeholder = np.ones((150, 150, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder, item, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        images[item] = placeholder

# Grid layout (2 rows x 3 cols)
box_size = 150
margin = 20
positions = []
for row in range(2):
    for col in range(3):
        x = margin + col * (box_size + margin)
        y = margin + row * (box_size + margin)
        positions.append((x, y))

# Order button
order_btn_pos = (screen_w // 2 - 75, screen_h - 100)
order_btn_size = (150, 60)

# ============ Utility Functions ============
def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

ordered_items = set()

# ============ Main Loop ============
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (screen_w, screen_h))
    overlay = frame.copy()

    selected_index = None

    # Draw menu boxes
    for i, item in enumerate(menu_items):
        x, y = positions[i]
        overlay[y:y+box_size, x:x+box_size] = images[item]
        cv2.rectangle(overlay, (x, y), (x+box_size, y+box_size), (255, 255, 255), 2)
        cv2.putText(overlay, item, (x+10, y+box_size+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw Order button
    ox, oy = order_btn_pos
    cv2.rectangle(overlay, (ox, oy), (ox+order_btn_size[0], oy+order_btn_size[1]), (0, 255, 0), -1)
    cv2.putText(overlay, "ORDER", (ox+20, oy+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            index_tip = lm[8]
            thumb_tip = lm[4]
            ix, iy = int(index_tip.x * screen_w), int(index_tip.y * screen_h)
            tx, ty = int(thumb_tip.x * screen_w), int(thumb_tip.y * screen_h)

            cv2.circle(overlay, (ix, iy), 10, (0, 255, 0), -1)
            cv2.circle(overlay, (tx, ty), 10, (255, 0, 0), -1)

            # Check menu selection
            for i, (x, y) in enumerate(positions):
                if x < ix < x+box_size and y < iy < y+box_size:
                    selected_index = i
                    cv2.rectangle(overlay, (x, y), (x+box_size, y+box_size), (0, 255, 255), 3)

                    # Collision gesture (index + thumb touch)
                    if distance((ix, iy), (tx, ty)) < 40:
                        item = menu_items[i]
                        if item in ordered_items:
                            ordered_items.remove(item)
                        else:
                            ordered_items.add(item)

            # Check Order button
            if ox < ix < ox+order_btn_size[0] and oy < iy < oy+order_btn_size[1]:
                cv2.rectangle(overlay, (ox, oy), (ox+order_btn_size[0], oy+order_btn_size[1]), (0, 255, 255), 3)
                if distance((ix, iy), (tx, ty)) < 40:
                    print("✅ Final Order:", list(ordered_items))

    # Show ordered items at bottom
    ordered_text = "Ordered: " + (", ".join(ordered_items) if ordered_items else "None")
    cv2.putText(overlay, ordered_text, (10, screen_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Merge overlay with frame
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    cv2.imshow("Food Menu", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
