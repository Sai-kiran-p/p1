import cv2
import numpy as np

menu_items = ["pizza", "burger", "spring_rolls", "chicken", "dosa", "idly"]
images = {}
for item in menu_items:
    img_path = f"{item}.jpg"   # adjust if you use .png
    img = cv2.imread(img_path)
    if img is not None:
        images[item] = cv2.resize(img, (150, 150))
    else:
        # fallback placeholder
        placeholder = np.ones((150, 150, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder, item, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        images[item] = placeholder
