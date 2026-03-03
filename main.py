import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog

# ---------- Detection Function ----------
def detect_potholes(frame):

    original = frame.copy()
    height, width = frame.shape[:2]

    roi_y = int(height * 0.4)
    roi = frame[roi_y:height, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 60, 160)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return original

    # Keep only large meaningful contours
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1000 < area < 70000:
            valid_contours.append(cnt)

    if len(valid_contours) == 0:
        return original

    # Take the largest contour only (prevents multiple circles)
    largest = max(valid_contours, key=cv2.contourArea)

    # Draw tight enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest)

    center = (int(x), int(y + roi_y))
    radius = int(radius * 0.9)  # Slight shrink to fit tighter

    cv2.circle(original, center, radius, (0,255,0), 3)
    cv2.putText(original, "Pothole",
                (center[0]-20, center[1]-radius-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2)

    return original

# ---------- Live Detection ----------
def live_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_potholes(frame)
        cv2.imshow("Live Pothole Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- Upload Detection ----------
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        result = detect_potholes(image)

        cv2.imshow("Image Pothole Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------- GUI ----------
root = Tk()
root.title("Road Defect Detection System")
root.geometry("400x250")
root.configure(bg="white")

Label(root, text="Pothole Detection System",
      font=("Arial", 16, "bold"),
      bg="white").pack(pady=20)

Button(root, text="Live Detection",
       command=live_detection,
       width=20, height=2,
       bg="green", fg="white").pack(pady=10)

Button(root, text="Upload Image",
       command=upload_image,
       width=20, height=2,
       bg="blue", fg="white").pack(pady=10)

root.mainloop()