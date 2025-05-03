import cv2
import sys

# Use OpenCV's built-in haarcascades directory
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"ERROR: Could not load cascade at {cascade_path}")
    sys.exit(1)

# Load the image passed as first command-line argument
if len(sys.argv) < 2:
    print("Usage: python detect_faces.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Could not read image '{img_path}'")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Annotate each face with a unique ID
for idx, (x, y, w, h) in enumerate(faces, start=1):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img,
        f'ID {idx}',
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

# Show result
cv2.imshow('Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
