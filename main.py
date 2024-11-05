import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()

# Load background images
img_dir = "Images"
if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Directory '{img_dir}' not found.")

listImg = os.listdir(img_dir)
imgList = [cv2.imread(f'{img_dir}/{imgPath}') for imgPath in listImg if cv2.imread(f'{img_dir}/{imgPath}') is not None]
if not imgList:
    raise ValueError("No valid images found in the directory.")

indexImg = 0
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam. Exiting.")
        break

    imgOut = segmentor.removeBG(img, imgList[indexImg], cutThreshold=0.6)
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    cv2.imshow("image", imgStacked)

    key = cv2.waitKey(20)
    if key == ord('a'):
        indexImg = (indexImg - 1) % len(imgList)  # Cycle backwards
    elif key == ord('d'):
        indexImg = (indexImg + 1) % len(imgList)  # Cycle forwards
    elif key == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()