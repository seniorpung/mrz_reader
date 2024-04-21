import json
import pytesseract
import cv2
from detector import Detector
from reader import Reader
# from readmrz import MrzDetector, MrzReader

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
imgPath = 'images/3.png'

detector = Detector()
reader = Reader()

image = detector.read(imgPath)
cropped = detector.crop_area(image)
result = reader.process(cropped)
print(json.dumps(result))

# cv2.imshow('Cropped Image', cropped)
# cv2.waitKey(0)