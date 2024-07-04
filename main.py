import cv2
from pytesseract import pytesseract
import numpy as np

image = cv2.imread("image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

blank = np.zeros(image.shape, np.uint8)
drawn_on = blank

# Detect lines
linesP = cv2.HoughLinesP(
    wide,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=70,
    maxLineGap=40,
)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(drawn_on, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2, cv2.LINE_AA)


# Detect circles
circles = cv2.HoughCircles(
    tight,
    cv2.HOUGH_GRADIENT,
    1,
    image.shape[0] / 8,
    param1=100,
    param2=30,
    minRadius=1,
    maxRadius=100,
)

# Get nodes
letters = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(drawn_on, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(drawn_on, center, radius, (255, 0, 255), 3)

        # cut out letter
        mask = cv2.circle(
            np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8),
            center,
            radius,
            255,
            -1,
        )
        letter = cv2.copyTo(image, mask)

        # only keep red content
        letter = cv2.inRange(letter, (0, 0, 100), (100, 100, 255))

        letter = pytesseract.image_to_string(letter, config=r"--oem 3 --psm 6")
        letters.append(letter)


print("Nodes found", letters)
cv2.imshow("original", image)
cv2.imshow("image", drawn_on)
cv2.waitKey(000)
