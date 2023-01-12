import cv2
import os

if not os.path.exists('./test_gray_images'):
    os.mkdir('./test_gray_images')

for imagename in os.listdir('./test_images'):
    if imagename.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
        imagepath = './test_images/' + imagename
        image = cv2.imread(imagepath)
        image_red = image[:, :, 2]
        image_red = cv2.resize(image_red, (320, 240), interpolation=cv2.INTER_LINEAR_EXACT)
        cv2.imwrite('./test_gray_images/'+imagename, image_red)

