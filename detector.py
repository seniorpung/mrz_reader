import cv2
import numpy as np


class Detector:
    def __init__(self, rect_x=5, rect_y=13, sq_x=33, sq_y=33):
        self._rect_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (rect_x, rect_y)
        )

        self._sq_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (sq_x, sq_y)
        )

        self._height = 600

    @staticmethod
    def read(src):
        if not src.endswith(('.jpg', '.jpeg', '.png')):
            print('The file should be in image format.')
        image = cv2.imread(src)
        return image


    def resize(self, image):
        """
        Resize image

        :param np.ndarray image: image array
        :return: resized image
        :rtype: np.ndarray
        """
        # get image height and width
        height, width = image.shape[:2]

        # define ratio to resize image
        ratio = self._height / height
        new_width = int(width * ratio)

        # resize image
        return cv2.resize(image, (new_width, self._height))

    def smooth(self, image):
        """
        Convert image to gray scale and smooth image with gaussian blur

        :param np.ndarray image: resized image array
        :return: smoothed image
        :rtype: np.ndarray
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # smooth the image using a 3x3 Gaussian
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def find_dark_regions(self, image):
        """
        Morphological operator to find dark regions on a light background

        :param np.ndarray image: smoothed image array
        :return: blackhat image
        :rtype: np.ndarray
        """
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, self._rect_kernel)

    def apply_threshold(self, image):
        """
        Highlight the mrz code area with closing operations and erosions

        :param np.ndarray image: blackhat image array
        :return: threshold applied image
        :rtype: np.ndarray
        """
        # compute the Scharr gradient
        x_grad = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        x_grad = np.absolute(x_grad)
        min_val, max_val = np.min(x_grad), np.max(x_grad)

        scaled = (x_grad - min_val) / (max_val - min_val)

        # scale the result into the range [0, 255]
        x_grad = (255 * (scaled)).astype('uint8')

        # apply a closing operation using the rectangular kernel
        # to close gaps in between letters
        x_grad = cv2.morphologyEx(x_grad, cv2.MORPH_CLOSE, self._rect_kernel)

        # apply Otsu's thresholding method
        thresh = cv2.threshold(
            x_grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        # another closing operation to close gaps between lines of the MRZ
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._sq_kernel)

        # perform a series of erosions to break apart connected components
        thresh = cv2.erode(thresh, None, iterations=4)

        # set 5% of the left and right borders to zero because of
        # probability border pixels were included in the thresholding
        p_val = int(image.shape[1] * 0.05)
        thresh[:, 0:p_val] = 0
        thresh[:, image.shape[1] - p_val:] = 0

        return thresh

    def find_coordinates(self, im_thresh, im_dark):
        """
        Find coordinates of the mrz code area

        :param np.ndarray im_thresh: threshold applied image array
        :param np.ndarray im_dark: blackhat image array
        :return: coordinates of the mrz code area
        :rtype: tuple[y, y1, x, x1]
        """
        contours = cv2.findContours(
            im_thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[-2]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            # compute the bounding box of the contour
            # compute the aspect ratio and coverage ratio of the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / float(h)
            cr_width = w / float(im_dark.shape[1])

            # check to thresholds for aspect ratio and coverage width
            if aspect > 5 and cr_width > 0.5:
                px = int((x + w) * 0.03)
                py = int((y + h) * 0.03)
                x, y = x - px, y - py
                w, h = w + (px * 2), h + (py * 2)

                break

        return y, y + h + 10, x, x + w + 10

    def crop_area(self, image):
        """
        Crop mrz area from given image

        :param np.ndarray image: image array
        :return: cropped image
        :rtype: np.ndarray
        """
        # resize the image
        resized = self.resize(image)

        # smooth the image
        smoothed = self.smooth(resized)

        # blackhat image
        dark = self.find_dark_regions(smoothed)

        # apply threshold to blackhat image
        thresh = self.apply_threshold(dark)

        # find mrz code area coordinaties
        y, y1, x, x1 = self.find_coordinates(thresh, smoothed)

        return resized[y:y1, x:x1]

    # def resize(self, src):
    #     height, width, _ = src.shape
    #     ratio = self._height / height
    #     new_width = int(width * ratio)
    #     return cv2.resize(src, (new_width, self._height))
    #
    # def find_contours(self, src, origin):
    #     contours = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #
    #     for contour in contours:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(origin, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         aspect = w / float(h)
    #         width_ratio = w / float(src.shape[1])
    #         print(aspect, width_ratio)
    #         # if 5 < aspect < 10 and width_ratio > 0.7:
    #         #     tr_x, tr_y = x - 5, y - 5
    #         #     bl_x, bl_y = x + w + 10, y + h + 10
    #         #     print('width: ', w, ' height: ', h, 'ratio: ', int(w/h))
    #         #     cv2.rectangle(origin, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         #     break
    #     # return tr_x, bl_x, tr_y, bl_y
    #
    #     return origin
    #
    # def find_area(self, src):
    #     resized = self.resize(src)
    #     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #     smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    #
    #     _rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #     morph = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, _rect_kernel)
    #
    #     _kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (30, 30))
    #     crossed = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, _kernel)
    #     thresh = cv2.threshold(crossed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #     return self.find_contours(thresh, resized)
    #     # x, x1, y, y1 = self.find_contours(thresh, resized)
    #     # return resized[y:y1, x:x1]

