#!/usr/bin/python

from cv2 import *
from optparse import *
import numpy as np
import pylab as plt

import os.path


class ImageBase(object):
    img = None
    __ref_img = None

    window_height = 800.0


    def __init__(self, path_to_img):
        super(ImageBase, self).__init__()

        img = imread(path_to_img, CV_LOAD_IMAGE_GRAYSCALE)

        try:
            if not img.all():
                raise IOError("No image data \n")
        except AttributeError:
            raise IOError("No image data \n")

        self.img = img
        self.__ref_img = img
        self.channels = lambda: 1 if len(self.img.shape) == 2 else self.img.shape[2]

    def render(self, window_name="main", img = None):
        if not img:
            img = self.img

        scale = self.window_height / img.shape[0]

        destroyAllWindows()
        namedWindow(window_name)
        imshow(window_name, resize(img, None, fx=0.3, fy=0.3, interpolation=INTER_CUBIC))
        waitKey(0)


class PreprocessImage(ImageBase):

    __threshold = 150

    def __init__(self, path_to_img):
        super(PreprocessImage, self).__init__(path_to_img)

        self.apply_blur()
        self.apply_threshold()
        self.apply_denoise()
        self.apply_morphologyEx()

    def apply_adaptiveThreshold(self) :
        print  "Apply adaptiveThreshold ... "
        self.img = adaptiveThreshold(self.img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 10)

    def apply_blur(self):
        print  "Apply blur ... "
        self.img = blur(self.img, (3,3))

    def apply_threshold(self) :
        print  "Apply threshold ... "
        retval, self.img = threshold( self.img, self.__threshold, 255, THRESH_BINARY_INV | THRESH_OTSU )
        #threshold( self.img, self.img, 100, 255, THRESH_OTSU )

    def apply_denoise(self):
        print  "Apply denoise ... "
        (contours, hierarchy) =  findContours( self.img, RETR_CCOMP, CHAIN_APPROX_NONE )

        self.img = np.ones(self.img.shape, np.uint8) * 255
        color_parent = [0] * self.channels()
        color_child = [255] * self.channels()
        for i in range(len(contours)):
            area = contourArea(contours[i])
            if area > 5 and area < 10000:
                #print hierarchy
                if hierarchy[0,i, 3] == -1:
                    color = color_parent
                else:
                    color = color_child

                #color = Scalar( rand()&255, rand()&255, rand()&255 )
                #drawContours( self.img, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() )
                drawContours( self.img, contours, i, color, -1, 8, hierarchy, 0, None )

    def apply_morphologyEx(self) :
        print  "Apply morphologyEx ... "
        element = getStructuringElement( MORPH_CROSS, (3, 3), (1, 1))
        self.img = morphologyEx(self.img,  MORPH_CLOSE, element)
    

def parse_input():
    usage = "usage: %prog [options] arg1"
    parser = OptionParser(usage)
    parser.add_option("-i", "--image", dest="path_to_image",
                  help="Path to image.", metavar="path_to_image")

    (options, args) = parser.parse_args()

    if (options.path_to_image is None):
        parser.print_help()
        exit(-1)
    elif not os.path.exists(options.path_to_image):
        parser.error('image does not exists')

    return options



if __name__ == '__main__':
    opt = parse_input()

    timage = PreprocessImage(opt.path_to_image)
    timage.render()


