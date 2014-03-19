#!/usr/bin/python

import os.path
import optparse

import cv2
import numpy as np



class TImage(object):
    window_height = 800.0

    def __init__(self, path_to_img):
        super(TImage, self).__init__()

        img = cv2.imread(path_to_img, cv2.CV_LOAD_IMAGE_GRAYSCALE)

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

        cv2.destroyAllWindows()
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(0)


class ChainUnit(object):
    __next = None

    def add(self, next):
        self.__next = next
        return self

    def handle(self, img):
        if self.__next:
            self.__next.handle(img)


class Preprossing(ChainUnit):
    threshold = 150

    def apply_blur(self, t):
        print  "Apply blur ... "
        t.img = cv2.blur(t.img, (3,3))

    def apply_threshold(self, t) :
        print  "Apply threshold ... "
        retval, t.img = cv2.threshold( t.img, self.threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )
        #threshold( t.img, t.img, 100, 255, THRESH_OTSU )

    def apply_denoise(self, t):
        print  "Apply denoise ... "
        (contours, hierarchy) =  cv2.findContours( t.img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE )

        t.img = np.ones(t.img.shape, np.uint8) * 255
        color_parent = [0] * t.channels()
        color_child = [255] * t.channels()
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 5 and area < 10000:
                #print hierarchy
                if hierarchy[0,i, 3] == -1:
                    color = color_parent
                else:
                    color = color_child

                #color = Scalar( rand()&255, rand()&255, rand()&255 )
                #drawContours( t.img, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() )
                cv2.drawContours( t.img, contours, i, color, -1, 8, hierarchy, 0, None )

    def apply_morphologyEx(self, t) :
        print  "Apply morphologyEx ... "
        element = cv2.getStructuringElement( cv2.MORPH_CROSS, (3, 3), (1, 1))
        t.img = cv2.morphologyEx(t.img,  cv2.MORPH_CLOSE, element)

    def prepare(self, timage):
        # ToDo(Make chain as in wiki.)
        self.apply_blur(timage)
        self.apply_threshold(timage)
        self.apply_denoise(timage)
        self.apply_morphologyEx(timage)

        timage.render()

    def handle(self, timage):
        print "Preprossing..."
        self.prepare(timage)
        super(Preprossing, self).handle(timage)


class Extraction(ChainUnit):

    def handle(self, timage):
        print "Extraction"
        super(Extraction, self).handle(timage)


class Recognition(ChainUnit):

    def handle(self, timage):
        print "Recognition"
        super(Recognition, self).handle(timage)


def parse_input():
    usage = "usage: %prog [options] arg1"
    parser = optparse.OptionParser(usage)
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
    opts = parse_input()

    timage = TImage(opts.path_to_image)

    chain = Preprossing().add(Extraction().add(Recognition()))
    chain.handle(timage)
