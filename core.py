#!/usr/bin/python

from cv2 import *
from optparse import *
import numpy as np
import pylab as plt

class TImage:

    def __init__(self, imagePath):

        self.__image = imread( imagePath, CV_LOAD_IMAGE_GRAYSCALE )

        try:
            if not self.__image.all():
                raise IOError("No image data \n")
        except AttributeError:
            raise IOError("No image data \n")

        self.__channels = lambda: 1 if len(self.__image.shape) == 2  else self.image.shape[2]

    def render(self, window_name) :
        # Define height for showing
        h = 500.0
        scale = h / self.__image.shape[0]
        destroyAllWindows()
        namedWindow(window_name)#, WINDOW_NORMAL)
        imshow(window_name, resize(self.__image, None, fy=scale, fx=scale))
        #resizeWindow(window_name, 500, 500)
        print  "Render" 
        waitKey(0)
    

    def apply_adaptiveThreshold(self) :
        print  "Apply adaptiveThreshold ... "
        self.__image = adaptiveThreshold(self.__image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 10)

        self.render("adaptiveThreshold")
    

    def apply_threshold(self) :
        print  "Apply threshold ... "
        retval, self.__image = threshold( self.__image, 100, 255, THRESH_BINARY_INV | THRESH_OTSU )
        #threshold( self.__image, self.__image, 100, 255, THRESH_OTSU )

        self.render("threshold")
    

    def apply_morphologyEx(self) :
        print  "Apply morphologyEx ... "
        element = getStructuringElement( MORPH_CROSS, (3, 3), (1, 1))
        self.__image = morphologyEx(self.__image,  MORPH_CLOSE, element)

        self.render("morphologyEx")
    

    def apply_fastNlMeansDenoising(self) :
        """No such function in Python"""
#        print  "Use fastNlMeansDenoising ... "
#
#        self.__image = fastNlMeansDenoising(self.__image,
#                            3,# float h=3,
#                            7,# int templateWindowSize=7,
#                            22# int searchWindowSize=21 
#                    )
#        self.render("fastNlMeansDenoising")
    

    def apply_denoise(self):
        print  "Apply denoise ... "
        (contours, hierarchy) =  findContours( self.__image, RETR_CCOMP, CHAIN_APPROX_NONE )

        self.__image = np.ones(self.__image.shape, np.uint8) * 255
        color_parent = [0] * self.__channels()
        color_child = [255] * self.__channels()
        for i in range(len(contours)):
            area = contourArea(contours[i])
            if area > 5 and area < 10000:
                #print hierarchy
                if hierarchy[0,i, 3] == -1:
                    color = color_parent
                else:
                    color = color_child

                #color = Scalar( rand()&255, rand()&255, rand()&255 )
                #drawContours( self.__image, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() )
                drawContours( self.__image, contours, i, color, -1, 8, hierarchy, 0, None )

        self.render("Denoised")
    

    def drawRectOnImage(self,rect = np.array([40, 40, 30, 30])) :
        print  "Draw roi ... "
        pt1 = rect[:2] 
        pt2 = rect[:2] + rect[2:] 
        rectangle(self.__image, tuple(pt1 + np.array([-1, -1])), tuple(pt2 + np.array([1, 1])), [0] * self.__channels())
        rectangle(self.__image, tuple(pt1 + np.array([-3, -3])), tuple(pt2 + np.array([3, 3])), [0] * self.__channels())
        self.render("New rectangle")
    

    def getRoi(self,rect = np.array([40, 40, 30, 30])) :
        #Rect roi((int)(self.__image.size().width/2) - 10, 4, w, h)
        image_roi = self.__image[rect[0]:rect[0]+rect[2], rect[1]:rect[1]+rect[3]]
        return image_roi
    

    def getHist(self, roi, display = False):
        print  "Calculating histogram ... "

        hist, bin_edges  = plt.histogram(roi.reshape(roi.size), bins=2)

        if display :
            plt.bar(bin_edges[:-1], hist, width = 127)
            plt.xlim(np.min(bin_edges), np.max(bin_edges))
            plt.show()
            # add a line showing the expected distribution
            
            #waitKey(0)

        return float (hist[0]) / sum(hist)


def parse_input():
    # Parse input options
    usage = "usage: %prog <Image_Path>"
    parser=OptionParser(usage)
    #parser.add_option('-p','--plot', type='int', default=True, help="Matplotlib")
    (options, args) = parser.parse_args()

    if not args or len(args) > 1:
        raise IOError("usage: DisplayImage.py <Image_Path>\n")
    return args[0]

def main():

    try:

        imagePath = parse_input()
        timage = TImage(imagePath)

        timage.render("Image")
        timage.apply_threshold()
        #timage.apply_morphologyEx()
        timage.apply_denoise()

        rect = np.array([300, 100, 200, 200])
        timage.drawRectOnImage(rect)
        hist = timage.getHist(timage.getRoi(rect))#,True)
        print  "Black percentage: %s %%" % ( hist * 100) 

    except IOError as e:
        print "\nException raised: %s " % (e)


if __name__ == '__main__':
    main()

