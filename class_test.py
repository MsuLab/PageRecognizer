#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path
import optparse
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


from preco.TImage import TImage

from classifier import classify


if __name__ == '__main__':
    files_list = ['1-2.jpg', '3-1.jpg', '5-1.jpg', '5-2.jpg', '7-1.jpg', '8-1.jpg', '8-2.jpg']
    for file in files_list:
        timage = TImage(file)
        cv2.imshow('image', timage.img)
        cv2.waitKey(0)
        #img = timage.img
        #img = cv2.resize(timage.img, (28, 28));
        z = [px / 255.0 for px in cv2.resize(timage.img, (28, 28)).flatten('C').tolist()]
        probs = classify(z)
        max_value = max(probs)
        #print probs
        img = Image.open(file)
        draw = ImageDraw.Draw(img)
        
        if max_value < 0.5:
            s = 'ERROR'
        else:
            s = 'DIGIT: %i' % probs.index(max_value)
        print s
        font = ImageFont.truetype("ARIAL.TTF", 20)
        draw.text((0, 0), s, (0, 255, 0), font=font)
        img.save('sample-out.jpg')
        cv2.imshow('image', TImage('sample-out.jpg').img)
        cv2.waitKey(0)
