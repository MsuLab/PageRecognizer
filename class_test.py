#!/usr/bin/python

import os.path
import optparse

from preco.TImage import TImage

from classifier import classify


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
    z = [px / 255.0 for px in timage.img.flatten('C').tolist()]
    print classify(z)
