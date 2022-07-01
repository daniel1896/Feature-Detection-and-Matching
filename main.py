#!/usr/bin/env python
# coding: utf-8

# Feature Description and Matching

# Imports
import argparse
import cv2 as cv

from feature_matcher import FeatureMatcher


def parse_args():
    # Message from usage
    message = """main.py [-h]

                 --detector     {SIFT, SURF, KAZE, ORB, BRISK, AKAZE}
                 --descriptor   {SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK}
                 --matcher      {BF, FLANN}
                 --all          {run all combinations}"""
    # Create the parser
    parser = argparse.ArgumentParser(description='Feature Description and Matching.',
                                     usage=message)
    # Argument --detector
    parser.add_argument('--detector',
                        action='store',
                        choices=['SIFT', 'SURF', 'KAZE', 'ORB', 'BRISK', 'AKAZE'],
                        required=False,
                        metavar='',
                        dest='detector',
                        help='select the detector to be used in this experiment')
    # Argument --descriptor
    parser.add_argument('--descriptor',
                        action='store',
                        choices=['SIFT', 'SURF', 'KAZE', 'BRIEF', 'ORB', 'BRISK', 'AKAZE', 'FREAK'],
                        required=False,
                        metavar='',
                        dest='descriptor',
                        help='select the descriptor to be used in this experiment')
    # Argument --matcher
    parser.add_argument('--matcher',
                        action='store',
                        choices=['BF', 'FLANN'],
                        required=False,
                        metavar='',
                        dest='matcher',
                        help='select the matcher to be used in this experiment')
    # Argument --all
    parser.add_argument('--all', action='store_true', required=False, dest='all', help='run all combinations')
    # TODO: for Python 3.9+, use the following:
    # parser.add_argument('--all', action=argparse.BooleanOptionalAction, required=False, dest='all', help='...')

    # Parse the arguments
    args = parser.parse_args()

    # if either --detector or --descriptor or --matcher is not specified, then run all combinations
    if not(args.detector and args.descriptor and args.matcher):
        args.all = True

    # return arguments
    return args


def readImages(path1=None, path2=None):
    if path1 or path2 is None:
        # open file dialog
        import tkinter as tk
        from tkinter import filedialog
        if path1 is None:
            tk.Tk().withdraw()
            path1 = filedialog.askopenfilename()
        if path2 is None:
            tk.Tk().withdraw()
            path2 = filedialog.askopenfilename()

    # Read the images
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    if img1 is None:
        print('Error: image 1 not found at {}'.format(path1))
        exit(0)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
    if img2 is None:
        print('Error: image 2 not found at {}'.format(path2))
        exit(0)

    return img1, img2


if __name__ == '__main__':
    # Parse the arguments
    arguments = parse_args()

    # read images
    image1, image2 = readImages()

    if not arguments.all:
        print('Running {} {} {}'.format(arguments.detector, arguments.descriptor, arguments.matcher))
        # measure time
        start = cv.getTickCount()
        matcher = FeatureMatcher(arguments.detector, arguments.descriptor, arguments.matcher)
        matches = matcher.match(image1, image2)
        matcher.save_matches()
        end = cv.getTickCount()
        time = (end - start) / cv.getTickFrequency()
        print('Time: {}'.format(time))
        print('Matches: {}'.format(len(matches)))
    else:
        # Run all combinations
        detectors = ['SIFT', 'SURF', 'KAZE', 'ORB', 'BRISK', 'AKAZE']
        descriptors = ['SIFT', 'SURF', 'KAZE', 'BRIEF', 'ORB', 'BRISK', 'AKAZE', 'FREAK']
        matcher_types = ['BF', 'FLANN']
        for matcher_type in matcher_types:
            for detector in detectors:
                for descriptor in descriptors:
                    print('--------------------------')
                    try:
                        print('Running {} {} {}'.format(detector, descriptor, matcher_type))
                        # measure time
                        start = cv.getTickCount()
                        matcher = FeatureMatcher(detector, descriptor, matcher_type)
                        matches = matcher.match(image1, image2)
                        matcher.save_matches()
                        end = cv.getTickCount()
                        time = (end - start) / cv.getTickFrequency()
                        print('Time: {}'.format(time))
                        print('Matches: {}'.format(len(matches)))
                    except Exception as e:
                        print("Some combinations are not supported and fail.")
                        print('Error: {}'.format(e))
                        continue
