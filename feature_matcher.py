#!/usr/bin/env python
# coding: utf-8

# Feature Description and Matching

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class FeatureMatcher:
    BF_LIMIT = 20           # Limits the number of matches to be displayed
    FLANN_INDEX_KDTREE = 1  # FLANN INDEX KDTREE parameter
    FLANN_RATIO_TH = 0.7    # Limits the number of matches to be displayed
    FLANN_CHECKS = 50       # higher: more accurate, but slower

    def __init__(self, detector_name, descriptor_name, matcher_name):
        """
        :param detector_name: (SIFT, SURF, KAZE, ORB, BRISK, AKAZE)
        :param descriptor_name: (SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK)
        :param matcher_name: (BF, FLANN)

        Attributes:
            self.detector: detector object (SIFT, SURF, KAZE, ORB, BRISK, AKAZE)
            self.descriptor: descriptor object (SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK)
            self.matcher: matcher (Brute Force Matcher or FlannBasedMatcher)
        """
        # names
        self.detector_name = detector_name
        self.descriptor_name = descriptor_name
        self.matcher_name = matcher_name
        # objects
        self.detector = self.get_detector()
        self.descriptor = self.get_descriptor()
        # resulting matches
        self.matches = []
        self.matches_img = None
        self.time = -1.0

    def get_detector(self):
        """
        Uses the detector name to return the detector object
        :return: detector object (SIFT, SURF, KAZE, ORB, BRISK, AKAZE)
        """
        if self.detector_name == 'SIFT':
            sift = cv.xfeatures2d.SIFT_create()
            return sift
        elif self.detector_name == 'SURF':
            surf = cv.xfeatures2d.SURF_create()
            return surf
        elif self.detector_name == 'KAZE':
            kaze = cv.KAZE_create()
            return kaze
        elif self.detector_name == 'ORB':
            orb = cv.ORB_create()
            return orb
        elif self.detector_name == 'BRISK':
            brisk = cv.BRISK_create()
            return brisk
        elif self.detector_name == 'AKAZE':
            akaze = cv.AKAZE_create()
            return akaze
        else:
            raise ValueError('Invalid detector name')

    def get_descriptor(self):
        """
        Uses the descriptor name to return the descriptor object
        :return: descriptor object (SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK)
        """
        if self.descriptor_name == 'SIFT':
            sift = cv.xfeatures2d.SIFT_create()
            return sift
        elif self.descriptor_name == 'SURF':
            surf = cv.xfeatures2d.SURF_create()
            return surf
        elif self.descriptor_name == 'KAZE':
            kaze = cv.KAZE_create()
            return kaze
        elif self.descriptor_name == 'BRIEF':
            brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
            return brief
        elif self.descriptor_name == 'ORB':
            orb = cv.ORB_create()
            return orb
        elif self.descriptor_name == 'BRISK':
            brisk = cv.BRISK_create()
            return brisk
        elif self.descriptor_name == 'AKAZE':
            akaze = cv.AKAZE_create()
            return akaze
        elif self.descriptor_name == 'FREAK':
            freak = cv.xfeatures2d.FREAK_create()
            return freak
        else:
            raise ValueError('Invalid descriptor name')

    def get_features(self, image):
        """
        :param image: input image
        :return: keypoints, descriptors
        """
        # detect keypoints
        keypoints = self.detector.detect(image)
        # compute descriptors
        descriptors = self.descriptor.compute(image, keypoints)[1]
        return keypoints, descriptors

    def match(self, image1, image2):
        """
        :param image1: input image 1
        :param image2: input image 2
        :return: matches_img
        """
        # measure time
        start = cv.getTickCount()

        # get features
        keypoints1, descriptors1 = self.get_features(image1)
        keypoints2, descriptors2 = self.get_features(image2)

        # match features
        if self.matcher_name == 'BF':
            # if descriptor_name is SIFT, SURF or KAZE, set normType=cv.NORM_L2
            if self.descriptor_name in ['SIFT', 'SURF', 'KAZE']:
                normType = cv.NORM_L2
            else:
                normType = cv.NORM_HAMMING

            # brute force matching
            bf = cv.BFMatcher_create(normType=normType, crossCheck=True)  # TODO: why crossCheck=True?
            matches_all = bf.match(descriptors1, descriptors2)
            # sort matches_img in the order of their distance
            matches_all = sorted(matches_all, key=lambda x: x.distance)
            self.matches = matches_all[:self.BF_LIMIT]
            # return the best matches_img
            self.matches_img = cv.drawMatches(image1, keypoints1, image2, keypoints2, self.matches,
                                              outImg=None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        elif self.matcher_name == 'FLANN':
            # FLANN parameters
            index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=self.FLANN_CHECKS)

            # flann matching
            flann = cv.FlannBasedMatcher(index_params, search_params)
            # matching descriptor vectors using FLANN Matcher
            matches_all = flann.knnMatch(np.float32(descriptors1), np.float32(descriptors2), k=2)

            # Lowe's ratio test to filter matches_img
            self.matches = []
            for m, n in matches_all:
                if m.distance < self.FLANN_RATIO_TH * n.distance:
                    self.matches.append(m)

            # draw good matches_img
            self.matches_img = cv.drawMatches(image1, keypoints1, image2, keypoints2, self.matches,
                                              outImg=None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        else:
            raise ValueError('Invalid matcher name')

        # measure time
        end = cv.getTickCount()
        self.time = (end - start) / cv.getTickFrequency()
        print('Time: %.2fs' % self.time)
        print('Matches: {}'.format(len(self.matches)))

        return self.matches_img

    # Call function saveMatcher
    def plot_matches(self):
        """
        Shows the matches_img by plotting them on the images and saving them in the Results folder
        :param matches_img: input matches_img (optional: if not provided, use self.matches_img)
        """
        # Create a new figure
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.axis('off')
        ax.margins(0)

        # Plot the images
        ax.imshow(self.matches_img)
        ax.title.set_text('Matcher: ' + self.matcher_name +
                          ',  Detector: ' + self.detector_name +
                          ',  Descriptor: ' + self.descriptor_name)
        # put the time in the bottom right corner and the number of matches in the bottom left corner
        ax.text(0.99, 0.01, 'Time: %.2fs' % self.time, color='orange',
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=10)
        ax.text(0.01, 0.01, 'Matches: {}'.format(len(self.matches)), color='orange',
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=10)

        # save and show the figure
        fig.tight_layout()
        plt.savefig('Results/%s-with-%s-%s.png' % (self.matcher_name, self.detector_name, self.descriptor_name),
                    bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
