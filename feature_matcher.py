#!/usr/bin/env python
# coding: utf-8

# Feature Description and Matching

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from copy import copy


class FeatureMatcher:
    BF_LIMIT = 20           # Limits the number of matches to be displayed
    FLANN_INDEX_KDTREE = 1  # FLANN INDEX KDTREE parameter
    FLANN_RATIO_TH = 0.7    # Limits the number of matches to be displayed
    FLANN_CHECKS = 50       # higher: more accurate, but slower
    HOMOGRAPHY_MATCH_TH = 10  # number of matches that are necessary for homography
    HOMOGRAPHY_RANSAC_TH = 5  # Maximum reprojection error in the RANSAC algorithm to consider a point as an inlier.

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
        # images to match
        self.img1 = None
        self.img2 = None
        # keypoints and descriptors
        self.kpt1, self.des1 = None, None
        self.kpt2, self.des2 = None, None
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
        # detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match(self, image1, image2, equalize=False):
        """
        :param image1: input image 1
        :param image2: input image 2
        :param equalize: if True, equalizes the images before matching
        :return: matches_img
        """
        # measure time
        start = cv.getTickCount()

        # histogram equalization
        if equalize:
            image1 = cv.equalizeHist(image1)
            image2 = cv.equalizeHist(image2)

        self.img1 = copy(image1)
        self.img2 = copy(image2)

        # get features
        self.kpt1, self.des1 = self.get_features(self.img1)
        self.kpt2, self.des2 = self.get_features(self.img2)

        # match features
        if self.matcher_name == 'BF':
            # if descriptor_name is SIFT, SURF or KAZE, set normType=cv.NORM_L2
            if self.descriptor_name in ['SIFT', 'SURF', 'KAZE']:
                normType = cv.NORM_L2
            else:
                normType = cv.NORM_HAMMING

            # brute force matching
            bf = cv.BFMatcher_create(normType=normType, crossCheck=True)  # TODO: why crossCheck=True?
            matches_all = bf.match(self.des1, self.des2)
            # sort matches_img in the order of their distance
            matches_all = sorted(matches_all, key=lambda x: x.distance)
            self.matches = matches_all[:self.BF_LIMIT]

        elif self.matcher_name == 'FLANN':
            # FLANN parameters
            index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=self.FLANN_CHECKS)

            # flann matching
            flann = cv.FlannBasedMatcher(index_params, search_params)
            # matching descriptor vectors using FLANN Matcher
            matches_all = flann.knnMatch(np.float32(self.des1), np.float32(self.des2), k=2)

            # Lowe's ratio test to filter matches_img
            self.matches = []
            for m, n in matches_all:
                if m.distance < self.FLANN_RATIO_TH * n.distance:
                    self.matches.append(m)

        else:
            raise ValueError('Invalid matcher name')

        # measure time
        end = cv.getTickCount()
        self.time = (end - start) / cv.getTickFrequency()
        print('Time: %.2fs' % self.time)
        print('Matches: {}'.format(len(self.matches)))

        return self.matches

    def homography(self, plot=True):
        if len(self.matches) > self.HOMOGRAPHY_MATCH_TH:
            # get the points
            src_pts = np.float32([self.kpt1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kpt2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            # compute the homography matrix
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, self.HOMOGRAPHY_RANSAC_TH)
            matchesMask = mask.ravel().tolist()

            h, w = self.img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            if plot:
                self.img2 = cv.polylines(self.img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

            return M, dst, matchesMask

        else:
            print("Not enough matches are found - {}/{}".format(len(self.matches), self.HOMOGRAPHY_MATCH_TH))
            return None, None, None

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
        M, dst, matchesMask = self.homography()
        # draw good matches_img
        self.matches_img = cv.drawMatches(self.img1, self.kpt1, self.img2, self.kpt2, self.matches,
                                          matchesMask=matchesMask, outImg=None,
                                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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
