# Feature Detection and Matching

[comment]: <> (![GitHub language count]&#40;https://img.shields.io/github/languages/count/whoisraibolt/Feature-Detection-and-Matching&#41;)
[comment]: <> (![GitHub top language]&#40;https://img.shields.io/github/languages/top/whoisraibolt/Feature-Detection-and-Matching&#41;)
[comment]: <> (![GitHub repo size]&#40;https://img.shields.io/github/repo-size/whoisraibolt/Feature-Detection-and-Matching&#41;)
[comment]: <> (![GitHub]&#40;https://img.shields.io/github/license/whoisraibolt/Feature-Detection-and-Matching&#41;)

Feature Detection and Matching between two images using Local Feature Descriptors and Local Binary Descriptors through the Brute Force and FLANN algorithms.

From this application it is possible to solve several problems in the area of Computer Vision, such as: image recovery, motion tracking, motion structure detection, object detection, recognition and tracking, 3D object reconstruction, and others.

## Overview

This project performs Feature Detection and Matching with SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE and FREAK through the Brute Force and FLANN algorithms using OpenCV (version 3.4.2). You can use different Versions, but it cannot be guaranteed that all feature matching approaches will work. SURF for example is not available for OpenCV versions higher than 3.4.2.

![Feature Detection and Matching with KAZE through the Brute Force algorithm](https://raw.githubusercontent.com/daniel1896/Feature-Detection-and-Matching/master/Results/FLANN-with-SURF-SURF.png)

## Dependencies

To install the dependencies run:

`pip install -r requirements.txt`

## Usage

`python main.py --detector <detector> --descriptor <descriptor> --matcher <matcher>`

| Arguments     | Info                                                                    |
| :------------ | :---------------------------------------------------------------------- |
| `-h`, `--help`| Show help message and exit                                              |
| `--detector`  | Specify SIFT or SURF or KAZE or ORB or BRISK or AKAZE                   |
| `--descriptor`| Specify SIFT or SURF or KAZE or BRIEF or ORB or BRISK or AKAZE or FREAK |
| `--matcher `  | Specify BF or FLANN                                                     |
| `--all`       | Run all combinations of detector, descriptor and matcher             |

If no arguments are given or one of the arguments (detector, descriptor or matcher) is missing, it defaults to run all combinations.

## Examples

####  Help
`python main.py --help`

#### Brute Force with ORB
`python main.py --detector ORB --descriptor ORB --matcher BF`

#### Run all combinations
`python main.py` </br>
`python main.py --all`

## Recommended Readings
- [Feature Detection and Description](https://github.com/whoisraibolt/Feature-Detection-and-Description "Feature Detection and Description")

- KRIG, Scott. [Computer vision metrics: Survey, taxonomy, and analysis](https://link.springer.com/content/pdf/10.1007%2F978-1-4302-5930-5.pdf "Computer vision metrics: Survey, taxonomy, and analysis"). Apress, 2014.

- [Keypoints and Descriptors](https://www.cs.utah.edu/~srikumar/cv_spring2017_files/Keypoints&Descriptors.pdf "Keypoints and Descriptors")

- [Feature Detection and Matching](https://www.comp.nus.edu.sg/~cs4243/lecture/feature.pdf "Feature Detection and Matching")

## License

Code released under the [MIT](https://github.com/whoisraibolt/Feature-Detection-and-Matching/blob/master/LICENSE "MIT") license.
