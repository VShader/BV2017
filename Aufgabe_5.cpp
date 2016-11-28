//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/features2d.hpp>
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;


cv::Mat shrink_image(cv::Mat& input)
{
    cv::Mat output;
    auto max = std::max(input.rows, input.cols);
    auto shrinkValue = max > 512 ? 512.0f/max : 1;
    cv::resize(input, output, cv::Size( input.cols*shrinkValue, input.rows*shrinkValue ));
    return output;
}


int main()
{

    constexpr float inlier_threshold = 2.5f; // Distance threshold to identify inliers
    constexpr float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

    auto A = cv::imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe5.jpg");
    auto Orginal = cv::imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe5-Orginal.jpg");

//    cv::cvtColor(A, A, CV_BGR2GRAY);
//    cv::cvtColor(Orginal, Orginal, CV_BGR2GRAY);


//    Mat dstA, dstB;
//    dstA = Mat::zeros( A.size(), CV_32FC1 );
//    dstB = Mat::zeros( Orginal.size(), CV_32FC1 );
//    int blockSize = 2;
//    int apertureSize = 3;
//    double k = 0.04;
//    cornerHarris( A, dstA, blockSize, apertureSize, k, BORDER_DEFAULT );
//    cornerHarris( Orginal, dstB, blockSize, apertureSize, k, BORDER_DEFAULT );
//    //auto Trans = cv::getAffineTransform(dstA, dstB);
//    cv::imshow("asd", dstB);

//    //cv::warpAffine(A, A, Trans);

//    A = shrink_image(A);
//    Orginal = shrink_image(Orginal);
//    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
//    cv::imshow("Display Image", A);
//    cv::imshow("Orginal Image", Orginal);
    Mat img1 = imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe5.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe5-Orginal.jpg", IMREAD_GRAYSCALE);
    Mat homography;
    homography.convertTo(homography, CV_64FC1);
    FileStorage fs("../data/H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64FC1);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("res.png", res);
    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "A-KAZE Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;
    return 0;

    cv::waitKey(0);
    return 0;
}
