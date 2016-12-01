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


    A = shrink_image(A);
    Orginal = shrink_image(Orginal);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", A);
    cv::imshow("Orginal Image", Orginal);

    vector<KeyPoint> orginalKeypoints, rotatedKeypoints;
    Ptr<ORB> orb = ORB::create();

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    orb->detectAndCompute( Orginal, Mat(), keypoints_1, descriptors_1 );
    orb->detectAndCompute( A, Mat(), keypoints_2, descriptors_2 );


    vector<DMatch> inliner_matches;
    BFMatcher matcher(NORM_L2);
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //matcher->knnMatch(orginalKeypoints, rotatedKeypoints, inliner_matches, 2);
    matcher.match(descriptors_1, descriptors_2, inliner_matches);

    // sort matches
    std::map<float, cv::DMatch&> distanceMap;
    for(auto& match : inliner_matches)
    {
        distanceMap.insert(std::make_pair(match.distance, std::ref(match)));
    }

    cout << inliner_matches.begin()->distance << " " << (--inliner_matches.end())->distance << endl;
    cout << distanceMap.begin()->first << " " << (--distanceMap.end())->first;

    std::array<cv::Point2f, 4> orginalQuad;
    std::array<cv::Point2f, 4> rotatedQuad;
    auto itMap = distanceMap.begin();
    auto itOrginal = orginalQuad.begin();
    auto itRotated = rotatedQuad.begin();
    for(uint8_t count = 0; count < 4; ++count)
    {
        *itOrginal = keypoints_1[itMap->second.queryIdx].pt;
        *itRotated = keypoints_2[itMap->second.trainIdx].pt;
        ++itMap;
        ++itOrginal;
        ++itRotated;
    }


    cv::Mat Result;
    auto lambda = getAffineTransform(rotatedQuad.data(), orginalQuad.data());
    warpAffine(A, Result, lambda, A.size());

    cv::imshow("Result", Result);


    cv::waitKey(0);
    return 0;
}
