 #include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define PI 3.14159265

using namespace cv;
using namespace std;


void rotate(cv::Mat& originalImage,cv::Mat& rotatedImage,cv::InputArray rotated,
cv::Mat& dst) {
    std::vector<cv::Point2f> original(4);
    original[0] = cv::Point( 0, 0);
    original[1] = cv::Point( originalImage.cols, 0 );
    original[2] = cv::Point( originalImage.cols, originalImage.rows );
    original[3] = cv::Point( 0, originalImage.rows );

    dst = cv::Mat::zeros(originalImage.rows, originalImage.cols, CV_8UC3);
    cv::Mat transform = cv::getPerspectiveTransform(rotated, original);
    cv::warpPerspective(rotatedImage, dst, transform, dst.size() );
}



float angleBetween(const Point &v1, const Point &v2)
{
    float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);

    float dot = v1.x * v2.x + v1.y * v2.y;

    float a = dot / (len1 * len2);

    if (a >= 1.0)
        return 0.0;
    else if (a <= -1.0)
        return PI;
    else{
        int degree;
        degree = acos(a)*180/PI;
        return degree;
        };
}



int main()
{

    Mat char1 = imread( "/Users/Rodrane/Documents/XCODE/OpenCV/mkedenemeleri/anarev/rotated.jpg",CV_LOAD_IMAGE_GRAYSCALE );

    Mat image = imread("/Users/Rodrane/Documents/XCODE/OpenCV/mkedenemeleri/anarev/gain2000_crop.jpg", CV_LOAD_IMAGE_GRAYSCALE );




    if( !char1.data )
    {
        std::cout<< "Error reading object " << std::endl;
        return -1;
    }

    GaussianBlur( char1, char1, Size(3, 3), 2, 2 );
    GaussianBlur( image, image, Size(3, 3), 2, 2 );
    adaptiveThreshold(char1,char1,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,9,14);
    adaptiveThreshold(image,image,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,9,14);

    //Detect the keypoints using SURF Detector
    int minHessian = 200;

    SurfFeatureDetector detector( minHessian );
    std::vector<KeyPoint> kp_object;

    detector.detect( char1, kp_object );

    //Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    Mat des_object;

    extractor.compute( char1, kp_object, des_object );

    FlannBasedMatcher matcher;


    namedWindow("Good Matches");

    std::vector<Point2f> obj_corners(4);

    //Get the corners from the object
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( char1.cols, 0 );
    obj_corners[2] = cvPoint( char1.cols, char1.rows );
    obj_corners[3] = cvPoint( 0, char1.rows );



    Mat frame;




    Mat des_image, img_matches;
    std::vector<KeyPoint> kp_image;
    std::vector<vector<DMatch > > matches;
    std::vector<DMatch > good_matches;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    std::vector<Point2f> scene_corners(4);
    Mat H;


    detector.detect( image, kp_image );
    extractor.compute( image, kp_image, des_image );

    matcher.knnMatch(des_object, des_image, matches, 2);

    for(int i = 0; i < min(des_image.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
    {
        if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
        {
            good_matches.push_back(matches[i][0]);
        }
    }



    //Draw only "good" matches


    drawMatches( char1, kp_object, image, kp_image, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    if (good_matches.size() >= 4)
    {
        for( int i = 0; i < good_matches.size(); i++ )
        {
            //Get the keypoints from the good matches
            obj.push_back( kp_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
            cout<<angleBetween(obj[i],scene[i])<<endl; //angles between images

        }

        H = findHomography( obj, scene, CV_RANSAC );


        perspectiveTransform( obj_corners, scene_corners, H);

       // cout<<angleBetween(obj[0], scene[0])<<endl;


        //Draw lines between the corners (the mapped object in the scene image )

    }

    //Show detected matches
    // resize(img_matches, img_matches, Size(img_matches.cols/2, img_matches.rows/2));

    imshow( "Good Matches", img_matches );
    waitKey();

    return 0;
}
