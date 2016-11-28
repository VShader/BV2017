#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <memory>

using namespace cv;
using namespace std::string_literals;


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
    auto A = cv::imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe5.jpg");
    auto Orginal = cv::imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe5-Orginal.jpg");
    std::vector<cv::Mat> imgs, imgs2;
    std::string path = "E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe6-";
    for( uint8_t i=1; i<8; ++i)
    {
        imgs.push_back(cv::imread(path + std::to_string(i) + ".jpg"s));
        //imgs.back() = shrink_image(imgs.back());
        //cv::imshow(std::to_string(i), imgs.back());
    }



    cv::Mat pano;
    cv::Stitcher stitcher = cv::Stitcher::createDefault(true);//cv::Stitcher::create(cv::Panorama, true);
    Stitcher::Status status = stitcher.stitch(imgs, pano);
    if (status != Stitcher::OK)
    {
        std::cerr << "Can't stitch images, error code = " << int(status) << std::endl;
        return -1;
    }
    //imwrite(result_name, pano);
    pano = shrink_image(pano);
    imshow("Pano", pano);


    cv::waitKey(0);
    return 0;
}
