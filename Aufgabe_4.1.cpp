#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>

cv::Mat shrink_image(cv::Mat& input)
{
    cv::Mat output;
    auto max = std::max(input.rows, input.cols);
    auto shrinkValue = max > 512 ? 512.0f/max : 1;
    cv::resize(input, output, cv::Size( input.cols*shrinkValue, input.rows*shrinkValue ));
    return output;
}

cv::Mat gammaCorrection(cv::Mat& input, const double gamma)
{
    cv::Mat output(input.clone());
    output.convertTo(output, CV_64F);
    constexpr uint8_t wmin = 0;
    constexpr uint8_t wmax = 255;

    double gmin = 255;
    double gmax = 0;
    for (int colIt = 0; colIt < output.cols; colIt++)
    {
        for (int rowIt = 0; rowIt < output.rows; rowIt++)
        {
            auto& pixel = output.at<double>(rowIt, colIt);
            gmin = pixel < gmin ? pixel : gmin;
            gmax = pixel > gmax ? pixel : gmax;
        }
    }

    for (int colIt = 0; colIt < output.cols; colIt++)
    {
        for (int rowIt = 0; rowIt < output.rows; rowIt++)
        {
            auto& pixel = output.at<double>(rowIt, colIt);

            double tmpVal = (wmax - wmin) * std::pow((double)(pixel - gmin)/(double)(gmax - gmin), gamma) + wmin;
            pixel = tmpVal;
            //pixel = gammaCorrection(pixel, 0, 255, wmin, wmax, gamma);
        }
    }

    double min, max;
    cv::minMaxIdx(output, &min, &max);
    cv::convertScaleAbs(output, output, 255 / max);
    return output;
}

cv::Mat histogramStreching(cv::Mat& input)
{
    return gammaCorrection(input, 1.0);
}




double conv(const cv::Mat& ROI, const cv::Mat& filter)
{
    double output = 0;
    for (int col = 0; col < ROI.cols; col++)
    {
        for (int row = 0; row < ROI.rows; row++)
        {
            //double a = ROI.at<double>(row, col);
            //double b = filter.at<double>(filter.rows - 1 - row, filter.cols - 1 - col);
            output += ROI.at<double>(row, col) * filter.at<double>(filter.rows - 1 - row, filter.cols - 1 - col);
        }
    }
    return output;
}

cv::Mat convolution(const cv::Mat& input, const cv::Mat& filter)
{
    cv::Mat output;
    cv::Mat zero(input.rows + 2, input.cols + 2, input.type(), cv::Scalar(0)); // Zero Mat
    input.copyTo(zero(cv::Rect(1, 1, input.cols , input.rows))); // Place image inside this one
    zero.convertTo(zero, CV_64F);
    input.convertTo(output, CV_64F);

//    //BVMat output = this->clone();
//    //cv::Mat output = cv::Mat::zeros(this->rows + 1, this->cols + 1, this->type);

//    // 3. rest

    for (int i = 1; i < zero.cols - 1; i++)
    {
        for (int j = 1; j < zero.rows - 1; j++)
        {
            cv::Mat roi(zero, cv::Rect(i -1, j -1, 3, 3));
            output.at<double>(j-1, i-1) = conv(roi, filter);
        }
    }




//    double min, max;
//    cv::minMaxIdx(output, &min, &max);
//    cv::convertScaleAbs(output, output, 255 / max);
    output = histogramStreching(output);
    return output;
}


int main()
{
    auto A = cv::imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe4.jpg");

    cv::cvtColor(A, A, CV_BGR2GRAY);
    A = shrink_image(A);

    cv::Mat h1 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
    constexpr double h2FilterFaktor = 1.0 / 16;
    cv::Mat h2 = (cv::Mat_<double>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) * h2FilterFaktor;
    cv::Mat h3 = (cv::Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);


    cv::Mat B = convolution(A, h1);
    cv::Mat C = convolution(A, h2);
    cv::Mat D = convolution(A, h3);


    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", A);
    cv::imshow("h1 Image", B);
    cv::imshow("h2 Image", C);
    cv::imshow("h3 Image", D);

    cv::waitKey(0);
    return 0;
}
