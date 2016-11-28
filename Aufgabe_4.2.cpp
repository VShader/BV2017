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
    zero.convertTo(zero, CV_64FC2);
    input.convertTo(output, CV_64FC2);

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




    double min, max;
    cv::minMaxIdx(output, &min, &max);
    cv::convertScaleAbs(output, output, 255 / max);
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


    cv::Mat xGrad = (cv::Mat_<double>(3, 1) << 1, -1, 0);
    cv::Mat yGrad = (cv::Mat_<double>(1, 3) << 0, 1, -1);

    cv::Mat blur, filter2, gauss, lapla, sobelX, sobelY, canny;
    //DONE: Normierter Mittelwertsfilter mit Eingabe der Filtergr??e
    cv::blur(A, blur, { 3, 3 });
    //imshow("blur", blur);
    //DONE: Gaussian
    cv::GaussianBlur(A, gauss, {3, 3}, 3);
    //imshow("gaussian", gauss);
    //TODO: 1.te x-Ableitung als Vorwaertsgradient
    //TODO: 1.te y-Ableitung als Vorwaertsgradient
    cv::filter2D(A, filter2, -1, xGrad, { -1, -1 }, 0, cv::BORDER_DEFAULT);
    imshow("ocvConv", filter2);
    cv::Laplacian(A, lapla, -1);
    //imshow("laplacian", lapla);
    cv::Sobel(A, sobelX, -1, 1, 0);
    imshow("sobelX", sobelX);
    cv::Sobel(A, sobelY, -1, 0, 1);
    imshow("sobelY", sobelY);
    //cv::Canny(image, canny, 1, 1);
    //imshow("canny", canny);
    //TODO: Gradientenbetrag nach Sobel, Formel auf Folie 7-49
    cv::Mat grad = sobelX.mul(sobelX) + sobelY.mul(sobelY);
    grad.convertTo(grad, CV_64F);
    cv::sqrt(grad, grad);
    cv::imshow("Gradientenbetrag", grad);





    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", A);

    cv::waitKey(0);
    return 0;
}
