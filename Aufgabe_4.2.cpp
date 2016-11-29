#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>

#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QComboBox>
#include <QtWidgets/QLineEdit>
#include <QLayout>
#include <QtWidgets/QFileDialog>
#include <QDebug>

enum Mode {normierterMittelwertsfilter, gaussfilter, xDerivative, yDerivative, laplace, sobelInX, sobelInY, gradientenbetrag, cannyEdgeDetector};
cv::Mat A, Result;

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




//    double min, max;
//    cv::minMaxIdx(output, &min, &max);
//    cv::convertScaleAbs(output, output, 255 / max);
    output = histogramStreching(output);
    return output;
}

void loadImage(std::string path)
{
    A = cv::imread(path, cv::IMREAD_GRAYSCALE);
    A = shrink_image(A);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", A);
}

void calc(Mode modus, double gamma = 0)
{
    qDebug() << modus;

    std::string title;
    cv::Mat xGrad = (cv::Mat_<double>(3, 1) << 1, -1, 0);
    cv::Mat yGrad = (cv::Mat_<double>(1, 3) << 0, 1, -1);
    cv::Mat sobelX, sobelY;

    switch (modus) {
    case normierterMittelwertsfilter:
        title = "Normierter Mittelwertsfilter";
        //DONE: Normierter Mittelwertsfilter mit Eingabe der Filtergroesse
        cv::blur(A, Result, { 3, 3 });
        break;
    case gaussfilter:
        cv::equalizeHist(A, Result);
        title = "Gaussfilter";
        cv::GaussianBlur(A, Result, {3, 3}, 3);
        break;
    case xDerivative:
        title = "x Gradient";
        cv::filter2D(A, Result, -1, xGrad, { -1, -1 }, 0, cv::BORDER_DEFAULT);
        break;
    case yDerivative:
        title = "y Gradient";
        cv::filter2D(A, Result, -1, yGrad, { -1, -1 }, 0, cv::BORDER_DEFAULT);
        break;
    case laplace:
        cv::Laplacian(A, Result, -1);
        break;
    case sobelInX:
        cv::Sobel(A, Result, -1, 1, 0);
        break;
    case sobelInY:
        cv::Sobel(A, Result, -1, 0, 1);
        break;
    case gradientenbetrag:
        cv::Sobel(A, sobelX, -1, 1, 0);
        cv::Sobel(A, sobelY, -1, 0, 1);
        Result = sobelX.mul(sobelX) + sobelY.mul(sobelY);
        Result.convertTo(Result, CV_64F);
        cv::sqrt(Result, Result);
        break;
    case cannyEdgeDetector:
        cv::Canny(A, Result, 1, 1);
        break;
    }

    cv::Mat with;
    with = histogramStreching(Result);
    cv::imshow(title, Result);
    cv::imshow(title + "+", with);
}



int main(int argc, char** argv)
{
    QApplication app(argc, argv);


    loadImage("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe4.jpg");

    QWidget wid;
    QComboBox* combobox = new QComboBox();
    combobox->addItem("Normierter Mittelwertsfilter");
    combobox->addItem("Gaussfilter");
    combobox->addItem("1. x-Ableitung");
    combobox->addItem("1. y-Ableitung");
    combobox->addItem("Laplace");
    combobox->addItem("Sobel in x");
    combobox->addItem("Sobel in y");
    combobox->addItem("Gradientenbetrag nach Sobel");
    combobox->addItem("Canny Edge Detector");
    QLineEdit* gammaVal = new QLineEdit();
    QPushButton* button = new QPushButton("OK");
    QPushButton* loadButton = new QPushButton("Load");
    QVBoxLayout* vLayout1  = new QVBoxLayout();
    QHBoxLayout* hLayout1  = new QHBoxLayout();
    hLayout1->addWidget(gammaVal);
    hLayout1->addWidget(button);
    vLayout1->addWidget(loadButton);
    vLayout1->addWidget(combobox);
    vLayout1->addLayout(hLayout1);
    wid.setLayout(vLayout1);
    wid.show();

    QObject::connect(button, &QPushButton::clicked, [combobox, gammaVal](){calc((Mode)combobox->currentIndex(), gammaVal->text().toDouble());});
    QObject::connect(loadButton, &QPushButton::clicked, []()
    {
        QFileDialog dia;
        loadImage(dia.getOpenFileUrl().toString().remove("file:///").toStdString());
    });

    return app.exec();
}
