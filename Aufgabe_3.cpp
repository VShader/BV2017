#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
#include <QApplication>
#include <QWidget>
#include <QBarset>
#include <QBarSeries>
#include <QChart>
#include <QPushButton>
#include <QComboBox>
#include <QLayout>
#include <QDebug>

enum Mode : uint8_t {stretching, linearisation, gammacorrection};
cv::Mat A, Result;

cv::Mat shrink_image(cv::Mat& input)
{
    cv::Mat output;
    auto max = std::max(input.rows, input.cols);
    auto shrinkValue = max > 512 ? 512.0f/max : 1;
    cv::resize(input, output, cv::Size( input.cols*shrinkValue, input.rows*shrinkValue ));
    return output;
}


std::array<uint32_t, 256> makeHist(cv::Mat& input)
{
    //check if grayscale, if not convert.
    if(input.dims > 2)
        cv::cvtColor(input, input, CV_BGR2GRAY);

    std::array<uint32_t, 256> hist;
    hist.fill(0);
    for (int colIt = 0; colIt < input.cols; colIt++)
    {
        for (int rowIt = 0; rowIt < input.rows; rowIt++)
        {
            auto& pixel = input.at<uint8_t>(rowIt, colIt);
            hist[pixel] += 1;
        }
    }
    return hist;
}

std::array<double, 256> makeNormalisedHist(std::array<uint32_t, 256>& hist)
{
    std::array<double, 256> output;
    uint32_t pixelSum = 0;
    for(auto pixel : hist)
    {
        pixelSum += pixel;
    }
    std::transform(hist.begin(), hist.end(), output.begin(),
                   [pixelSum](double value) { return value/(double)pixelSum; });
    return output;
}

std::array<double, 256> makeCumulativeHist(std::array<double, 256>& normalisedHist)
{
    auto output = normalisedHist;
    for (auto iterator = ++output.begin(); iterator != output.end(); iterator++)
    {
        *iterator += *(iterator - 1);
    }
    return output;
}

//void histStretch()
//{
//    constexpr uchar wmin = 0;
//    constexpr uchar wmax = 255;

//    auto hist = makeHist();
//    const auto minMax = getMinMax(hist);

//    for (int colIt = 0; colIt < this->cols; colIt++)
//    {
//        for (int rowIt = 0; rowIt < this->rows; rowIt++)
//        {
//            auto& pixel = this->at<uchar>(rowIt, colIt);
//            pixel = histStretch(pixel, minMax.first, minMax.second, wmin, wmax);
//        }
//    }
//}


cv::Mat gammaCorrection(cv::Mat& input, const double gamma)
{
    cv::Mat output(input.clone());
    output.convertTo(output, CV_64F);
    constexpr uint8_t wmin = 0;
    constexpr uint8_t wmax = 255;

    auto histogram = makeHist(input);

    uint8_t gmin = 255;
    uint8_t gmax = 0;
    for(auto item : histogram)
    {
        gmin = item < gmin ? item : gmin;
        gmax = item > gmax ? item : gmax;
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


void calc(Mode modus, double gamma = 0)
{
    qDebug() << modus;
    auto histogram = makeHist(A);
    auto normHistogram = makeNormalisedHist(histogram);
    auto cumHistogram = makeCumulativeHist(normHistogram);

    std::string title;





    switch (modus) {
    case stretching:
        Result = histogramStreching(A);
        title = "HistStrecht Image";
        break;
    case linearisation:
        cv::equalizeHist(A, Result);
        title = "equalizeHist Image";
        break;
    case gammacorrection:
        Result = gammaCorrection(A, gamma);
        title = "Gamma Image";
        break;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", A);
    cv::imshow(title, Result);
}

int main(int32_t argc, char** argv)
{
    A = cv::imread("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe3.jpg");

    cv::cvtColor(A, A, CV_BGR2GRAY);
    A = shrink_image(A);


//    auto histogram = makeHist(A);
//    auto normHistogram = makeNormalisedHist(histogram);
//    auto cumHistogram = makeCumulativeHist(normHistogram);

//    cv::Mat B(histogramStreching(A));
//    cv::Mat C(gammaCorrection(A, 5));
//    cv::Mat D;
//    cv::equalizeHist(A, D);

//    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
//    cv::imshow("Display Image", A);
//    cv::imshow("HistStrecht Image", B);
//    cv::imshow("Gamma Image", C);
//    cv::imshow("equalizeHist Image", D);

    QApplication app(argc, argv);
//    QBarSet *set0 = new QBarSet("Jane");
//    *set0 << 1 << 2 << 3 << 4 << 5 << 6;
//    QBarSeries *series = new QBarSeries();
//    series->append(set0);

//    QChart *chart = new QChart();
//    chart->addSeries(series);
//    chart->setTitle("Simple barchart example");
//    chart->setAnimationOptions(QChart::SeriesAnimations);
    QWidget wid;
    QComboBox* combobox = new QComboBox();
    combobox->addItem("Histogramm Stretching");
    combobox->addItem("Histogramm Linearisierung");
    combobox->addItem("Gammakorrektur");
    QPushButton* button = new QPushButton("OK");
    QVBoxLayout* layout  = new QVBoxLayout();
    layout->addWidget(combobox);
    layout->addWidget(button);
    wid.setLayout(layout);
    wid.show();

    QObject::connect(button, &QPushButton::clicked, [combobox](){calc((Mode)combobox->currentIndex(), 5);});
    //cv::waitKey(0);
    //return 0;
    return app.exec();
}
