#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
#include <QApplication>
#include <QWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QLegend>
#include <QtCharts/QBarCategoryAxis>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QLayout>
#include <QtWidgets/QFileDialog>
#include <QDebug>

QT_CHARTS_USE_NAMESPACE

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

void loadImage(std::string path, QChart* chart)
{
    A = cv::imread(path, cv::IMREAD_GRAYSCALE);
    A = shrink_image(A);

    auto newHist = makeHist(A);
    QBarSet *set0 = new QBarSet("Grayval");
    set0->setColor(Qt::black);
    for(uint16_t i=0; i<newHist.size(); ++i)
    {
        set0->insert(i, newHist[i]);
    }
    QBarSeries *series = new QBarSeries();
    series->append(set0);
    chart->removeAllSeries();
    chart->addSeries(series);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", A);
}

void calc(Mode modus, double gamma = 0, QChart* chart = nullptr)
{
    qDebug() << modus;
    auto histogram = makeHist(A);
    auto normHistogram = makeNormalisedHist(histogram);
    auto cumHistogram = makeCumulativeHist(normHistogram);

    std::string title;
    QBarSet *set = new QBarSet("Grayval");


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

    auto newHist = makeHist(Result);
    QBarSet *set0 = new QBarSet("Grayval");
    set0->setColor(Qt::black);
    for(uint16_t i=0; i<newHist.size(); ++i)
    {
        set0->insert(i, newHist[i]);
    }
    QBarSeries *series = new QBarSeries();
    series->append(set0);
    chart->removeAllSeries();
    chart->addSeries(series);
    cv::imshow(title, Result);
}

int main(int32_t argc, char** argv)
{
    //auto histogram = makeHist(A);
    QApplication app(argc, argv);

//    QBarSet *set0 = new QBarSet("Grayval");
//    set0->setColor(Qt::black);
//    for(int i=0; i< histogram.size(); ++i)
//        set0->insert(i, histogram[i]);
//    QBarSeries *series = new QBarSeries();
//    series->append(set0);

    QChart *chart = new QChart();
    //chart->addSeries(series);
    chart->setTitle("Orginal Image");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    //chartView->show();

    //QBarSeries *seriesResult = new QBarSeries();
    QChart *chartResult = new QChart();
//    chartResult->addSeries(seriesResult);
    chartResult->setTitle("Result Image");
    chartResult->setAnimationOptions(QChart::SeriesAnimations);
    QChartView *chartViewResult = new QChartView(chartResult);
    chartViewResult->setRenderHint(QPainter::Antialiasing);
    //chartViewResult->show();

    loadImage("E:\\FH-Aachen\\5.\ Semerster\\Bildverarbeitung\\BV_Bilder\\Aufgabe3.jpg", chart);

    QWidget wid;
    QComboBox* combobox = new QComboBox();
    combobox->addItem("Histogramm Stretching");
    combobox->addItem("Histogramm Linearisierung");
    combobox->addItem("Gammakorrektur");
    QSpinBox* gammaVal = new QSpinBox();
    QPushButton* button = new QPushButton("OK");
    QPushButton* loadButton = new QPushButton("Load");
    QVBoxLayout* vLayout1  = new QVBoxLayout();
    QHBoxLayout* hLayout1  = new QHBoxLayout();
    QVBoxLayout* vLayout2  = new QVBoxLayout();
    QHBoxLayout* hLayout2  = new QHBoxLayout();
    hLayout1->addWidget(gammaVal);
    hLayout1->addWidget(button);
    vLayout1->addWidget(loadButton);
    vLayout1->addWidget(combobox);
    vLayout1->addLayout(hLayout1);
    vLayout2->addWidget(chartView);
    vLayout2->addWidget(chartViewResult);
    hLayout2->addLayout(vLayout1);
    hLayout2->addLayout(vLayout2);
    wid.setLayout(hLayout2);
    wid.show();

    QObject::connect(button, &QPushButton::clicked, [combobox, gammaVal, chartResult](){calc((Mode)combobox->currentIndex(), gammaVal->value(), chartResult);});
    QObject::connect(loadButton, &QPushButton::clicked, [chart]()
    {
        QFileDialog dia;
        loadImage(dia.getOpenFileUrl().toString().remove("file:///").toStdString(), chart);
    });
    //cv::waitKey(0);
    //return 0;
    return app.exec();
}
