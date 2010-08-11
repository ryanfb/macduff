#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

#define MACBETH_WIDTH   6
#define MACBETH_HEIGHT  4
#define MACBETH_SQUARES MACBETH_WIDTH * MACBETH_HEIGHT
#define WHITE_PATCH     18

// BabelColor averages in sRGB:
//   http://www.babelcolor.com/main_level/ColorChecker.htm
// (converted to BGR order for comparison)
CvScalar colorchecker_srgb[MACBETH_HEIGHT][MACBETH_WIDTH] =
    {
        {
            cvScalar(67,81,115),
            cvScalar(129,149,196),
            cvScalar(157,123,93),
            cvScalar(65,108,90),
            cvScalar(176,129,130),
            cvScalar(171,191,99)
        },
        {
            cvScalar(45,123,220),
            cvScalar(168,92,72),
            cvScalar(98,84,195),
            cvScalar(105,59,91),
            cvScalar(62,189,160),
            cvScalar(41,161,229)
        },
        {
            cvScalar(147,62,43),
            cvScalar(72,149,71),
            cvScalar(56,48,176),
            cvScalar(22,200,238),
            cvScalar(150,84,188),
            cvScalar(166,136,0)
        },
        {
            cvScalar(240,245,245),
            cvScalar(201,201,200),
            cvScalar(161,161,160),
            cvScalar(121,121,120),
            cvScalar(85,84,83),
            cvScalar(50,50,50)
        }
    };


// ColorChecker XYZ vendor information from 2005,
// taken from http://www.babelcolor.com/main_level/ColorChecker.htm
double colorchecker_xyz[MACBETH_SQUARES][3] = {
    {10.7846492458451,9.81077849310112,7.70815627733434},
    {36.6852533323943,34.0026228274656,28.7808430583573},
    {15.8548147702032,17.6262277403054,35.6451162787379},
    {10.1128870138148,13.0229995780292,8.00651438080897},
    {22.877874100483,22.0251459896225,46.4889291382332 },
    {28.315347793307,40.7854321257436,48.8343537306251 },
    {38.08855026808,30.4515999685149,9.86964876829322  },
    {11.8273224924775,10.4789901634399,40.7526434703343},
    {27.9304847386873,18.6767748895944,15.7877503722228},
    {8.08839666594609,6.0155088618919,14.9984736208684 },
    {32.6431067060316,43.8995088995326,15.0731798688446},
    {45.4597574924436,42.7562725233556,12.0596519910001},
    {6.65597048768225,5.21320891079771,28.9877731066139},
    {13.7600185368047,22.9070673624638,11.9403205301331},
    {20.3624066906357,12.1131506597928,6.74719666886478},
    {55.1018843173195,58.8866752609815,14.5735569637824},
    {29.4169723142285,19.0876602614381,33.7239597694646},
    {12.8387045600875,18.5459610077425,42.4160176728439},
    {82.2406517798089,88.6827944210071,105.851085257239},
    {52.9906523247944,57.2174950869026,69.8804626631961},
    {32.6063504411145,35.2616841034241,43.2483741898732},
    {17.2741131602921,18.5936401635115,22.7358630679439},
    {7.93386184429607,8.56775332784279,10.818847695165 },
    {2.80207274806336,3.01238408341872,3.82277704246866}
};

double euclidean_distance(CvScalar p_1, CvScalar p_2)
{   
    double sum = 0;
    for(int i = 0; i < 3; i++) {
        sum += pow(p_1.val[i]-p_2.val[i],2.);
    }
    return sqrt(sum);
}

double euclidean_distance(CvPoint p_1, CvPoint p_2)
{
    return euclidean_distance(cvScalar(p_1.x,p_1.y,0),cvScalar(p_2.x,p_2.y,0));
}

double euclidean_distance_lab(CvScalar p_1, CvScalar p_2)
{
    // convert to Lab for better perceptual distance
    IplImage * convert = cvCreateImage( cvSize(2,1), 8, 3);
    cvSet2D(convert,0,0,p_1);
    cvSet2D(convert,0,1,p_2);
    cvCvtColor(convert,convert,CV_BGR2Lab);
    p_1 = cvGet2D(convert,0,0);
    p_2 = cvGet2D(convert,0,1);
    cvReleaseImage(&convert);
    
    return euclidean_distance(p_1, p_2);
}

struct ColorChecker {
    double error;
    CvMat * values;
    CvMat * points;
    double size;
};