#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define MACBETH_WIDTH   6
#define MACBETH_HEIGHT  4
#define MACBETH_SQUARES MACBETH_WIDTH * MACBETH_HEIGHT

IplImage * find_macbeth( const char *img )
{
    IplImage * macbeth_img = cvLoadImage( img,
        CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH );
    
    
    // BabelColor averages in sRGB:
    //   http://www.babelcolor.com/main_level/ColorChecker.htm
    CvScalar colorchecker_srgb[MACBETH_HEIGHT][MACBETH_WIDTH] =
        {
            {
                cvScalar(115,81,67),
                cvScalar(196,149,129),
                cvScalar(93,123,157),
                cvScalar(90,108,65),
                cvScalar(130,129,176),
                cvScalar(99,191,171)
            },
            {
                cvScalar(220,123,45),
                cvScalar(72,92,168),
                cvScalar(195,84,98),
                cvScalar(91,59,105),
                cvScalar(160,189,62),
                cvScalar(229,161,41)
            },
            {
                cvScalar(43,62,147),
                cvScalar(71,149,72),
                cvScalar(176,48,56),
                cvScalar(238,200,22),
                cvScalar(188,84,150),
                cvScalar(0,136,166)
            },
            {
                cvScalar(245,245,240),
                cvScalar(200,201,201),
                cvScalar(160,161,161),
                cvScalar(120,121,121),
                cvScalar(83,84,85),
                cvScalar(50,50,50)
            }
        };
        
    IplImage * macbeth_split[3];
    IplImage * macbeth_split_thresh[3];
    
    for(int i = 0; i < 3; i++) {
        macbeth_split[i] = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), macbeth_img->depth, 1 );
        macbeth_split_thresh[i] = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), macbeth_img->depth, 1 );
    }
    
    cvSplit(macbeth_img, macbeth_split[0], macbeth_split[1], macbeth_split[2], NULL);
    
    if( macbeth_img )
    {
        int adaptive_method = CV_ADAPTIVE_THRESH_GAUSSIAN_C;
        int threshold_type = CV_THRESH_BINARY_INV;
        int block_size = cvRound(
            MIN(macbeth_img->width,macbeth_img->height)*0.02)|1;
        printf("Using %d as block size\n", block_size);
        
        double offset = 7;
        
        for(int i = 0; i < 3; i++) {
            cvAdaptiveThreshold(macbeth_split[i], macbeth_split_thresh[i], 255, adaptive_method, threshold_type, block_size, offset);
        }
        
        IplImage * adaptive = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), IPL_DEPTH_8U, 1 );
        
        cvOr(macbeth_split_thresh[0],macbeth_split_thresh[1],adaptive);
        cvOr(macbeth_split_thresh[2],adaptive,adaptive);
        
        // cvReleaseImage( &macbeth_img );
        
        for(int i = 0; i < 3; i++) {
            cvReleaseImage( &(macbeth_split[i]) );
            cvReleaseImage( &(macbeth_split_thresh[i]) );
        }
        
        int element_size = (block_size/10)+2;
        printf("Using %d as element size\n", element_size);
        
        IplConvKernel * element = cvCreateStructuringElementEx(element_size,element_size,element_size/2,element_size/2,CV_SHAPE_RECT);
        cvMorphologyEx(adaptive,adaptive,NULL,element,CV_MOP_OPEN);
        cvReleaseStructuringElement(&element);
        
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* results = cvHoughLines2(
            adaptive,
            storage,
            CV_HOUGH_PROBABILISTIC,
            1,
            CV_PI/180, block_size*4, block_size*10,
            block_size
        );
        
        printf("%d lines\n",results->total);
        for(int i = 0; i < results->total; i++) {
            CvPoint* line = (CvPoint*)cvGetSeqElem(results, i);
            cvLine(macbeth_img, line[0], line[1], CV_RGB( rand()&255, rand()&255, rand()&255 ), 5);
        }
        
        cvReleaseMemStorage( &storage );
        
        return macbeth_img;
    }

    if( macbeth_img ) cvReleaseImage( &macbeth_img );

    return NULL;
}

int main( int argc, char *argv[] )
{
    if( argc < 3 )
    {
        fprintf( stderr, "Usage: %s image_file out_file\n", argv[0] );
    }

    const char *img_file = argv[1];

    IplImage *out = find_macbeth( img_file );
    cvSaveImage( argv[2], out );
    cvReleaseImage( &out );

    return 0;
}

