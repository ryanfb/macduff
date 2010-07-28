#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define MACBETH_SQUARES 24

IplImage * find_macbeth( const char *img )
{
    IplImage * macbeth_img = cvLoadImage( img,
        CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH );
    
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
        
        cvReleaseImage( &macbeth_img );
        
        for(int i = 0; i < 3; i++) {
            cvReleaseImage( &(macbeth_split[i]) );
            cvReleaseImage( &(macbeth_split_thresh[i]) );
        }
        
        int element_size = (block_size/10)+2;
        printf("Using %d as element size\n", element_size);
        
        IplConvKernel * element = cvCreateStructuringElementEx(element_size,element_size,element_size/2,element_size/2,CV_SHAPE_RECT);
        cvMorphologyEx(adaptive,adaptive,NULL,element,CV_MOP_OPEN);
        cvReleaseStructuringElement(&element);
        
        return adaptive;
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

