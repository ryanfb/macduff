#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define PYRAMID_WIDTH   5120
#define PYRAMID_HEIGHT  3840
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
    
    IplImage * pyramidable = cvCreateImage( cvSize(PYRAMID_WIDTH, PYRAMID_HEIGHT), macbeth_img->depth, macbeth_img->nChannels );
    
    cvResize( macbeth_img, pyramidable );

    IplImage * segments = cvCreateImage( cvSize(pyramidable->width, pyramidable->height), macbeth_img->depth, macbeth_img->nChannels );
    
    if( pyramidable && segments )
    {
        int adaptive_method = CV_ADAPTIVE_THRESH_GAUSSIAN_C;
        int threshold_type = CV_THRESH_BINARY_INV;
        int block_size = cvRound(
            MIN(macbeth_img->width,macbeth_img->height)*0.2)|1;
        printf("Using %d as block size\n", block_size);
        double offset = 15;
        
        for(int i = 0; i < 3; i++) {
            cvAdaptiveThreshold(macbeth_split[i], macbeth_split_thresh[i], 255, adaptive_method, threshold_type, block_size, offset);
        }
        
        IplImage * adaptive = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), IPL_DEPTH_8U, 1 );
        
        cvOr(macbeth_split_thresh[0],macbeth_split_thresh[1],adaptive);
        cvOr(macbeth_split_thresh[2],adaptive,adaptive);
        
        cvSaveImage( "adaptive.png", adaptive);
        cvReleaseImage( &adaptive );
        
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* comp = NULL;
        
        cvPyrSegmentation( pyramidable, segments, storage, &comp, 4, 150, 30);
        
        int n_comp = comp->total;
        for( int i = 0; i < n_comp; i++ ) {
            CvConnectedComp* cc = (CvConnectedComp*) cvGetSeqElem( comp, i );
            // if((cc->area > ((PYRAMID_HEIGHT*PYRAMID_WIDTH)/(MACBETH_SQUARES*32))) && (cc->area < ((PYRAMID_HEIGHT*PYRAMID_WIDTH)/(MACBETH_SQUARES*8)))) {
            if(1) {
                printf("Area: %f\n", cc->area);
                printf("Color: %f %f %f\n", cc->value.val[0], cc->value.val[1], cc->value.val[2]);
                printf("Rect: %d %d %d %d\n", cc->rect.x, cc->rect.y, cc->rect.width, cc->rect.height);
                printf("\n");

                CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );

                // cvRectangle(segments, cvPoint(cc->rect.x, cc->rect.y), cvPoint(cc->rect.x+cc->rect.width, cc->rect.y+cc->rect.height), color);
                // cvDrawContours(segments, cc->contour, color, CV_RGB(0,0,0), 0, 5);
            }
        }
        
        cvReleaseMemStorage( &storage );

        cvReleaseImage( &macbeth_img );
        
        for(int i = 0; i < 3; i++) {
            cvReleaseImage( &(macbeth_split[i]) );
            cvReleaseImage( &(macbeth_split_thresh[i]) );
        }
        
        cvReleaseImage( &pyramidable );

        return segments;
    }

    if( macbeth_img ) cvReleaseImage( &macbeth_img );
    if( segments ) cvReleaseImage( &segments );

    return NULL;
}

int main( int argc, char *argv[] )
{
    if( argc < 2 )
    {
        fprintf( stderr, "Usage: %s image_file\n", argv[0] );
    }

    const char *img_file = argv[1];

    IplImage *out = find_macbeth( img_file );
    cvSaveImage( "result.png", out );
    cvReleaseImage( &out );

    return 0;
}

