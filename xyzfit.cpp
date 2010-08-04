#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "colorchecker.h"

int main( int argc, char *argv[] )
{
    if( argc < 2 )
    {
        fprintf( stderr, "Usage: %s macbeth_locations.csv [input_image_1 input_image_2 input_image_3 ...]\n", argv[0] );
        return 1;
    }

    int n = argc - 2;
    IplImage ** input_channels = (IplImage**)malloc(n*sizeof(IplImage**));
    
    for(int i = 0; i < n; i++) {
        printf("Loading channel %d (%s)\n", i, argv[i+2]);
        input_channels[i] = cvLoadImage( argv[i+2],
            CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH );
    }
    
    printf("Releasing images...\n");
    for(int i = 0; i < n; i++) {
        cvReleaseImage( &(input_channels[i]) );
    }

    return 0;
}