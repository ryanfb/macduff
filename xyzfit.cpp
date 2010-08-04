#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "colorchecker.h"

ColorChecker read_colorchecker_csv(char * filename)
{
    ColorChecker input_colorchecker;
    input_colorchecker.points = cvCreateMat( MACBETH_SQUARES, 1, CV_32FC2 );
    
    std::ifstream data(filename);

    std::string line;
    
    int line_number = 0;
    
    while(std::getline(data,line))
    {
        std::stringstream  lineStream(line);
        std::string        cell;
    
        if(line_number < MACBETH_SQUARES) {
            bool read_x = false;
            int x = 0;
            int y = 0;
        
            while(std::getline(lineStream,cell,','))
            {
                if(!read_x) {
                    x = atoi(cell.c_str());
                    read_x = true;
                }
                else {
                    y = atoi(cell.c_str());
                    break;
                }
            }
            
            printf("Got patch %d at %d,%d\n",line_number,x,y);
        
            cvSet1D(input_colorchecker.points, line_number, cvScalar(x,y));
        }
        else if(line_number == MACBETH_SQUARES) {
            std::getline(lineStream,cell);
            
            input_colorchecker.size = atoi(cell.c_str());
            printf("Got size %0.f\n", input_colorchecker.size);
        }
        else {
            break;
        }
        
        line_number++;
    }
    
    return input_colorchecker;
}

int main( int argc, char *argv[] )
{
    if( argc < 2 )
    {
        fprintf( stderr, "Usage: %s patch_locations.csv [input_image_1 input_image_2 input_image_3 ...]\n", argv[0] );
        return 1;
    }
    
    ColorChecker input_colorchecker = read_colorchecker_csv(argv[1]);

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