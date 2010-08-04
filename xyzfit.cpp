#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "colorchecker.h"

double rect_average(CvRect rect, IplImage* image)
{       
    double average = 0;
    int count = 0;
    for(int x = rect.x; x < (rect.x+rect.width); x++) {
        for(int y = rect.y; y < (rect.y+rect.height); y++) {
            if((x >= 0) && (y >= 0) && (x < image->width) && (y < image->height)) {
                CvScalar s = cvGet2D(image,y,x);
                average += s.val[0];
                
                count++;
            }
        }
    }
    
    average /= count;
    
    return average;
}

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
    
    // colorchecker_channels[n][MACBETH_SQUARES]
    double ** colorchecker_channels = (double **)malloc(sizeof(double*)*n);
    
    for(int i = 0; i < n; i++) {
        fprintf(stderr, "Loading channel %d (%s)\n", i, argv[i+2]);
        IplImage * input_channel = cvLoadImage( argv[i+2],
            CV_LOAD_IMAGE_GRAYSCALE|CV_LOAD_IMAGE_ANYDEPTH );
        
        colorchecker_channels[i] = (double *)malloc(sizeof(double)*MACBETH_SQUARES);
        
        for(int j = 0; j < MACBETH_SQUARES; j++) {
            CvScalar point = cvGet1D(input_colorchecker.points, j);
            double average = rect_average(
                cvRect(point.val[0]-input_colorchecker.size/2,
                       point.val[1]-input_colorchecker.size/2,
                       input_colorchecker.size,
                       input_colorchecker.size),
                input_channel);
            colorchecker_channels[i][j] = average;
            printf("%0.f,",average);
        }
        printf("\n");
        
        cvReleaseImage( &input_channel );
    }
    
    return 0;
}