#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <armadillo>

#include "colorchecker.h"

using namespace arma;

CvScalar rect_average(CvRect rect, IplImage* image)
{       
    CvScalar average = cvScalarAll(0);
    int count = 0;
    for(int x = rect.x; x < (rect.x+rect.width); x++) {
        for(int y = rect.y; y < (rect.y+rect.height); y++) {
            if((x >= 0) && (y >= 0) && (x < image->width) && (y < image->height)) {
                CvScalar s = cvGet2D(image,y,x);
                average.val[0] += s.val[0];
                average.val[1] += s.val[1];
                average.val[2] += s.val[2];
            
                count++;
            }
        }
    }
    
    for(int i = 0; i < 3; i++){
        average.val[i] /= count;
    }
    
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

bool is_colorchecker_point(ColorChecker input_colorchecker, int x, int y)
{
    for(int j = 0; j < MACBETH_SQUARES; j++) {
        CvScalar point = cvGet1D(input_colorchecker.points, j);
        if((point.val[0] == x) && (point.val[1] == y)) {
            return true;
        }
    }
    return false;
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
    
    CvSize input_size;
    int input_depth;
    for(int i = 0; i < n; i++) {
        fprintf(stderr, "Loading channel %d (%s)\n", i, argv[i+2]);
        IplImage * input_channel = cvLoadImage( argv[i+2],
            CV_LOAD_IMAGE_GRAYSCALE|CV_LOAD_IMAGE_ANYDEPTH );
        input_size.width = input_channel->width;
        input_size.height = input_channel->height;
        input_depth = input_channel->depth;
        
        colorchecker_channels[i] = (double *)malloc(sizeof(double)*MACBETH_SQUARES);
        
        for(int j = 0; j < MACBETH_SQUARES; j++) {
            CvScalar point = cvGet1D(input_colorchecker.points, j);
            double average = rect_average(
                cvRect(point.val[0]-input_colorchecker.size/2,
                       point.val[1]-input_colorchecker.size/2,
                       input_colorchecker.size,
                       input_colorchecker.size),
                input_channel).val[0];
            colorchecker_channels[i][j] = average;
            printf("%0.f,",average);
        }
        printf("\n");
        
        cvReleaseImage( &input_channel );
    }
    
    mat P = zeros<mat>(MACBETH_SQUARES,3);
    for(int i = 0; i < MACBETH_SQUARES; i++) {
        for(int j = 0; j < 3; j++) {
            P(i,j) = colorchecker_xyz[i][j];
        }
    }
    
    imat V = zeros<imat>(MACBETH_SQUARES,n);
    for(int i = 0; i < MACBETH_SQUARES; i++) {
        for(int j = 0; j < n; j++) {
            V(i,j) = (int)round(colorchecker_channels[j][i]);
        }
    }
    
    P.print("P =");
    V.print("V =");
    
    imat VT = trans(V);
    VT.print("VT =");
    
    imat VTV = VT*V;
    VTV.print("VTV =");
    
    mat VTVinv = pinv(conv_to<mat>::from(VTV));
    VTVinv.print("VTVinv =");
    
    mat VTVinvVT = zeros<mat>(n,MACBETH_SQUARES);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < MACBETH_SQUARES; j++) {
            // combine row i of VTVinv with col j of VT
            for(int k = 0; k < n; k++) {
                VTVinvVT(i,j) += VTVinv(i,k)*VT(k,j);
            }
        }
    }
    VTVinvVT.print("VTVinvVT =");
    
    mat A = zeros<mat>(n,3);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < 3; j++) {
            // combine row i of VTVinvVT with col j of P
            for(int k = 0; k < MACBETH_SQUARES; k++) {
                A(i,j) += VTVinvVT(i,k)*P(k,j);
            }
        }
    }

    A.print("A =");
    
    IplImage * xyz_recon = cvCreateImage(input_size, IPL_DEPTH_32F, 3);
    cvSet(xyz_recon, cvScalarAll(0));

    printf("Depth: %d\n",input_depth);
    for(int i = 0; i < n; i++) {
        fprintf(stderr, "Loading channel %d (%s)\n", i, argv[i+2]);
        IplImage * input_channel = cvLoadImage( argv[i+2],
            CV_LOAD_IMAGE_GRAYSCALE|CV_LOAD_IMAGE_ANYDEPTH );
        
        for(int y = 0; y < input_size.height; y++) {
            for(int x = 0; x < input_size.width; x++) {
                bool interested = false; // is_colorchecker_point(input_colorchecker,x,y);
                CvScalar xyz_value = cvGet2D(xyz_recon,y,x);
                CvScalar channel_value = cvGet2D(input_channel,y,x);
                
                for(int j = 0; j < 3; j++) {
                    double scaled = A(i,j)*channel_value.val[0];
                    xyz_value.val[j] += scaled;
                    if(interested) {
                        printf("%f: %f\t",scaled,xyz_value.val[j]);
                    }
                }
                if(interested) {
                    printf("\n");
                }
                
                cvSet2D(xyz_recon,y,x,xyz_value);
            }
        }
        
        cvReleaseImage( &input_channel );
    }
    
    for(int j = 0; j < MACBETH_SQUARES; j++) {
        CvScalar point = cvGet1D(input_colorchecker.points, j);
        CvScalar average = rect_average(
            cvRect(point.val[0]-input_colorchecker.size/2,
                   point.val[1]-input_colorchecker.size/2,
                   input_colorchecker.size,
                   input_colorchecker.size),
            xyz_recon);
        printf("%f,%f,%f\n",average.val[0],average.val[1],average.val[2]);
    }
    
    cvCvtColor(xyz_recon, xyz_recon, CV_XYZ2BGR);
    
    cvSaveImage( "xyzrgb.png", xyz_recon );
    cvReleaseImage( &xyz_recon );
    
    return 0;
}