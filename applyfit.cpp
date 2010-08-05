#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

struct FitMatrix {
    double ** matrix;
    int n;
};

FitMatrix read_fit_matrix_csv(char * filename, int n)
{
    FitMatrix input_matrix;
    input_matrix.n = n;
    input_matrix.matrix = (double **)malloc(sizeof(double*)*n);
    
    std::ifstream data(filename);

    std::string line;
    
    int line_number = 0;
    
    while(std::getline(data,line))
    {
        std::stringstream  lineStream(line);
        std::string        cell;
    
        if(line_number < n) {
            input_matrix.matrix[line_number] = (double *)malloc(sizeof(double)*3);
            
            int pos = 0;
            
            while(std::getline(lineStream,cell,','))
            {
                if(pos < 3) {
                    input_matrix.matrix[line_number][pos] = atof(cell.c_str());
                }
                else {
                    break;
                }
                pos++;
            }
        }
        else {
            break;
        }
        
        line_number++;
    }
    
    return input_matrix;
}

int main( int argc, char *argv[] )
{
    if( argc < 4 )
    {
        fprintf( stderr, "Usage: %s fit_matrix.csv [input_image_1 input_image_2 input_image_3 ...] output_image\n", argv[0] );
        return 1;
    }
    
    int n = argc - 3;
    
    FitMatrix A = read_fit_matrix_csv(argv[1], n);
    fprintf(stderr,"Got fit:\n");
    for(int i = 0; i < n; i++) {
        fprintf(stderr,"%f,%f,%f\n",A.matrix[i][0],A.matrix[i][1],A.matrix[i][2]);
    }
    
    IplImage * xyz_recon = NULL;
    
    for(int i = 0; i < n; i++) {
        fprintf(stderr, "Loading channel %d (%s)\n", i, argv[i+2]);
        IplImage * input_channel = cvLoadImage( argv[i+2],
            CV_LOAD_IMAGE_GRAYSCALE|CV_LOAD_IMAGE_ANYDEPTH );
        
        if(xyz_recon == NULL) {
            xyz_recon = cvCreateImage(cvSize(input_channel->width, input_channel->height), IPL_DEPTH_32F, 3);
            cvSet(xyz_recon, cvScalarAll(0));
        }
        
        for(int y = 0; y < input_channel->height; y++) {
            for(int x = 0; x < input_channel->width; x++) {
                CvScalar xyz_value = cvGet2D(xyz_recon,y,x);
                CvScalar channel_value = cvGet2D(input_channel,y,x);
                
                for(int j = 0; j < 3; j++) {
                    double scaled = A.matrix[i][j]*channel_value.val[0];
                    xyz_value.val[j] += scaled;
                }
                
                cvSet2D(xyz_recon,y,x,xyz_value);
            }
        }
        
        cvReleaseImage( &input_channel );
    }
    
    cvCvtColor(xyz_recon, xyz_recon, CV_XYZ2BGR);
    
    cvSaveImage( argv[argc-1], xyz_recon );
    cvReleaseImage( &xyz_recon );
    
    return 0;
}