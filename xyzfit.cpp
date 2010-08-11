#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "colorchecker.h"

// http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html
CvScalar xyz_to_lab(CvScalar xyz, CvScalar reference_white)
{
    double epsilon = 216./24389.;
    double kappa = 24389./27.;
    
    CvScalar scaled;
    for(int i = 0; i < 3; i++) {
        scaled.val[i] = xyz.val[i] / reference_white.val[i];
    }
    
    CvScalar f;
    for(int i = 0; i < 3; i++) {
        if(scaled.val[i] > epsilon) {
            f.val[i] = pow(scaled.val[i],1./3.);
        }
        else {
            f.val[i] = (kappa * scaled.val[i] + 16.)/116.;
        }
    }
    
    CvScalar Lab = cvScalar(
        116. * f.val[1] - 16.,        // L
        500. * (f.val[0] - f.val[1]), // a
        200. * (f.val[1] - f.val[2])  // b
    );
    
    return Lab;
}

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
            
            fprintf(stderr,"Got patch %d at %d,%d\n",line_number,x,y);
        
            cvSet1D(input_colorchecker.points, line_number, cvScalar(x,y));
        }
        else if(line_number == MACBETH_SQUARES) {
            std::getline(lineStream,cell);
            
            input_colorchecker.size = atoi(cell.c_str());
            fprintf(stderr,"Got size %0.f\n", input_colorchecker.size);
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

CvScalar patch_average(ColorChecker input_colorchecker, int j, IplImage * xyz_recon)
{
    CvScalar point = cvGet1D(input_colorchecker.points, j);
    CvScalar average = rect_average(
        cvRect(point.val[0]-input_colorchecker.size/2,
               point.val[1]-input_colorchecker.size/2,
               input_colorchecker.size,
               input_colorchecker.size),
        xyz_recon);
    
    return average;
}

int main( int argc, char *argv[] )
{
    if( argc < 4 )
    {
        fprintf( stderr, "Usage: %s patch_locations.csv [input_image_1 input_image_2 input_image_3 ...] output_image\n", argv[0] );
        return 1;
    }
    
    ColorChecker input_colorchecker = read_colorchecker_csv(argv[1]);

    int n = argc - 3;
    
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
            fprintf(stderr,"%0.f,",average);
        }
        fprintf(stderr,"\n");
        
        cvReleaseImage( &input_channel );
    }
    
    gsl_matrix * P = gsl_matrix_calloc(MACBETH_SQUARES,3);
    for(int i = 0; i < MACBETH_SQUARES; i++) {
        for(int j = 0; j < 3; j++) {
            gsl_matrix_set(P, i, j, colorchecker_xyz[i][j]);
        }
    }
    
    gsl_matrix * V = gsl_matrix_calloc(MACBETH_SQUARES,n);
    for(int i = 0; i < MACBETH_SQUARES; i++) {
        for(int j = 0; j < n; j++) {
            gsl_matrix_set(V,i,j,colorchecker_channels[j][i]);
        }
    }
    
    gsl_matrix * VT = gsl_matrix_alloc(n,MACBETH_SQUARES);
    gsl_matrix_transpose_memcpy(VT,V);
    
    // zeros matrix for dgemm add
    // also stores result
    gsl_matrix * VTV = gsl_matrix_calloc(n,n);
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, VT, V,
                   0.0, VTV);
    
    gsl_matrix * VTP = gsl_matrix_calloc(n,3);
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, VT, P,
                   0.0, VTP);
    
    gsl_matrix * A = gsl_matrix_calloc(n,3);
    
    gsl_permutation * p = gsl_permutation_alloc(n);
    int lu_signum;
    gsl_linalg_LU_decomp(VTV, p, &lu_signum); // decompose once
    
    for(int i = 0; i < 3; i++) {
        // solve A x = b for each column
        // where x = A
        //       A = VTV
        //       b = VTP
        gsl_vector_view b = gsl_matrix_column(VTP,i);
        gsl_vector_view x = gsl_matrix_column(A,i);
        
        gsl_linalg_LU_solve(VTV,p,&b.vector,&x.vector);
    }
    gsl_permutation_free(p);
    
    // for (int i = 0; i < n; i++)
    //   for (int j = 0; j < 3; j++)
    //     fprintf(stderr,"A(%d,%d) = %g\n", i, j, 
    //             gsl_matrix_get (A, i, j));
    
    IplImage * xyz_recon = cvCreateImage(input_size, IPL_DEPTH_32F, 3);
    cvSet(xyz_recon, cvScalarAll(0));
    
    fprintf(stderr,"Depth: %d\n",input_depth);
    for(int i = 0; i < n; i++) {
        fprintf(stderr, "Loading channel %d (%s)\n", i, argv[i+2]);
        IplImage * input_channel = cvLoadImage( argv[i+2],
            CV_LOAD_IMAGE_GRAYSCALE|CV_LOAD_IMAGE_ANYDEPTH );
        
        printf("%f,%f,%f,%s\n", gsl_matrix_get(A,i,0), gsl_matrix_get(A,i,1), gsl_matrix_get(A,i,2), argv[i+2]);
        
        for(int y = 0; y < input_size.height; y++) {
            for(int x = 0; x < input_size.width; x++) {
                bool interested = false; // is_colorchecker_point(input_colorchecker,x,y);
                CvScalar xyz_value = cvGet2D(xyz_recon,y,x);
                CvScalar channel_value = cvGet2D(input_channel,y,x);
                
                for(int j = 0; j < 3; j++) {
                    double scaled = gsl_matrix_get(A,i,j)*channel_value.val[0];
                    xyz_value.val[j] += scaled;
                    if(interested) {
                        fprintf(stderr,"%f: %f\t",scaled,xyz_value.val[j]);
                    }
                }
                if(interested) {
                    fprintf(stderr,"\n");
                }
                
                cvSet2D(xyz_recon,y,x,xyz_value);
            }
        }
        
        cvReleaseImage( &input_channel );
    }
    
    fprintf(stderr,"ΔEab:\n");
    CvScalar average_white = patch_average(input_colorchecker, WHITE_PATCH, xyz_recon);
        
    for(int j = 0; j < MACBETH_SQUARES; j++) {
        CvScalar average = patch_average(input_colorchecker, j, xyz_recon);
        // fprintf(stderr,"%f,%f,%f\n",average.val[0],average.val[1],average.val[2]);
        
        fprintf(stderr,"%d\t%f\n",j+1,
            euclidean_distance(xyz_to_lab(average, average_white),
                               xyz_to_lab(
                                   cvScalar(colorchecker_xyz[j][0],
                                            colorchecker_xyz[j][1],
                                            colorchecker_xyz[j][2]),
                                   cvScalar(colorchecker_xyz[WHITE_PATCH][0],
                                            colorchecker_xyz[WHITE_PATCH][1],
                                            colorchecker_xyz[WHITE_PATCH][2])
                               )));
    }
    
    cvCvtColor(xyz_recon, xyz_recon, CV_XYZ2BGR);
    
    cvSaveImage( argv[argc-1], xyz_recon );
    cvReleaseImage( &xyz_recon );
    
    return 0;
}