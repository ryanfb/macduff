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
        fprintf( stderr, "Usage: %s A.csv [input_image_1 input_image_2 input_image_3 ...] output_image\n", argv[0] );
        return 1;
    }
    
    int n = argc - 3;
    
    FitMatrix A = read_fit_matrix_csv(argv[1], n);
    
    return 0;
}