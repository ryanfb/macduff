#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define MACBETH_WIDTH   6
#define MACBETH_HEIGHT  4
#define MACBETH_SQUARES MACBETH_WIDTH * MACBETH_HEIGHT

#define MAX_CONTOUR_APPROX  7

CvSeq * find_quad( CvSeq * src_contour, CvMemStorage *storage, int min_size)
{
    // stolen from icvGenerateQuads
    CvMemStorage * temp_storage = cvCreateChildMemStorage( storage );
    
    int flags = CV_CALIB_CB_FILTER_QUADS;
    CvSeq *dst_contour = 0;
    
    const int min_approx_level = 2, max_approx_level = MAX_CONTOUR_APPROX;
    int approx_level;
    for( approx_level = min_approx_level; approx_level <= max_approx_level; approx_level++ )
    {
        dst_contour = cvApproxPoly( src_contour, sizeof(CvContour), temp_storage,
                                    CV_POLY_APPROX_DP, (float)approx_level );
        // we call this again on its own output, because sometimes
        // cvApproxPoly() does not simplify as much as it should.
        dst_contour = cvApproxPoly( dst_contour, sizeof(CvContour), temp_storage,
                                    CV_POLY_APPROX_DP, (float)approx_level );

        if( dst_contour->total == 4 )
            break;
    }

    // reject non-quadrangles
    if( dst_contour->total == 4 && cvCheckContourConvexity(dst_contour) )
    {
        CvPoint pt[4];
        double d1, d2, p = cvContourPerimeter(dst_contour);
        double area = fabs(cvContourArea(dst_contour, CV_WHOLE_SEQ));
        double dx, dy;

        for( int i = 0; i < 4; i++ )
            pt[i] = *(CvPoint*)cvGetSeqElem(dst_contour, i);

        dx = pt[0].x - pt[2].x;
        dy = pt[0].y - pt[2].y;
        d1 = sqrt(dx*dx + dy*dy);

        dx = pt[1].x - pt[3].x;
        dy = pt[1].y - pt[3].y;
        d2 = sqrt(dx*dx + dy*dy);

        // philipg.  Only accept those quadrangles which are more square
        // than rectangular and which are big enough
        double d3, d4;
        dx = pt[0].x - pt[1].x;
        dy = pt[0].y - pt[1].y;
        d3 = sqrt(dx*dx + dy*dy);
        dx = pt[1].x - pt[2].x;
        dy = pt[1].y - pt[2].y;
        d4 = sqrt(dx*dx + dy*dy);
        if( !(flags & CV_CALIB_CB_FILTER_QUADS) ||
            (d3*4 > d4 && d4*4 > d3 && d3*d4 < area*1.5 && area > min_size &&
            d1 >= 0.15 * p && d2 >= 0.15 * p) )
        {
            // CvContourEx* parent = (CvContourEx*)(src_contour->v_prev);
            // parent->counter++;
            // if( !board || board->counter < parent->counter )
            //     board = parent;
            // dst_contour->v_prev = (CvSeq*)parent;
            //for( i = 0; i < 4; i++ ) cvLine( debug_img, pt[i], pt[(i+1)&3], cvScalar(200,255,255), 1, CV_AA, 0 );
            // cvSeqPush( root, &dst_contour );
            return dst_contour;
        }
    }
    
    return NULL;
}

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
        
        IplImage * macbeth_masked = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), macbeth_img->depth, macbeth_img->nChannels );
        
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
            CV_PI/180, block_size*8, block_size*5,
            block_size*20
        );
        
        printf("%d lines\n",results->total);
        for(int i = 0; i < results->total; i++) {
            CvPoint* line = (CvPoint*)cvGetSeqElem(results, i);
            
            double slope = ((double)(line[0].y - line[1].y))/((double)(line[0].x - line[1].x));
            double x_amount = cos(1./slope) * element_size * 8;
            double y_amount = sin(1./slope) * element_size * 8;
            
            for(int i = 0; i < 2; i++) {
                CvPoint this_line[2];
                if(i % 2) {
                    x_amount *= -1;
                    y_amount *= -1;
                }
                this_line[0].x = line[0].x - x_amount;
                this_line[1].x = line[1].x - x_amount;
                this_line[0].y = line[0].y + y_amount;
                this_line[1].y = line[1].y + y_amount;
                
                // cvLine(macbeth_img, this_line[0], this_line[1], CV_RGB( rand()&255, rand()&255, rand()&255 ), element_size);
            }
            
            // cvLine(macbeth_img, line[0], line[1], CV_RGB( rand()&255, rand()&255, rand()&255 ), element_size-2);
        }
        
        CvSeq * contours = NULL;
        cvFindContours(adaptive,storage,&contours);
        
        int min_size = (macbeth_img->width*macbeth_img->height)/
            (MACBETH_SQUARES*100);
        
        if(contours) {
            for( CvSeq* c = contours; c != NULL; c = c->h_next) {
                CvRect rect = ((CvContour*)c)->rect;
                if(CV_IS_SEQ_HOLE(c) && rect.width*rect.height >= min_size) {
                    CvSeq * quad_contour = find_quad(c, storage, min_size);
                    if(quad_contour) {
                        cvDrawContours(
                            macbeth_img,
                            quad_contour,
                            cvScalar(255,0,0),
                            cvScalar(0,0,255),
                            0,
                            element_size
                        );
                    }
                }
            }
        }
        
        cvReleaseMemStorage( &storage );
        
        // cvSetZero(macbeth_masked);
        // cvNot(adaptive,adaptive);
        // cvCopy( macbeth_img, macbeth_masked, adaptive );
        // cvNot(adaptive,adaptive);
        
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

