#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define MACBETH_WIDTH   6
#define MACBETH_HEIGHT  4
#define MACBETH_SQUARES MACBETH_WIDTH * MACBETH_HEIGHT

#define MAX_CONTOUR_APPROX  7

#define MAX_RGB_DISTANCE 444

double euclidean_distance(CvScalar p_1, CvScalar p_2)
{   
    double sum = 0;
    for(int i = 0; i < 3; i++) {
        sum += pow(p_1.val[i]-p_2.val[i],2.);
    }
    return sqrt(sum);
}

double euclidean_distance_lab(CvScalar p_1, CvScalar p_2)
{
    // convert to Lab for better perceptual distance
    IplImage * convert = cvCreateImage( cvSize(2,1), 8, 3);
    cvSet2D(convert,0,0,p_1);
    cvSet2D(convert,0,1,p_2);
    cvCvtColor(convert,convert,CV_BGR2Lab);
    p_1 = cvGet2D(convert,0,0);
    p_2 = cvGet2D(convert,0,1);
    cvReleaseImage(&convert);
    
    return euclidean_distance(p_1, p_2);
}

CvScalar contour_average(CvContour* contour, IplImage* image)
{
    CvRect rect = ((CvContour*)contour)->rect;
    
    CvScalar average = cvScalarAll(0);
    int count = 0;
    for(int x = rect.x; x < (rect.x+rect.width); x++) {
        for(int y = rect.y; y < (rect.y+rect.height); y++) {
            if(cvPointPolygonTest(contour, cvPointTo32f(cvPoint(x,y)),0) == 100) {
                CvScalar s = cvGet2D(image,y,x);
                average.val[0] += s.val[0];
                average.val[1] += s.val[1];
                average.val[2] += s.val[2];
                // printf("B=%f, G=%f, R=%f\n",s.val[0],s.val[1],s.val[2]);
                
                count++;
            }
        }
    }
    
    for(int i = 0; i < 3; i++){
        average.val[i] /= count;
    }
    
    return average;
}

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
    // (converted to BGR order for comparison)
    CvScalar colorchecker_srgb[MACBETH_HEIGHT][MACBETH_WIDTH] =
        {
            {
                cvScalar(67,81,115),
                cvScalar(129,149,196),
                cvScalar(157,123,93),
                cvScalar(65,108,90),
                cvScalar(176,129,130),
                cvScalar(171,191,99)
            },
            {
                cvScalar(45,123,220),
                cvScalar(168,92,72),
                cvScalar(98,84,195),
                cvScalar(105,59,91),
                cvScalar(62,189,160),
                cvScalar(41,161,229)
            },
            {
                cvScalar(147,62,43),
                cvScalar(72,149,71),
                cvScalar(56,48,176),
                cvScalar(22,200,238),
                cvScalar(150,84,188),
                cvScalar(166,136,0)
            },
            {
                cvScalar(240,245,245),
                cvScalar(201,201,200),
                cvScalar(161,161,160),
                cvScalar(121,121,120),
                cvScalar(85,84,83),
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
        int adaptive_method = CV_ADAPTIVE_THRESH_MEAN_C;
        int threshold_type = CV_THRESH_BINARY_INV;
        int block_size = cvRound(
            MIN(macbeth_img->width,macbeth_img->height)*0.02)|1;
        printf("Using %d as block size\n", block_size);
        
        double offset = 6;
        
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
        
        CvSeq* stack = cvCreateSeq( 0, sizeof(*stack), sizeof(void*), storage );
        
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
                        rect = ((CvContour*)quad_contour)->rect;
                        
                        CvScalar average = contour_average((CvContour*)quad_contour, macbeth_img);
                        
                        CvBox2D box = cvMinAreaRect2(quad_contour,storage);
                        printf("Center: %f %f\n", box.center.x, box.center.y);
                        
                        double min_distance = MAX_RGB_DISTANCE;
                        CvPoint closest_color_idx = cvPoint(-1,-1);
                        for(int y = 0; y < MACBETH_HEIGHT; y++) {
                            for(int x = 0; x < MACBETH_WIDTH; x++) {
                                double distance = euclidean_distance_lab(average,colorchecker_srgb[y][x]);
                                if(distance < min_distance) {
                                    closest_color_idx.x = x;
                                    closest_color_idx.y = y;
                                    min_distance = distance;
                                }
                            }
                        }
                        
                        CvScalar closest_color = colorchecker_srgb[closest_color_idx.y][closest_color_idx.x];
                        printf("Closest color: %f %f %f (%d %d)\n",
                            closest_color.val[2],
                            closest_color.val[1],
                            closest_color.val[0],
                            closest_color_idx.x,
                            closest_color_idx.y
                        );
                        
                        cvDrawContours(
                            macbeth_img,
                            quad_contour,
                            cvScalar(255,0,0),
                            cvScalar(0,0,255),
                            0,
                            element_size
                        );
                        // cvCircle(
                        //     macbeth_img,
                        //     cvPointFrom32f(box.center),
                        //     element_size*6,
                        //     cvScalarAll(255),
                        //     -1
                        // );
                        cvCircle(
                            macbeth_img,
                            cvPointFrom32f(box.center),
                            element_size*6,
                            closest_color,
                            -1
                        );
                        cvCircle(
                            macbeth_img,
                            cvPointFrom32f(box.center),
                            element_size*4,
                            average,
                            -1
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

