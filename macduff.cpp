#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "colorchecker.h"

#define MAX_CONTOUR_APPROX  7

#define MAX_RGB_DISTANCE 444

CvRect contained_rectangle(CvBox2D box)
{
    return cvRect(box.center.x - box.size.width/4,
                  box.center.y - box.size.height/4,
                  box.size.width/2,
                  box.size.height/2);
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
                
                count++;
            }
        }
    }
    
    for(int i = 0; i < 3; i++){
        average.val[i] /= count;
    }
    
    return average;
}

void * cvBox(CvBox2D box, IplImage *image, CvScalar color, int thickness)
{
    CvPoint2D32f pt[4];
    
    cvBoxPoints(box, pt);
    
    for(int i = 0; i < 4; i++) {
        cvLine(image, cvPointFrom32f(pt[i]), cvPointFrom32f(pt[(i+1)%4]), color, thickness);
        cvCircle(image, cvPointFrom32f(pt[i]), thickness, cvScalarAll((255/4)*i), -1);
    }
}

void rotate_box(CvPoint2D32f * box_corners)
{
    CvPoint2D32f last = box_corners[3];
    for(int i = 3; i > 0; i--) {
        box_corners[i] = box_corners[i-1];
    }
    box_corners[0] = last;
}

double check_colorchecker(CvMat * colorchecker)
{
    double difference = 0;
    
    for(int x = 0; x < MACBETH_WIDTH; x++) {
        for(int y = 0; y < MACBETH_HEIGHT; y++) {
            CvScalar known_value = colorchecker_srgb[y][x];
            CvScalar test_value = cvGet2D(colorchecker,y,x);
            for(int i = 0; i < 3; i++){
                difference += pow(known_value.val[i]-test_value.val[i],2);
            }
        }
    }
    
    return difference;
}

void * draw_colorchecker(CvMat * colorchecker_values, CvMat * colorchecker_points, IplImage * image, int size)
{
    for(int x = 0; x < MACBETH_WIDTH; x++) {
        for(int y = 0; y < MACBETH_HEIGHT; y++) {
            CvScalar this_color = cvGet2D(colorchecker_values,y,x);
            CvScalar this_point = cvGet2D(colorchecker_points,y,x);
            
            cvCircle(
                image,
                cvPoint(this_point.val[0],this_point.val[1]),
                size,
                colorchecker_srgb[y][x],
                -1
            );
            
            cvCircle(
                image,
                cvPoint(this_point.val[0],this_point.val[1]),
                size/2,
                this_color,
                -1
            );
        }
    }
}

ColorChecker find_colorchecker(CvSeq * quads, CvSeq * boxes, CvMemStorage *storage, IplImage *image, IplImage *original_image)
{
    CvPoint2D32f box_corners[4];
    
    CvMat* points = cvCreateMat( boxes->total , 1, CV_32FC2 );
    for(int i = 0; i < boxes->total; i++)
    {
        CvBox2D box = (*(CvBox2D*)cvGetSeqElem(boxes, i));
        cvSet1D(points, i, cvScalar(box.center.x,box.center.y));
    }
    CvBox2D passport_box = cvMinAreaRect2(points,storage);
    cvBoxPoints(passport_box, box_corners);
    
    // cvBox(passport_box, image, cvScalarAll(128), 10);
    
    if(euclidean_distance(cvPointFrom32f(box_corners[0]),cvPointFrom32f(box_corners[1])) <
       euclidean_distance(cvPointFrom32f(box_corners[1]),cvPointFrom32f(box_corners[2]))) {
        fprintf(stderr,"Box is upright, rotating\n");
        rotate_box(box_corners);
    }

    double horizontal_spacing = euclidean_distance(
        cvPointFrom32f(box_corners[0]),cvPointFrom32f(box_corners[1]))/(double)(MACBETH_WIDTH-1);
    double vertical_spacing = euclidean_distance(
        cvPointFrom32f(box_corners[1]),cvPointFrom32f(box_corners[2]))/(double)(MACBETH_HEIGHT-1);
    double horizontal_slope = (box_corners[1].y - box_corners[0].y)/(box_corners[1].x - box_corners[0].x);
    double horizontal_mag = sqrt(1+pow(horizontal_slope,2));
    double vertical_slope = (box_corners[3].y - box_corners[0].y)/(box_corners[3].x - box_corners[0].x);
    double vertical_mag = sqrt(1+pow(vertical_slope,2));
    double horizontal_orientation = box_corners[0].x < box_corners[1].x ? -1 : 1;
    double vertical_orientation = box_corners[0].y < box_corners[3].y ? -1 : 1;
        
    fprintf(stderr,"Spacing is %f %f\n",horizontal_spacing,vertical_spacing);
    fprintf(stderr,"Slope is %f %f\n", horizontal_slope,vertical_slope);
    
    int average_size = 0;
    for(int i = 0; i < boxes->total; i++)
    {
        CvBox2D box = (*(CvBox2D*)cvGetSeqElem(boxes, i));
        
        CvRect rect = contained_rectangle(box);
        average_size += MIN(rect.width, rect.height);
    }
    average_size /= boxes->total;
    
    fprintf(stderr,"Average contained rect size is %d\n", average_size);
    
    CvMat * this_colorchecker = cvCreateMat(MACBETH_HEIGHT, MACBETH_WIDTH, CV_32FC3);
    CvMat * this_colorchecker_points = cvCreateMat( MACBETH_HEIGHT, MACBETH_WIDTH, CV_32FC2 );
    
    // calculate the averages for our oriented colorchecker
    for(int x = 0; x < MACBETH_WIDTH; x++) {
        for(int y = 0; y < MACBETH_HEIGHT; y++) {
            CvPoint2D32f row_start;
            row_start.x = box_corners[0].x + vertical_spacing * y * (1 / vertical_mag);
            row_start.y = box_corners[0].y + vertical_spacing * y * (vertical_slope / vertical_mag);
            
            CvRect rect = cvRect(0,0,average_size,average_size);
            
            rect.x = row_start.x - horizontal_spacing * x * ( 1 / horizontal_mag ) * horizontal_orientation;
            rect.y = row_start.y - horizontal_spacing * x * ( horizontal_slope / horizontal_mag ) * vertical_orientation;
            
            cvSet2D(this_colorchecker_points, y, x, cvScalar(rect.x,rect.y));
            
            rect.x = rect.x - average_size / 2;
            rect.y = rect.y - average_size / 2;
            
            // cvRectangle(
            //     image,
            //     cvPoint(rect.x,rect.y),
            //     cvPoint(rect.x+rect.width, rect.y+rect.height),
            //     cvScalarAll(0),
            //     10
            // );
            
            CvScalar average_color = rect_average(rect, original_image);
            
            cvSet2D(this_colorchecker,y,x,average_color);
        }
    }
    
    double orient_1_error = check_colorchecker(this_colorchecker);
    cvFlip(this_colorchecker,NULL,-1);
    double orient_2_error = check_colorchecker(this_colorchecker);
    
    fprintf(stderr,"Orientation 1: %f\n",orient_1_error);
    fprintf(stderr,"Orientation 2: %f\n",orient_2_error);
    
    if(orient_1_error < orient_2_error) {
        cvFlip(this_colorchecker,NULL,-1);
    }
    else {
        cvFlip(this_colorchecker_points,NULL,-1);
    }
    
    // draw_colorchecker(this_colorchecker,this_colorchecker_points,image,average_size);
    
    ColorChecker found_colorchecker;
    
    found_colorchecker.error = MIN(orient_1_error,orient_2_error);
    found_colorchecker.values = this_colorchecker;
    found_colorchecker.points = this_colorchecker_points;
    found_colorchecker.size = average_size;
    
    return found_colorchecker;
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
            (d3*1.1 > d4 && d4*1.1 > d3 && d3*d4 < area*1.5 && area > min_size &&
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
        
    IplImage * macbeth_original = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), macbeth_img->depth, macbeth_img->nChannels );
    cvCopy(macbeth_img, macbeth_original);
        
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
        fprintf(stderr,"Using %d as block size\n", block_size);
        
        double offset = 6;
        
        // do an adaptive threshold on each channel
        for(int i = 0; i < 3; i++) {
            cvAdaptiveThreshold(macbeth_split[i], macbeth_split_thresh[i], 255, adaptive_method, threshold_type, block_size, offset);
        }
        
        IplImage * adaptive = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), IPL_DEPTH_8U, 1 );
        
        // OR the binary threshold results together
        cvOr(macbeth_split_thresh[0],macbeth_split_thresh[1],adaptive);
        cvOr(macbeth_split_thresh[2],adaptive,adaptive);
        
        for(int i = 0; i < 3; i++) {
            cvReleaseImage( &(macbeth_split[i]) );
            cvReleaseImage( &(macbeth_split_thresh[i]) );
        }
                
        int element_size = (block_size/10)+2;
        fprintf(stderr,"Using %d as element size\n", element_size);
        
        // do an opening on the threshold image
        IplConvKernel * element = cvCreateStructuringElementEx(element_size,element_size,element_size/2,element_size/2,CV_SHAPE_RECT);
        cvMorphologyEx(adaptive,adaptive,NULL,element,CV_MOP_OPEN);
        cvReleaseStructuringElement(&element);
        
        CvMemStorage* storage = cvCreateMemStorage(0);
        
        CvSeq* initial_quads = cvCreateSeq( 0, sizeof(*initial_quads), sizeof(void*), storage );
        CvSeq* initial_boxes = cvCreateSeq( 0, sizeof(*initial_boxes), sizeof(CvBox2D), storage );
        
        // find contours in the threshold image
        CvSeq * contours = NULL;
        cvFindContours(adaptive,storage,&contours);
        
        int min_size = (macbeth_img->width*macbeth_img->height)/
            (MACBETH_SQUARES*100);
        
        if(contours) {
            int count = 0;
            
            for( CvSeq* c = contours; c != NULL; c = c->h_next) {
                CvRect rect = ((CvContour*)c)->rect;
                // only interested in contours with these restrictions
                if(CV_IS_SEQ_HOLE(c) && rect.width*rect.height >= min_size) {
                    // only interested in quad-like contours
                    CvSeq * quad_contour = find_quad(c, storage, min_size);
                    if(quad_contour) {
                        cvSeqPush( initial_quads, &quad_contour );
                        count++;
                        rect = ((CvContour*)quad_contour)->rect;
                        
                        CvScalar average = contour_average((CvContour*)quad_contour, macbeth_img);
                        
                        CvBox2D box = cvMinAreaRect2(quad_contour,storage);
                        cvSeqPush( initial_boxes, &box );
                        
                        // fprintf(stderr,"Center: %f %f\n", box.center.x, box.center.y);
                        
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
                        // fprintf(stderr,"Closest color: %f %f %f (%d %d)\n",
                        //     closest_color.val[2],
                        //     closest_color.val[1],
                        //     closest_color.val[0],
                        //     closest_color_idx.x,
                        //     closest_color_idx.y
                        // );
                        
                        // cvDrawContours(
                        //     macbeth_img,
                        //     quad_contour,
                        //     cvScalar(255,0,0),
                        //     cvScalar(0,0,255),
                        //     0,
                        //     element_size
                        // );
                        // cvCircle(
                        //     macbeth_img,
                        //     cvPointFrom32f(box.center),
                        //     element_size*6,
                        //     cvScalarAll(255),
                        //     -1
                        // );
                        // cvCircle(
                        //     macbeth_img,
                        //     cvPointFrom32f(box.center),
                        //     element_size*6,
                        //     closest_color,
                        //     -1
                        // );
                        // cvCircle(
                        //     macbeth_img,
                        //     cvPointFrom32f(box.center),
                        //     element_size*4,
                        //     average,
                        //     -1
                        // );
                        // CvRect rect = contained_rectangle(box);
                        // cvRectangle(
                        //     macbeth_img,
                        //     cvPoint(rect.x,rect.y),
                        //     cvPoint(rect.x+rect.width, rect.y+rect.height),
                        //     cvScalarAll(0),
                        //     element_size
                        // );
                    }
                }
            }
            
            ColorChecker found_colorchecker;

            fprintf(stderr,"%d initial quads found", initial_quads->total);
            if(count > MACBETH_SQUARES) {
                fprintf(stderr," (probably a Passport)\n");
                
                CvMat* points = cvCreateMat( initial_quads->total , 1, CV_32FC2 );
                CvMat* clusters = cvCreateMat( initial_quads->total , 1, CV_32SC1 );
                
                CvSeq* partitioned_quads[2];
                CvSeq* partitioned_boxes[2];
                for(int i = 0; i < 2; i++) {
                    partitioned_quads[i] = cvCreateSeq( 0, sizeof(**partitioned_quads), sizeof(void*), storage );
                    partitioned_boxes[i] = cvCreateSeq( 0, sizeof(**partitioned_boxes), sizeof(CvBox2D), storage );
                }
                
                // set up the points sequence for cvKMeans2, using the box centers
                for(int i = 0; i < initial_quads->total; i++) {
                    CvBox2D box = (*(CvBox2D*)cvGetSeqElem(initial_boxes, i));
                    
                    cvSet1D(points, i, cvScalar(box.center.x,box.center.y));
                }
                
                // partition into two clusters: passport and colorchecker
                cvKMeans2( points, 2, clusters, 
                           cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,
                                           10, 1.0 ) );
        
                for(int i = 0; i < initial_quads->total; i++) {
                    CvPoint2D32f pt = ((CvPoint2D32f*)points->data.fl)[i];
                    int cluster_idx = clusters->data.i[i];
                    
                    cvSeqPush( partitioned_quads[cluster_idx],
                               cvGetSeqElem(initial_quads, i) );
                    cvSeqPush( partitioned_boxes[cluster_idx],
                               cvGetSeqElem(initial_boxes, i) );

                    // cvCircle(
                    //     macbeth_img,
                    //     cvPointFrom32f(pt),
                    //     element_size*2,
                    //     cvScalar(255*cluster_idx,0,255-(255*cluster_idx)),
                    //     -1
                    // );
                }
                
                ColorChecker partitioned_checkers[2];
                
                // check each of the two partitioned sets for the best colorchecker
                for(int i = 0; i < 2; i++) {
                    partitioned_checkers[i] =
                        find_colorchecker(partitioned_quads[i], partitioned_boxes[i],
                                      storage, macbeth_img, macbeth_original);
                }
                
                // use the colorchecker with the lowest error
                found_colorchecker = partitioned_checkers[0].error < partitioned_checkers[1].error ?
                    partitioned_checkers[0] : partitioned_checkers[1];
                
                cvReleaseMat( &points );
                cvReleaseMat( &clusters );
            }
            else { // just one colorchecker to test
                fprintf(stderr,"\n");
                found_colorchecker = find_colorchecker(initial_quads, initial_boxes,
                                  storage, macbeth_img, macbeth_original);
            }
            
            // render the found colorchecker
            draw_colorchecker(found_colorchecker.values,found_colorchecker.points,macbeth_img,found_colorchecker.size);
            
            // print out the colorchecker info
            for(int y = 0; y < MACBETH_HEIGHT; y++) {            
                for(int x = 0; x < MACBETH_WIDTH; x++) {
                    CvScalar this_value = cvGet2D(found_colorchecker.values,y,x);
                    CvScalar this_point = cvGet2D(found_colorchecker.points,y,x);
                    
                    printf("%.0f,%.0f,%.0f,%.0f,%.0f\n",
                        this_point.val[0],this_point.val[1],
                        this_value.val[2],this_value.val[1],this_value.val[0]);
                }
            }
            printf("%0.f\n%f\n",found_colorchecker.size,found_colorchecker.error);
            
        }
                
        cvReleaseMemStorage( &storage );
        
        if( macbeth_original ) cvReleaseImage( &macbeth_original );
        if( adaptive ) cvReleaseImage( &adaptive );
        
        return macbeth_img;
    }

    if( macbeth_img ) cvReleaseImage( &macbeth_img );

    return NULL;
}

int main( int argc, char *argv[] )
{
    if( argc < 2 )
    {
        fprintf( stderr, "Usage: %s image_file [output_image]\n", argv[0] );
        return 1;
    }

    const char *img_file = argv[1];

    IplImage *out = find_macbeth( img_file );
    if( argc == 3) {
        cvSaveImage( argv[2], out );
    }
    cvReleaseImage( &out );

    return 0;
}

