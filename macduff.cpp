#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

IplImage * find_macbeth( const char *img )
{
  IplImage * macbeth_img = cvLoadImage( img,
      CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH );
  //printf( "Loading '%s', depth %u\n", flat, flat_img->depth );
  IplImage * image = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), 32, 1 );
  IplImage * image_8 = cvCreateImage( cvSize(macbeth_img->width, macbeth_img->height), 8, 1 );
  //printf( "Loading '%s', depth %u\n", img, image->depth );

  if( macbeth_img && image )
  {
    cvCornerHarris(macbeth_img,image,5,5,0.1);
    
    cvConvertScale( image, image_8, 255. );
    
    cvReleaseImage( &macbeth_img );
    cvReleaseImage( &image );
    
    return image_8;
  }

  if( macbeth_img ) cvReleaseImage( &macbeth_img );
  if( image ) cvReleaseImage( &image );

  return NULL;
}

int main( int argc, char *argv[] )
{
  if( argc < 2 )
  {
    fprintf( stderr, "Usage: %s image_file\n", argv[0] );
  }
  
  const char *img_file = argv[1];

  IplImage *out = find_macbeth( img_file );
  cvSaveImage( "result.png", out );
  cvReleaseImage( &out );

  return 0;
}

