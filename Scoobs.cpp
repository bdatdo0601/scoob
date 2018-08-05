// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    The face detector we use is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
       One Millisecond Face Alignment with an Ensemble of Regression Trees by
       Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset (see
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
       C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
       300 faces In-the-wild challenge: Database and results. 
       Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
    You can get the trained model file from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
    Note that the license for the iBUG 300-W dataset excludes commercial use.
    So you should contact Imperial College London to find out if it's OK for
    you to use this model file in a commercial product.


    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = dlib::get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        dlib::deserialize(argv[1]) >> sp;

        image_window win, win_faces;
        // Loop over all the images provided on the command line.
        for (int i = 2; i < argc; ++i)
        {
            cout << "processing image " << argv[i] << endl;
            dlib::array2d<rgb_pixel> img;
            load_image(img, argv[i]);
            // Make the image larger so we can detect small faces.
            // pyramid_up(img);

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
            std::vector<dlib::rectangle> dets = detector(img);
            cout << "Number of faces detected: " << dets.size() << endl;

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]);
                cout << "number of parts detected: " << shape.num_parts() << endl;
                // shape will contains the list of all point on the image that is facial landmarks
                shapes.push_back(shape);
            }

            // this is for edge detection
            cv::Mat src = imread(argv[i]);

            cv::Mat cannyImg;
            cv::Mat gray_image;

            cvtColor(src, gray_image, CV_RGB2GRAY );

            cv::Canny(src,cannyImg,35,90);

            imshow("Canny Image", cannyImg);

            imshow("gray Image", gray_image);

            // Change the background from white to black, since that will help later to extract
            // better results during the use of Distance Transform
            for( int x = 0; x < src.rows; x++ ) {
                for( int y = 0; y < src.cols; y++ ) {
                    if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
                        src.at<Vec3b>(x, y)[0] = 0;
                        src.at<Vec3b>(x, y)[1] = 0;
                        src.at<Vec3b>(x, y)[2] = 0;
                    }
                }
            }
            // Show output image
            imshow("Black Background Image", src);

            // Create a kernel that we will use for accuting/sharpening our image
            Mat kernel = (Mat_<float>(3,3) <<
                    1,  1, 1,
                    1, -8, 1,
                    1,  1, 1); // an approximation of second derivative, a quite strong kernel
            // do the laplacian filtering as it is
            // well, we need to convert everything in something more deeper then CV_8U
            // because the kernel has some negative values,
            // and we can expect in general to have a Laplacian image with negative values
            // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
            // so the possible negative number will be truncated
            cv::Mat imgLaplacian;
            cv::Mat sharp = src; // copy source image to another temporary one
            filter2D(sharp, imgLaplacian, CV_32F, kernel);
            src.convertTo(sharp, CV_32F);
            cv::Mat imgResult = sharp - imgLaplacian;
            // convert back to 8bits gray scale
            imgResult.convertTo(imgResult, CV_8UC3);
            imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
            // imshow( "Laplace Filtered Image", imgLaplacian );
            cv::imshow( "New Sharped Image", imgResult );

            // Create binary image from source image
            cv::Mat bw;
            cv::cvtColor(src, bw, CV_BGR2GRAY);
            cv::threshold(bw, bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            cv::imshow("Binary Image", bw);

            // Perform the distance transform algorithm
            cv::Mat dist;
            cv::distanceTransform(bw, dist, CV_DIST_L2, 3);
            // Normalize the distance image for range = {0.0, 1.0}
            // so we can visualize and threshold it
            cv::normalize(dist, dist, 0, 1., NORM_MINMAX);
            imshow("Distance Transform Image", dist);

            // Threshold to obtain the peaks
            // This will be the markers for the foreground objects
            cv::threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
            // Dilate a bit the dist image
            cv::Mat kernel1 = Mat::ones(3, 3, CV_8U);
            cv::dilate(dist, dist, kernel1);
            cv::imshow("Peaks", dist);

            // Create the CV_8U version of the distance image
            // It is needed for findContours()
            cv::Mat dist_8u;
            dist.convertTo(dist_8u, CV_8U);
            // Find total markers
            std::vector<std::vector<Point> > contours;
            findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            // Create the marker image for the watershed algorithm
            cv::Mat markers = Mat::zeros(dist.size(), CV_32S);
            // Draw the foreground markers
            for (size_t i = 0; i < contours.size(); i++)
            {
                drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
            }
            // Draw the background marker
            circle(markers, Point(5,5), 3, Scalar(255), -1);
            imshow("Markers", markers*10000);

            // Perform the watershed algorithm
            watershed(imgResult, markers);
            cv::Mat mark;
            markers.convertTo(mark, CV_8U);
            bitwise_not(mark, mark);
            imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
            // image looks like at that point
            // Generate random colors
            std::vector<Vec3b> colors;
            for (size_t i = 0; i < contours.size(); i++)
            {
                int b = theRNG().uniform(0, 256);
                int g = theRNG().uniform(0, 256);
                int r = theRNG().uniform(0, 256);
                colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
            }
            // Create the result image
            Mat dst = Mat::zeros(markers.size(), CV_8UC3);
            // Fill labeled objects with random colors
            for (int i = 0; i < markers.rows; i++)
            {
                for (int j = 0; j < markers.cols; j++)
                {
                    int index = markers.at<int>(i,j);
                    if (index > 0 && index <= static_cast<int>(contours.size()))
                    {
                        dst.at<Vec3b>(i,j) = colors[index-1];
                    }
                }
            }
            // Visualize the final image
            imshow("Final Result", dst);

            dlib::array2d<rgb_pixel> dlibImage;
            dlib::assign_image(dlibImage, dlib::cv_image<rgb_pixel>(dst));

            // Now let's view our face poses on the screen.
            win.clear_overlay();
            win.set_image(dlibImage);
            win.add_overlay(render_face_detections(shapes));
            
            cv::waitKey(0);

            cout << "Hit enter to process the next image..." << endl;
            cin.get();
        }
    }
    catch (exception &e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------
