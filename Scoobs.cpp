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
#include <algorithm> 
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

cv::Point getPointFromDlib(full_object_detection face, int location) {
    dlib::point pt = face.part(location);
    return cv::Point(pt.x(), pt.y());
}

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
            std::vector<full_object_detection> shapes;
            cout << "processing image " << argv[i] << endl;
            dlib::array2d<rgb_pixel> img;
            load_image(img, argv[i]);
            array2d<rgb_pixel> newImg;
            newImg.set_size(img.nr(), img.nc());
            // get edge
            // this is for edge detection
            cv::Mat src = dlib::toMat(img);

            cv::Mat cannyImg;
            cv::Mat gray_image;

            cvtColor(src, gray_image, CV_RGB2GRAY);

            cv::Canny(src, cannyImg, 80, 90);

            double start_time = omp_get_wtime();
            #pragma omp parallel 
            {
                
                // Make the image larger so we can detect small faces.
                // pyramid_up(img);

                #pragma omp single 
                {
                    // Now tell the face detector to give us a list of bounding boxes
                    // around all the faces in the image.
                    std::vector<dlib::rectangle> dets = detector(img);
                    cout << "Number of faces detected: " << dets.size() << endl;

                    // Now we will go ask the shape_predictor to tell us the pose of
                    // each face we detected.
                    for (unsigned long j = 0; j < dets.size(); ++j)
                    {
                        full_object_detection shape = sp(img, dets[j]);
                        cout << "number of parts detected: " << shape.num_parts() << endl;
                        // shape will contains the list of all point on the image that is facial landmarks
                        shapes.push_back(shape);
                    }
                }
                // imshow("Original Image", src);
                
                #pragma omp single
                {
                    cout << "Image dimensions: " << img.nr() << " " << img.nc() << endl;
                    cout << "Canny dim: " << cannyImg.rows << " " << cannyImg.cols << endl;
                }
                int ksize = 10;
                for (int blurRate = 0; blurRate < 10; blurRate++)
                {
                    //loop through image
                    #pragma omp for
                    for(int r=0; r<newImg.nr()-1; r++)
                    {
                        for(int c=0; c<newImg.nc()-1; c++)
                        {
                            int rsum=0;
                            int gsum=0;
                            int bsum=0;

                            double left = 0;
                            double top = 0;
                            double right = newImg.nc() - 1;
                            int bottom = newImg.nr() - 1;
                            int rstart = (r-(ksize/2)) < 0 ? 0 : (r-(ksize/2)); //std::max(0, );
                            int rend = (r+(ksize/2)) > right ? right: (r+(ksize/2)); //std::min(r+(ksize/2), ); 
                            int cstart = (c-(ksize/2)) < 0 ? 0 : (c-(ksize/2));// std::max(0, (c-(ksize/2)));
                            int cend = (c+(ksize/2)) > bottom ? bottom : (c+(ksize/2)); // std::min(c+(ksize/2),  );

                            int count = 0;
                            int edgeCount = 0;
                            int shouldIgnore = 0;
                            //loop through kmatrix
                            for(int xc=rstart; xc<=rend; xc++)
                            {
                                for(int yc=cstart; yc<=cend; yc++)
                                {
                                    //handle is it edge or is it part of another block here
                                    cv::Vec3b newVal = cannyImg.at<cv::Vec3b>(xc,yc/3);
                                    int isEdgeR = newVal[0];
                                    int isEdgeG = newVal[1];
                                    int isEdgeB = newVal[2];
                                    if (isEdgeR != 0 || isEdgeG != 0 || isEdgeB != 0) {
                                        edgeCount++;
                                        if (edgeCount > ksize*(ksize-1)) {
                                            shouldIgnore = 1;
                                        }
                                        newImg[xc][yc].red = 0;
                                        newImg[xc][yc].green = 0;
                                        newImg[xc][yc].blue = 0;
                                    } else {
                                        count++;
                                        rsum+=(int)img[xc][yc].red;
                                        gsum+=(int)img[xc][yc].green;
                                        bsum+=(int)img[xc][yc].blue;
                                    }
                                }
                            }
            
                            if (shouldIgnore == 0 && count > 0) {
                                int avgR = (int) ((double) rsum)/count;
                                int avgG = (int) ((double) gsum)/count;
                                int avgB = (int) ((double) bsum)/count;

                                newImg[r][c].red = avgR;
                                newImg[r][c].green = avgG;
                                newImg[r][c].blue = avgB;
                            }

                        }
                    }
                    img.swap(newImg);
                }
            }

            cv::Mat dst;
            cv::cvtColor(dlib::toMat(img), dst, CV_BGR2RGB);
            double end_time = omp_get_wtime();

            cout << "Time Elapsed " << end_time - start_time << endl;
            // convert facial landmark to opencv
            for (int i = 0; i < shapes.size(); i++)
            {
                dlib::full_object_detection face = shapes[i];
                std::vector<cv::Point> left_eyebrows;
                for (int point = 17; point <= 21; point++)
                {
                    left_eyebrows.push_back(getPointFromDlib(face, point));
                }

                std::vector<cv::Point> right_eyebrows;
                for (int point = 22; point <= 26; point++)
                {
                    right_eyebrows.push_back(getPointFromDlib(face, point));
                }

                std::vector<cv::Point> left_eye;
                for (int point = 36; point <= 38; point++)
                {
                    left_eye.push_back(getPointFromDlib(face, point));
                }
                dlib::point point_38 = face.part(38);
                dlib::point point_39 = face.part(39);
                int avg39Y = (((point_39.y() + point_38.y()) / (2.0)));
                left_eye.push_back(cv::Point(point_39.x(), avg39Y));
                int avgLeftEyeY = (point_39.y() + point_38.y()) / (2.0);
                int avgLeftEyeX = (point_39.x() + point_38.x()) / (2.0);
                int radius_left_eye = sqrt(pow(point_38.x() - avgLeftEyeX, 2) + pow(point_39.y() - avgLeftEyeY, 2));

                std::vector<cv::Point> right_eye;
                dlib::point point_43 = face.part(43);
                dlib::point point_42 = face.part(42);
                int avg42Y = (((point_42.y() + point_43.y()) / (2.0)));
                right_eye.push_back(cv::Point(point_42.x(), avg42Y));
                int avgRightEyeY = (point_42.y() + point_43.y()) / (2.0);
                int avgRightEyeX = (point_42.x() + point_43.x()) / (2.0);
                int radius_right_eye = sqrt(pow(point_43.x() - avgRightEyeX, 2) + pow(point_42.y() - avgRightEyeY, 2));
                for (int point = 43; point <= 45; point++)
                {
                    right_eye.push_back(getPointFromDlib(face, point));
                }

                std::vector<cv::Point> mouth;
                mouth.push_back(getPointFromDlib(face, 48));
                mouth.push_back(getPointFromDlib(face, 67));
                mouth.push_back(getPointFromDlib(face, 66));
                mouth.push_back(getPointFromDlib(face, 65));
                mouth.push_back(getPointFromDlib(face, 54));

                std::vector<cv::Point> face_border;
                for (int point = 5; point <= 16; point++)
                {
                    face_border.push_back(getPointFromDlib(face, point));
                }

                cv::circle(dst, getPointFromDlib(face, 32), radius_left_eye / 4, Scalar(1, 1, 1), -1);
                cv::circle(dst, getPointFromDlib(face, 34), radius_left_eye / 4, Scalar(1, 1, 1), -1);
                cv::circle(dst, cv::Point(point_38.x(), point_39.y()), radius_left_eye, Scalar(1, 1, 1), -1);
                cv::circle(dst, cv::Point(point_43.x(), point_42.y()), radius_right_eye, Scalar(1, 1, 1), -1);

                // draws the curve using polylines and line width (BLACK)
                cv::polylines(dst, mouth, false, Scalar(1, 1, 1), (point_39.x() - point_38.x()) / 4.25);
                cv::polylines(dst, right_eye, false, Scalar(1, 1, 1), (point_39.x() - point_38.x()) / 4.2);
                cv::polylines(dst, left_eye, false, Scalar(1, 1, 1), (point_39.x() - point_38.x()) / 4.2);
                cv::polylines(dst, left_eyebrows, false, Scalar(1, 1, 1), (point_39.x() - point_38.x()) / 6.15);
                cv::polylines(dst, right_eyebrows, false, Scalar(1, 1, 1), (point_39.x() - point_38.x()) / 6.15);
                cv::polylines(dst, face_border, false, Scalar(1, 1, 1), (point_39.x() - point_38.x()) / 6.15);
            }
            
            // Now let's view our face poses on the screen for dlib.
            // win.clear_overlay();
            // win.set_image(img);
            // win.add_overlay(render_face_detections(shapes));
            imshow("result", dst);
            

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
