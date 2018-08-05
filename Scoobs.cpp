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

            // get edge
            // this is for edge detection
            cv::Mat src = dlib::toMat(img);

            cv::Mat cannyImg;
            cv::Mat gray_image;

            cvtColor(src, gray_image, CV_RGB2GRAY );

            cv::Canny(src,cannyImg,35,120);

            imshow("Canny Image", cannyImg);

            array2d<rgb_pixel> newImg;
            load_image(img, argv[i]);
            newImg.set_size(img.nr(), img.nc());
            cout << "Image dimensions: " << img.nr() << " " << img.nc() << endl;
            cout << "Canny dim: " << cannyImg.rows << " " << cannyImg.cols << endl;
            int ksize = 5;
            int rsum, gsum, bsum;
            for (int fuckme = 0; fuckme < 100; fuckme++) {
            //loop through image
                for(int r=0; r<newImg.nr()-1; r++)
                {
                    for(int c=0; c<newImg.nc()-1; c++)
                    {
                        rsum=0;
                        gsum=0;
                        bsum=0;

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
                                cv::Vec3b newVal = cannyImg.at<cv::Vec3b>(xc,yc);
                                int isEdgeR = newVal[0];
                                int isEdgeG = newVal[1];
                                int isEdgeB = newVal[2];
                                if (isEdgeR != 0 || isEdgeG != 0 || isEdgeB != 0) {
                                    edgeCount++;
                                    if (edgeCount > 0) {
                                        shouldIgnore = 1;
                                    }
                                } else {
                                    count++;
                                    rsum+=(int)img[xc][yc].red;
                                    gsum+=(int)img[xc][yc].green;
                                    bsum+=(int)img[xc][yc].blue;
                                }
                            }
                        }

                        if (shouldIgnore == 0) {
                            int avgR = (int) ((double) rsum)/count;
                            int avgG = (int) ((double) gsum)/count;
                            int avgB = (int) ((double) bsum)/count;

                            if (avgR > 255 || avgB > 255 || avgG > 255) {
                                cout << "Squid " + avgR + avgB + avgG <<endl; 
                            }

                            newImg[r][c].red = avgR;
                            newImg[r][c].green = avgG;
                            newImg[r][c].blue = avgB;
                        }

                    }
                }
                
                img.swap(newImg);
            }

            // Now let's view our face poses on the screen.
            win.clear_overlay();
            win.set_image(img);
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
