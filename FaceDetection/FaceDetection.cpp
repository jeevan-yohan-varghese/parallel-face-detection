//C:/Users/jeeva/Downloads/reference.png
//C:/Users/jeeva/Videos/anitta_chechi.mp4
//C:/Users/jeeva/Downloads/testvideo.mp4
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    VideoCapture cap("C:/Users/jeeva/Downloads/testvideo6.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    CascadeClassifier face_cascade;
    face_cascade.load("haarcascade_frontalface_default.xml");

    bool continue_flag = true;
    int count = 0;
    int last_write_count = -100;

#pragma omp parallel
    {
        Mat frame, gray;
        vector<Rect> faces;

        while (continue_flag) {
            bool read_flag;

#pragma omp critical
            {
                read_flag = cap.read(frame);
            }

            if (!read_flag)
                continue;

            cvtColor(frame, gray, COLOR_BGR2GRAY);

            // Resize the frame to a maximum width of 800
            if (frame.size().width > 800) {
                double scale = 800.0 / frame.size().width;
                resize(frame, frame, Size(0, 0), scale, scale);
            }

            face_cascade.detectMultiScale(gray, faces, 1.3, 5);

            for (int i = 0; i < faces.size(); i++) {
               // rectangle(frame, faces[i], Scalar(0, 0, 255), 2);
                rectangle(gray, faces[i], Scalar(0, 0, 255), 2);

                
                // Export detected faces as images
                if (count % 2 == 0) {
                    Mat face_roi = gray(faces[i]);
                    //Mat face_roi = frame(faces[i]);
                    string filename = "face_" + to_string(count) + ".jpg";
                    imwrite(filename, face_roi);
                    last_write_count = count;
                }
                    
                
                count++;
                
            }

#pragma omp critical
            {
                //imshow("Face Detection", frame);
                imshow("Face Detection", gray);
                if (waitKey(10) == 27)
                    continue_flag = false;
            }
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
