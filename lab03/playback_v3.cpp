#include <opencv2/opencv.hpp>

#include <fstream>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include "HomographyData.h"
#include <string>
#include <vector>

using namespace cv;
using namespace std;
// ######################### config ##################################
#define VIDEO_FILE "robot.mp4"
#define HOMOGRAPHY_FILE "robot-homography.yml"

#define HACK_FLOOR_H 12
#define HACK_FLOOR_S 32
#define HACK_FLOOR_V 145

#define TRAIN_IMGS_SIZE 21
#define TEST_IMGS_SIZE 10
#define shift_h 5
#define shift_s 4
#define shift_v 4
// ##################################################################
// To do more
// 1) change 64,16,16  bin 
// 2)  using leave one out cv acc 0.95
// 3) F1  floor 0.98 and obstables 0.51
//
// ##  
template<typename Char, typename Traits, typename Allocator>
std::basic_string<Char, Traits, Allocator> operator *
(const std::basic_string<Char, Traits, Allocator> s, size_t n)
{
   std::basic_string<Char, Traits, Allocator> tmp = s;
   for (size_t i = 0; i < n; ++i)
   {
      tmp += s;
   }
   return tmp;
}
template<typename Char, typename Traits, typename Allocator>
std::basic_string<Char, Traits, Allocator> operator *
(size_t n, const std::basic_string<Char, Traits, Allocator>& s)
{
   return s * n;
}
// For pixels likely to be floor, highlight with a transparent red mask

void segmentFrame(cv::Mat &matFrame, double aProbFloorHS[64][16][16], double aProbNonFloorHS[64][16][16])
{ 
  cv:: Mat matFrameHSV;
  cvtColor(matFrame, matFrameHSV, COLOR_BGR2HSV);

   for(int iRow = 0; iRow < matFrame.rows;iRow++)
  {
      
      for(int iCol = 0; iCol < matFrame.cols; iCol++)
      {   
          Vec3b HSV = matFrameHSV.at<Vec3b>(iRow, iCol);
          double probHSVgivenFloor = aProbFloorHS[HSV[0]>>6][HSV[1]>>4][HSV[2]>>4];
          double probHSVgivenNonFloor = aProbNonFloorHS[HSV[0]>>6][HSV[1]>>4][HSV[2]>>4];
         // cout<<"probHSVgivenFloor :"<<probHSVgivenFloor<<endl;
        //  cout<<"probHSVgivenNonFloor :"<< probHSVgivenNonFloor<<endl;

         if(probHSVgivenFloor > probHSVgivenNonFloor)
          {
             // Likely floor pixel
             Vec3b &BGR = matFrame.at<Vec3b>(iRow, iCol);
              

             BGR[0] = (int) (0.5 * BGR[0] + 0.5 *0);
             BGR[1] = (int) (0.5 * BGR[1] + 0.5 *0);
             BGR[2] = (int) (0.5 * BGR[2] + 0.5 * 255);
           //  cout<<"probHSVgivenFloor > probHSVgivenNonFloor"<<endl;
           }
      }

  }
 


}
void displayFrame(cv::Mat& matFrameDisplay, int iFrame, int cFrames, HomographyData* pHomographyData) {
    for (int i = 0; i < pHomographyData->cPoints; i++) {
        cv::circle(matFrameDisplay, pHomographyData->aPoints[i], 10, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
    }
    imshow(VIDEO_FILE, matFrameDisplay);
    stringstream ss;
    ss << "Frame " << iFrame << "/" << cFrames;
    ss << ": hit <space> for next frame or 'q' to quit";
    //cv::displayOverlay(VIDEO_FILE, ss.str(), 0);  // for linux + qt
    putText(matFrameDisplay, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 3);
}
void getHistrograms(double aProbFloorHS[64][16][16], double aProbNonFloorHS[64][16][16])
{
   // Hard code a set for  "tranning" imgs and mask
   //
   std::vector<std::string> aImages,aMasks;
  
   for(int i_img =0; i_img< TRAIN_IMGS_SIZE; i_img++)
   {  
      int digit = (18-i_img)/10;
      std::string s = "0";

      //aImages[i_img] =
      string s1 ="../video-frames/frame-"+(s*digit) + std::to_string(i_img+1) + ".png";
      string s2 = "../video-frames/frame-"+ (s*digit) + std::to_string(i_img+1) + "-mask.png";
      
      aImages.push_back(s1);
      aMasks.push_back(s2);
     // std::cout<<aImages[i_img]<<std::endl;
      //aMasks[i_img] = 
    //  std::cout<<aMasks[i_img]<<std::endl;
   }

  int nImages = aImages.size();
  int cFloor = 0, cNonFloor = 0;
  for(int iImage =0; iImage < nImages; iImage++)
  {
      //  std::cout<<aImages[iImage]<<std::endl;
        cv::Mat matFrame = cv::imread(aImages[iImage]);
      //  imshow(aImages[iImage],matFrame);
      //  waitKey(0);  
        cv::Mat matFrameHSV;
        // convert current frame into HSV value 
        // use matFrame HSV real image convert into bin 
        // to find the index in the table 
        cvtColor(matFrame, matFrameHSV, COLOR_BGR2HSV);    
        cv::Mat matMask = cv::imread(aMasks[iImage]);
        cout<<"Image" <<aImages[iImage]<< ":"<<matFrame.cols<<"x"<<matFrame.rows<<endl;
        cout<<"Mask" <<aMasks[iImage]<<":" << matMask.cols <<"x" << matMask.rows<<endl;
 
         
        for(int iRow=0; iRow < matFrame.rows; iRow++)
        {
             for(int iCol =0; iCol < matFrame.cols; iCol++)
            {
                Vec3b HSV = matFrameHSV.at<Vec3b>(iRow,iCol);
                // to get index in the table from real image from HSV format
                int Hindex = HSV[0] >>6;
                int Sindex = HSV[1] >>4;
                int Vindex = HSV[2] >>4;
               
                // use mask black white image
                // which is BGR format
                
                Vec3b MaskBGR = matMask.at<Vec3b>(iRow, iCol);
               //  
               //Blue channel are more 
               //conclude that they are counted as floor
               //but on the most floor are white 
                if(MaskBGR[0] > 128)
                {
                  aProbFloorHS[Hindex][Sindex][Vindex] += 1;
                  cFloor++;
                //  cout<<"make table floor:"<<aProbFloorHS[Hindex][Sindex][Vindex]<<endl;
                }
                else
                {
                  aProbNonFloorHS[Hindex][Sindex][Vindex] +=1;
                  cNonFloor++;
                 // cout<<"make table non - floor:"<<aProbNonFloorHS[Hindex][Sindex][Vindex]<<endl;
                }
                 

                

             }

        }
    
   }
   cout<<"Got :"<< cFloor <<" floor pixels and " << cNonFloor << "non-floor pixels" <<endl;


   double totalPixels = cFloor + cNonFloor;
   cout<<"Total pixels:"<<totalPixels<<endl;
   double probfloor =  cFloor/totalPixels; 
   double nonprobfloor = cNonFloor/totalPixels;         
   
   cout<<"Probability of Floor"<<probfloor<<endl;
   cout<<"Probability of Non  "<<nonprobfloor<<endl;
  
   for(int iH =0; iH < 64; iH++)
   {
         for(int iS =0; iS < 16; iS++)
         {
            for(int iV=0; iV < 16; iV++)
            {
    
               aProbFloorHS[iH][iS][iV] /= cFloor;
               aProbNonFloorHS[iH][iS][iV] /= cNonFloor;
               aProbFloorHS[iH][iS][iV] *= probfloor; 
               aProbNonFloorHS[iH][iS][iV] *=nonprobfloor;
            }  
         }

   }





  
   
}
void Test(double aProbFloorHS[64][16][16], double aProbNonFloorHS[64][16][16])
{       
       std::vector<std::string> aImages,aMasks;

      // test set image 21 to 31 
       for(int iNimg =0; iNimg< TEST_IMGS_SIZE; iNimg++)
       {
          int digit = (18-(iNimg+9))/10;
          std::string s = "0";
          string s1 ="../video-frames/frame-"+(s *digit) + std::to_string(iNimg+22) + ".png";
          string s2 = "../video-frames/frame-"+ (s *digit) + std::to_string(iNimg+22) + "-mask.png";
      
          aImages.push_back(s1);
          aMasks.push_back(s2);
          

       //     for(int iRow = 0 ; iRow < 

       }
       int  nImages = aImages.size();
       cout<<"nImages in Test function :"<< nImages<<endl;
       int cCorrect = 0 , cIncorrect =0;
       for(int iImage =0; iImage < nImages; iImage++)
       {  
             cv::Mat matFrame = cv::imread(aImages[iImage]);
             cv::Mat matMask = cv::imread(aMasks[iImage]);
             cout<<"Image" <<aImages[iImage]<< ":"<<matFrame.cols<<"x"<<matFrame.rows<<endl;
             cout<<"Mask" <<aMasks[iImage]<<":" << matMask.cols <<"x" << matMask.rows<<endl;
            
             cv:: Mat matFrameHSV;
             cvtColor(matFrame, matFrameHSV, COLOR_BGR2HSV);


             for(int iRow=0; iRow < matFrame.rows; iRow++)
             {
                     for(int iCol =0; iCol < matFrame.cols; iCol++)
                     {
                        Vec3b HSV = matFrameHSV.at<Vec3b>(iRow,iCol);
                        Vec3b MaskBGR = matMask.at<Vec3b>(iRow, iCol);

                        // to get index in the table from real image from HSV format
                        int Hindex = HSV[0] >>6;
                        int Sindex = HSV[1] >>4;
                        int Vindex = HSV[2] >>4;
                       
                        // use mask black white image
                        // which is BGR format
                         double probHSVgivenFloor = aProbFloorHS[Hindex][Sindex][Vindex];
                         double probHSVgivenNonFloor = aProbNonFloorHS[Hindex][Sindex][Vindex];

                       bool isCurrentPixfloor = false;
                       // prediction part

                       if(probHSVgivenFloor > probHSVgivenNonFloor)
                       {
                           isCurrentPixfloor = true;
                       }
                        // get the label part
                       bool labelMask = false;
                       if(MaskBGR[0] > 128)
                       {
                         labelMask = true;
                       }

                       if(isCurrentPixfloor == labelMask)
                       {
                           cCorrect++;
                       }
                       else
                       {
                           cIncorrect++;
                       }
                   }

                    

 
             }
       }       
       cout<<"Corect :"<< cCorrect<<endl;    
       cout<<"Incorrect :"<< cIncorrect<<endl;
       cout<<"The accuracy is :"<<(cCorrect*100)/(cCorrect+cIncorrect)<<"%"<<endl;

}

int main()
{
    cv::Mat matFrameCapture;
    cv::Mat matFrameDisplay;
    int cFrames;

    HomographyData homographyData;
    //  define the 3D table  HSV each 32 level
    //
    YAML::Node config = YAML::LoadFile("../config.yaml");
  
   //waitKey(0);
    double aProbFloorHS[64][16][16]= {0};
    double aProbNonFloorHs[64][16][16]= {0};
   
   

 
   // compute the probability prob of floor and not for 
   // in order to draw and segment floor and other objects 
    getHistrograms(aProbFloorHS,aProbNonFloorHs);
    cout<<"Test process"<<endl;
    Test(aProbFloorHS,aProbNonFloorHs); 

    cv::VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened()) {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }
    
    cFrames = (int)videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

    // Create a named window that will be used later to display each frame
    cv::namedWindow(VIDEO_FILE, (unsigned int)cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);

    // Read homography from file
    if (!homographyData.read(HOMOGRAPHY_FILE)) {
        cerr << "ERROR! Unable to read homography data file " << HOMOGRAPHY_FILE << endl;
        return -1;
    }

    int iFrame = 0;
   
        
    while (true) {

        // Block for next frame

        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty()) {
            // End of video file
            break;
        }
         cout<<"Before segment the img"<<endl;
         segmentFrame(matFrameCapture,aProbFloorHS,aProbNonFloorHs);  
         cout<<" Done Segment image"<<endl;
         displayFrame(matFrameCapture, iFrame, cFrames, &homographyData);
         int iKey;
       
        do {
            iKey = cv::waitKey(10);
            if (getWindowProperty(VIDEO_FILE, cv::WND_PROP_VISIBLE) == 0) {
                return 0;
            }
            if (iKey == (int)'q' || iKey == (int)'Q') {
                return 0;
            }
        } while (iKey != (int)' ');
        iFrame++;
    }

    return 0;
}
