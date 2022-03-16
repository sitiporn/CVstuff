#include <opencv2/opencv.hpp>
#include <numeric> 
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include "HomographyData.h"
#include <string>
#include <vector>
#include<algorithm>

using namespace cv;
using namespace std;
// ######################### config ##################################
#define VIDEO_FILE "robot.mp4"
#define HOMOGRAPHY_FILE "robot-homography.yml"
#define TRAIN_IMGS_SIZE 21
#define TEST_IMGS_SIZE 10
#define datasetSize 31


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

void segmentFrame(cv::Mat &matFrame, double ***aProbFloorHS, double ***aProbNonFloorHS,const int shiftH,const int shiftS,const int shiftV)
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
void getHistrograms(double ***aProbFloorHS, double ***aProbNonFloorHS,const int shiftH,const int shiftS,const int shiftV)
{
   // Hard code a set for  "tranning" imgs and mask
   //
    std::vector<std::string> aImages,aMasks;
  

    for(int iNimg =0; iNimg< datasetSize; iNimg++)
   {
      int digit;
      std::string s;

      if(((iNimg+1)/10) <= 1e-5){
          s =  "00"; 
      }
      else{
          s = "0";
      }
      
      string s1 ="../video-frames/frame-"+ s + std::to_string(iNimg+1) + ".png";
      string s2 = "../video-frames/frame-"+ s + std::to_string(iNimg+1) + "-mask.png";
  
      aImages.push_back(s1);
      aMasks.push_back(s2);
 
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
                int Hindex = HSV[0] >>shiftH;
                int Sindex = HSV[1] >>shiftS;
                int Vindex = HSV[2] >>shiftV;
               
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

float validateModel(double ***aProbFloorHS, double ***aProbNonFloorHS,const int shiftH,const int shiftS,const int shiftV,std::vector<std::string> aImages,std::vector<std::string> aMasks,vector<int> train_idx,int test_idx,bool train_only)
{
     int cCorrect = 0 , cIncorrect =0;
   // Train
     int countloop = 0;
     int cFloor = 0, cNonFloor = 0;
     // Train models and create tables
     //
     cout<<"Train idx:"<<train_idx.at(0)<<" "<<train_idx.at(29)<<endl;
     cout<<"Test idx:"<<test_idx<<endl;
     cout<<"Train Size :"<<train_idx.size()<<endl;

     for(auto& iImage : train_idx)
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
                int Hindex = HSV[0] >>shiftH;
                int Sindex = HSV[1] >>shiftS;
                int Vindex = HSV[2] >>shiftV;
               
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

    //calculate probability to make models
   cout<<"In Validate models !!!"<<endl;
   cout<<"Got :"<< cFloor <<" floor pixels and " << cNonFloor << "non-floor pixels" <<endl;


   double totalPixels = cFloor + cNonFloor;
   cout<<"Total pixels:"<<totalPixels<<endl;
   double probfloor =  cFloor/totalPixels; 
   double nonprobfloor = cNonFloor/totalPixels;         
   
   cout<<"Probability of Floor :"<<probfloor<<endl;
   cout<<"Probability of Non :"<<nonprobfloor<<endl;
  
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

  if(train_only==true)
  {
     return 0; 

  }
  // Test models
  cv::Mat matFrameTest = cv::imread(aImages[test_idx]);
  cv::Mat matMaskTest = cv::imread(aMasks[test_idx]);
  cout<<"Image" <<aImages[test_idx]<< ":"<<matFrameTest.cols<<"x"<<matFrameTest.rows<<endl;
  cout<<"Mask" <<aMasks[test_idx]<<":" << matMaskTest.cols <<"x" << matMaskTest.rows<<endl;
      
  cv:: Mat matFrameHSVTest;
  cvtColor(matFrameTest, matFrameHSVTest, COLOR_BGR2HSV);


  for(int iRow=0; iRow < matFrameTest.rows; iRow++)
  {
      for(int iCol =0; iCol < matFrameTest.cols; iCol++)
      {
          Vec3b HSV = matFrameHSVTest.at<Vec3b>(iRow,iCol);
          Vec3b MaskBGR = matMaskTest.at<Vec3b>(iRow, iCol);

        // to get index in the table from real image from HSV format
          int Hindex = HSV[0] >>shiftH;
          int Sindex = HSV[1] >>shiftS;
          int Vindex = HSV[2] >>shiftV;
       
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
        
  cout<<"Corect :"<< cCorrect<<endl;    
  cout<<"Incorrect :"<< cIncorrect<<endl;
  float acc = (cCorrect*100)/(cCorrect+cIncorrect);
  
  cout<<"The accuracy is :"<<acc<<"%"<<endl;

  // remove trainning history of trainning
  
  
  return acc;

}
void leftOneOut(double ***aProbFloorHS, double ***aProbNonFloorHS,const int shiftH,const int shiftS,const int shiftV,int hLv,int sLv,int vLv)
{

  std::vector<std::string> aImages,aMasks;
  cout<<"In leave oneout Eval"<<endl;
  // test set image 21 to  
   for(int iNimg =0; iNimg< datasetSize; iNimg++)
   {
      int digit;
      std::string s;

      if(((iNimg+1)/10) <= 1e-5){
          s =  "00"; 
      }
      else{
          s = "0";
      }
      
      string s1 ="../video-frames/frame-"+ s + std::to_string(iNimg+1) + ".png";
      string s2 = "../video-frames/frame-"+ s + std::to_string(iNimg+1) + "-mask.png";
  
      aImages.push_back(s1);
      aMasks.push_back(s2);
 
   }
    
 
     int  nImages = aImages.size();
     cout<<"nImages in Test function :"<< nImages<<endl;
     // initial range number of images
     std::vector <int> allData(31);
     std::iota(std::begin(allData), std::end(allData), 0);
     
     cout<<"size of vector data "<<allData.size()<<endl;
     cout<<" Leave one out pass:"<<endl;
     float sumAvg = 0;

     for(int test_idx =0; test_idx < allData.size() ;test_idx++)
     {  
        allData.erase(std::find(allData.begin(),allData.end(),test_idx));
        
        float acc_ = validateModel(aProbFloorHS,aProbNonFloorHS,shiftH,shiftS,shiftV,aImages,aMasks,allData,test_idx,false);
         
        sumAvg += acc_; 
        allData.push_back(test_idx);

        //clear table every do every validation 
        
        for (int i = 0; i <hLv; i++) {
     
              for (int j = 0; j < sLv; j++) {
                 for (int k = 0; k < vLv; k++)
                 {
                      aProbFloorHS[i][j][k] = 0;
                      aProbNonFloorHS[i][j][k] = 0;
                 }
               }
   
        } 
   
     

        // break;
     }
        
     cout<<"Average all cross validation :"<<sumAvg/allData.size()<<endl;
     
     // Train again with all data that we have to use with stream data
     // we put 0 but we do not test 
      float acc_ = validateModel(aProbFloorHS,aProbNonFloorHS,shiftH,shiftS,shiftV,aImages,aMasks,allData,0,true);

     
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
    
    const int hLv = config["HSVLevel"]["H_level"].as<int>();
    const int sLv = config["HSVLevel"]["S_level"].as<int>();
    const int vLv = config["HSVLevel"]["V_level"].as<int>();
    const int shiftH = config["Shiftbin"]["shiftH"].as<int>();
    const int shiftS = config["Shiftbin"]["shiftS"].as<int>();
    const int shiftV = config["Shiftbin"]["shiftV"].as<int>();
    
    cout<<"hLv  : " <<hLv<<endl;
    cout<<"sLv  : " <<sLv<<endl;
    cout<<"vLv  : " <<vLv<<endl;
    cout<<"shiftH :" <<shiftH<<endl;
    cout<<"shiftS :" <<shiftS<<endl;
    cout<<"shiftV :" <<shiftV<<endl;
    


    //waitKey(0);
   //
   
    double ***ptrProbFloorHS = new double**[hLv];
    double ***ptrNonProbFloorHS = new double**[hLv];

    for (int i = 0; i <hLv; i++) {
 
        // Allocate memory blocks for
        // rows of each 2D array
         ptrProbFloorHS[i] = new double*[sLv];
         ptrNonProbFloorHS[i] = new double*[sLv];

        for (int j = 0; j < sLv; j++) {
 
            // Allocate memory blocks for
            // columns of each 2D array
            ptrProbFloorHS[i][j] = new double[vLv];
            ptrNonProbFloorHS[i][j] = new double[vLv];
         }
    }

    for (int i = 0; i <hLv; i++) {
 
         for (int j = 0; j < sLv; j++) {
            for (int k = 0; k < vLv; k++)
            {
                ptrProbFloorHS[i][j][k] = 0;
                ptrNonProbFloorHS[i][j][k] = 0;
            }
        }
    } 
   


   

 
   // compute the probability prob of floor and not for 
   // in order to draw and segment floor and other objects 
   // getHistrograms(ptrProbFloorHS,ptrNonProbFloorHS,shiftH,shiftS,shiftV);
    cout<<"Test process"<<endl;
    leftOneOut(ptrProbFloorHS,ptrNonProbFloorHS,shiftH,shiftS,shiftV, hLv,sLv, vLv);
     
    

//    Test(ptrProbFloorHS,ptrNonProbFloorHS,shiftH,shiftS,shiftV); 
    //leaveOneOutEval(ptrProbFloorHS,ptrNonProbFloorHS,shiftH,shiftS,shiftV);

 
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
         segmentFrame(matFrameCapture,ptrProbFloorHS,ptrNonProbFloorHS,shiftH,shiftS,shiftV);  
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
