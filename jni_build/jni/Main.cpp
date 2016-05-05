// Main.cpp
#include <jni.h>
#include <android/log.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "DetectPlates.h"
#include "PossiblePlate.h"
#include "DetectChars.h"

#include<iostream>

#define LOG_TAG "OpenCVJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


// global variables ///////////////////////////////////////////////////////////////////////////////
bool blnShowSteps = false;
bool blnKNNTrainingSuccessful = false;

// global constants ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);

///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
	cv::Point2f p2fRectPoints[4];

	licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);

	for (int i = 0; i < 4; i++) {
		cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
	cv::Point ptCenterOfTextArea;
	cv::Point ptLowerLeftTextOrigin;

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (double) licPlate.imgPlate.rows / 30.0;
	int intFontThickness = (int) std::round(dblFontScale * 1.5);
	int intBaseline = 0;

	cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale,
										intFontThickness, &intBaseline);

	ptCenterOfTextArea.x = (int) licPlate.rrLocationOfPlateInScene.center.x;

	if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {
		ptCenterOfTextArea.y = (int) std::round(licPlate.rrLocationOfPlateInScene.center.y) +
							   (int) std::round((double) licPlate.imgPlate.rows * 1.6);
	} else {
		ptCenterOfTextArea.y = (int) std::round(licPlate.rrLocationOfPlateInScene.center.y) -
							   (int) std::round((double) licPlate.imgPlate.rows * 1.6);
	}

	ptLowerLeftTextOrigin.x = (int) (ptCenterOfTextArea.x - (textSize.width / 2));
	ptLowerLeftTextOrigin.y = (int) (ptCenterOfTextArea.y + (textSize.height / 2));

	cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace,
				dblFontScale, SCALAR_YELLOW, intFontThickness);
}

std::string jstring2str(JNIEnv* env, jstring jstr)
{
	char*   rtn   =   NULL;
	jclass   clsstring   =   env->FindClass("java/lang/String");
	jstring   strencode   =   env->NewStringUTF("GB2312");
	jmethodID   mid   =   env->GetMethodID(clsstring,   "getBytes",   "(Ljava/lang/String;)[B");
	jbyteArray   barr=   (jbyteArray)env->CallObjectMethod(jstr,mid,strencode);
	jsize   alen   =   env->GetArrayLength(barr);
	jbyte*   ba   =   env->GetByteArrayElements(barr,JNI_FALSE);
	if(alen   >   0)
	{
		rtn   =   (char*)malloc(alen+1);
		memcpy(rtn,ba,alen);
		rtn[alen]=0;
	}
	env->ReleaseByteArrayElements(barr,ba,0);
	std::string stemp(rtn);
	free(rtn);
	return   stemp;
}

extern "C"
{
    JNIEXPORT void JNICALL
    Java_org_opencv_samples_facedetect_LPDActivity_OpenCVInit(JNIEnv *env , jobject, jstring classfile, jstring imagefile)
	{
        cv::String strclassfile = jstring2str(env, classfile);
        cv::String strimagefile = jstring2str(env, imagefile);
		blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN(strclassfile, strimagefile);

	}

	JNIEXPORT void JNICALL
	Java_org_opencv_samples_facedetect_LPDActivity_OpenCVLPD(JNIEnv * , jobject, jlong addrGray, jlong addrRgba)
	{
		cv::Mat& mGr  = *(cv::Mat*)addrGray;
		cv::Mat& mRgb = *(cv::Mat*)addrRgba;

		if (blnKNNTrainingSuccessful == false) {
		//std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
            LOGI("error: error: KNN traning was not successfu");
            return;
		}

		cv::Mat imgOriginalScene = mRgb.clone();		// input image

		std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);

		vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);

		if (vectorOfPossiblePlates.empty()) {
			//std::cout << std::endl << "no license plates were detected" << std::endl;
			LOGI("no license plates were detected");
		} else {
			// if we get in here vector of possible plates has at leat one plate

			// sort the vector of possible plates in DESCENDING order (most number of chars to least number of chars)
			std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);


			// suppose the plate with the most recognized chars (the first plate in sorted by string length descending order)
			// is the actual plate
			PossiblePlate licPlate = vectorOfPossiblePlates.front();


			if (licPlate.strChars.length()== 0) {
				//d::cout << std::endl << "no characters were detected" << std::endl <<
				LOGI("no characters were detected");
				return;
			}
			drawRedRectangleAroundPlate(mRgb, licPlate);

			//std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;
			//std::cout << std::endl << "-----------------------------------------" << std::endl;

			writeLicensePlateCharsOnImage(mRgb, licPlate);
		}

	}
}






