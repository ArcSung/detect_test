LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#OPENCV_CAMERA_MODULES:=off
#OPENCV_INSTALL_MODULES:=off
#OPENCV_LIB_TYPE:=SHARED
include /home/arcsung/code/OpenCV/OpenCV-3.0.0-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_SRC_FILES  := Main.cpp \
                    DetectChars.cpp \
                    DetectPlates.cpp \
                    PossibleChar.cpp \
                    PossiblePlate.cpp \
                    Preprocess.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_C_INCLUDES += /home/arcsung/code/OpenCV/OpenCV-3.0.0-android-sdk/sdk/native/jni/include
LOCAL_LDLIBS     += -llog -ldl

LOCAL_MODULE     := OpenCV_Plate_Recognition

include $(BUILD_SHARED_LIBRARY)
