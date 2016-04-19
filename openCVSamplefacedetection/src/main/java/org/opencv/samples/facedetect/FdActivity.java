package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    public static final int NATIVE_DETECTOR = 1;

    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemFace10;
    private MenuItem mItemFace05;
    private MenuItem mItemType;

    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private File mCascadeFile2;
    private CascadeClassifier mJavaDetector;
    private DetectionBasedTracker mNativeDetector;
    private DetectionBasedTracker mNativeDetector2;

    private int mDetectorType = NATIVE_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.05f;
    private int mAbsoluteFaceSize = 0;

    private CameraView mOpenCvCameraView;
    private List<Camera.Size> mResolutionList;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.outputwg);
                        InputStream is2 = getResources().openRawResource(R.raw.output_char);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "outputwg.xml");
                        mCascadeFile2 = new File(cascadeDir, "output_char.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);
                        FileOutputStream os2 = new FileOutputStream(mCascadeFile2);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        while ((bytesRead = is2.read(buffer)) != -1) {
                            os2.write(buffer, 0, bytesRead);
                        }
                        is2.close();
                        os2.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
                        mNativeDetector2 = new DetectionBasedTracker(mCascadeFile2.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                    mNativeDetector.start();
                    mNativeDetector2.start();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraView) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();

        Camera.Size resolution = null;
        int id = 0;
        mResolutionList = mOpenCvCameraView.getResolutionList();
        for(; id <= mResolutionList.size(); id++)
        {
            resolution = mResolutionList.get(id);
            if(resolution.width == 1280 && resolution.height == 720)
                break;
        }

        if(id > mResolutionList.size())
            resolution = mResolutionList.get(0);

        mOpenCvCameraView.setResolution(resolution);
        resolution = mOpenCvCameraView.getResolution();
        String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
        Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
        Log.i(TAG, "resolution" + Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString());
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Rect MaxRect = new Rect();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
            mNativeDetector2.setMinFaceSize((int)(mAbsoluteFaceSize/4));
        }

        MatOfRect faces = new MatOfRect();

        /*if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else*/ if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                MaxRect = AdaBoost(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        /*Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            if(i == 0)
                MaxRect = facesArray[i];
            else
            {
                if(facesArray[i].width > MaxRect.width)
                    MaxRect = facesArray[i];
            }

        }*/

        MaxRect.tl().x = MaxRect.tl().x - (MaxRect.width * 0.1);
        if (MaxRect.tl().x <= 0) {
            MaxRect.tl().x = 1;
        }

        MaxRect.br().x = MaxRect.br().x + (MaxRect.width * 0.1);
        if (MaxRect.br().x >= mRgba.width()) {
            MaxRect.br().x = mRgba.width() - 1;
        }

        MaxRect.tl().y = MaxRect.tl().y - (MaxRect.height * 0.05);//+ 15;
        if (MaxRect.tl().y <= 0) {
            MaxRect.tl().y = 1;
        }

        MaxRect.br().y = MaxRect.br().y + (MaxRect.height * 0.05);
        if (MaxRect.br().y >= mRgba.height()) {
            MaxRect.br().y = mRgba.height() - 1;
        }

        //Mat img_crop2 = new Mat(inputFrame.rgba(), MaxRect);
        //SaveImage(img_crop2);
        Imgproc.rectangle(mRgba, MaxRect.tl(), MaxRect.br(), FACE_RECT_COLOR, 3);

        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemFace10 = menu.add("Face size 10%");
        mItemFace05 = menu.add("Face size 05%");
        mItemType = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemFace10)
            setMinFaceSize(0.1f);
        else if (item == mItemFace05)
            setMinFaceSize(0.05f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
                mNativeDetector2.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
                mNativeDetector2.stop();
            }
        }
    }

    private void SaveImage(Mat mat) {
        Mat mIntermediateMat = new Mat();

        Imgproc.cvtColor(mat, mIntermediateMat, Imgproc.COLOR_RGB2BGR, 3);

        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String filename = currentDateandTime + ".jpg";
        File file = new File(path, filename);

        Boolean bool = null;
        filename = file.toString();
        bool = Imgcodecs.imwrite(filename, mIntermediateMat);

        /*if (bool == true)
            Toast.makeText(this, "Save Success", Toast.LENGTH_SHORT).show();
        else
            Toast.makeText(this, "Save fail", Toast.LENGTH_SHORT).show();*/
    }

    public Rect AdaBoost(Mat imageGray, MatOfRect faces) {
        Rect PlateRect;
        PlateRect = new Rect();
        Mat PlateImg = new Mat();
        Rect MaxRect;
        MaxRect = new Rect();
        MatOfRect chars = new MatOfRect();

        //int[] LPD = opencv.LPD(cascadeFile, pix, Width, Height, 3);
        if (mNativeDetector != null)
            mNativeDetector.detect(mGray, faces);

        Rect[] facesArray = faces.toArray();

        if (facesArray.length != 0) {
            //找出所有車牌定位出來的影像區塊並將此區塊合併成一個區塊
            Log.i("ArcSung", "have plate");
            int mini_x_left = 2000;
            int max_x_right = 0;
            int mini_y_up = 2000;
            int max_y_down = 0;
            for (int i = 0; i < facesArray.length; i++) {
                if (facesArray[i].tl().x < mini_x_left) {
                    mini_x_left = (int) facesArray[i].tl().x;
                }
                if (facesArray[i].br().x > max_x_right) {
                    max_x_right = (int) facesArray[i].br().x;
                }
                if (facesArray[i].tl().y < mini_y_up) {
                    mini_y_up = (int) facesArray[i].tl().y;
                }
                if (facesArray[i].br().y > max_y_down) {
                    max_y_down = (int) facesArray[i].br().y;
                }
            }
            //合併完擴大一點 0.2為擴大5分之1
            //mini_x_left = (int)(mini_x_left - (mini_pix_w * 0.3));//如果發生強制關閉有可能加減完已超過記憶體宣告區段
            //max_x_right = (int)(max_x_right + (mini_pix_w * 0.3)+20);
            mini_x_left = mini_x_left - 45; //35
            max_x_right = max_x_right + 45; //35

            //mini_y_up = (int)(mini_y_up - (mini_pix_h * 0.2));
            //max_y_down= (int)(max_y_down+ (mini_pix_h * 0.2));
            mini_y_up = mini_y_up - 45;  //35
            max_y_down = max_y_down + 45;  //35

            //int[] mini_pix = new int[mini_pix_w * mini_pix_h];

            if(mini_x_left < 10)
                mini_x_left = 10;
            if(max_x_right > mGray.width() - 10)
                max_x_right = mGray.width() - 10;
            if(mini_y_up < 10)
                mini_x_left = 10;
            if(max_y_down > mGray.height() - 10)
                max_y_down = mGray.height() - 10;

            int mini_pix_w = max_x_right - mini_x_left;
            int mini_pix_h = max_y_down - mini_y_up;

            PlateRect =new Rect(mini_x_left, mini_y_up, mini_pix_w, mini_pix_h);
            MaxRect =new Rect(mini_x_left, mini_y_up, mini_pix_w, mini_pix_h);

            PlateImg = new Mat(mGray, PlateRect);

            if (mNativeDetector2 != null)
                mNativeDetector2.detect(mGray, chars);

            Rect[] charsArray = chars.toArray();


            if (charsArray.length != 0) {
                Log.i("ArcSung", "have char");
                int[] LPD_fin = new int[4];
                for(int i = 0; i < charsArray.length; i++)
                {
                    charsArray[i].tl().x = charsArray[i].tl().x + mini_x_left;
                    charsArray[i].br().x = charsArray[i].br().x + mini_x_left;
                    charsArray[i].tl().y = charsArray[i].tl().y + mini_y_up;
                    charsArray[i].br().y = charsArray[i].br().y + mini_y_up;
                }

                for(int i = 0; i < facesArray.length; i++)
                {
                    facesArray[i].tl().x = facesArray[i].tl().x - 20;
                    facesArray[i].br().x = facesArray[i].br().x + 20;
                }

                int count1 = 0;
                int count2 = -4;
                int index = 0;

                for (int i = 0; i < facesArray.length; i+=4)
                {
                    for (int j = 0; j < charsArray.length; j+=4)
                    {
                        if (facesArray[i].tl().y - 30 < charsArray[j].tl().y && facesArray[i].br().y + 30 > charsArray[j].br().y)
                        {
                            count1 = count1 + 1;
                        }
                    }
                    if(count1 != 0 && count1 > index)    //count1 >= index
                    {
                        count2 = count2 + 4;

                        LPD_fin[0] = (int)facesArray[i].tl().x;
                        LPD_fin[1] = (int)facesArray[i].br().x;
                        LPD_fin[2] = (int)facesArray[i].tl().y;
                        LPD_fin[3] = (int)facesArray[i].br().y;

                        index = count1;
                    }
                    //index = count1;
                    count1 = 0;
                }

                int OT = 30;

                if (LPD_fin[1] - LPD_fin[0] > 235 && LPD_fin[1] - LPD_fin[0] < 300)
                {
                    OT = 40;
                }

                /*for (int i = 0; i < LPD_fin.length; i+=4)
                {
                    for (int j = 0; j < charsArray.length; j+=4)
                    {
                        if(LPD_fin[i + 2] + OT > charsArray[j].tl().y && LPD_fin[i + 2] - OT < charsArray[j].tl().y)
                        {

                            if(charsArray[j].tl().x < LPD_fin[i + 0])
                            {
                                if(LPD_fin[i + 0] - charsArray[j].tl().x < OT)
                                {
                                    LPD_fin[i + 0] = (int)charsArray[j].tl().x;
                                }
                            }

                            if(charsArray[j].br().x > LPD_fin[i + 1])
                            {
                                if(charsArray[j].br().x - LPD_fin[i + 1] < OT)
                                {
                                    LPD_fin[i + 1] = (int)charsArray[j].br().x;
                                }
                            }

                        }

                    }
                }*/

                MaxRect = new Rect(new Point(LPD_fin[0], LPD_fin[2]), new Point(LPD_fin[1], LPD_fin[3]));
                return MaxRect;

            }


        }

        return MaxRect;
    }
}
