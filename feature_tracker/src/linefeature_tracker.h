
#pragma once

#include <iostream>
#include <queue>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"

#include "parameters.h"
#include "tic_toc.h"

//不使用opencv的lsd
// #include <opencv2/line_descriptor.hpp>
#include <opencv2/features2d.hpp>

#include "line_descriptor_custom.hpp"

using namespace cv::line_descriptor;
using namespace std;
using namespace cv;
using namespace camodocal;

struct Line
{
    Point2f start_xy;//endpoints xy in image frame, undistort
    Point2f end_xy;

    Point2f start_xy_visual, end_xy_visual;

    Point2f StartPt;
	Point2f EndPt;
	float lineWidth;
	Point2f Vp;

	Point2f Center;
	Point2f unitDir; // [cos(theta), sin(theta)]
	float length;
	float theta;

	// para_a * x + para_b * y + c = 0
	float para_a;
	float para_b;
	float para_c;

	float image_dx;
	float image_dy;
    float line_grad_avg;

	float xMin;
	float xMax;
	float yMin;
	float yMax;
	unsigned short id;
	int colorIdx;

    bool start_predict_fail = false;
    bool end_predict_fail = false;

    bool updated_forwframe = false;
    bool extended = false;

    int num_untracked = 0;

    enum valid_type {
        valid = 1,
        too_short = 2,
        bad_ZNCC = 3,
        line_out_of_view = 4,
        few_samples = 5,
        bad_gradient_direction = 6,
        bad_prediction = 7,
        large_untracked = 8
    };

    valid_type is_valid = valid;

    deque<Point2f> sample_points_undistort; //纠正后图像中的采样点
};

class FrameLines
{
public:
    int frame_id;
    Mat img;
    
    vector<Line> lines;
    vector< int > line_ID; //当前帧索引 --> 全局索引

    // opencv3 lsd+lbd
    std::vector<KeyLine> keylsd;
    Mat lbd_descr;
};
typedef shared_ptr< FrameLines > FrameLinesPtr;

class LineFeatureTracker
{
  public:
    LineFeatureTracker();

    void readIntrinsicParameter(const string &calib_file);
    void NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines, vector<pair<int, int> >& lineMatches);

    vector<Line> undistortedLineEndPoints();

    void readImage(const cv::Mat &_img);



    void getUndistortEndpointsXY(const vector<Line>& lines, vector<cv::Point2f>& endpoints);
    void getUndistortSample(const vector<Line>& lines, vector<cv::Point2f>& points, std::vector<int>& point2lineIdx);
    void initLinesPredict();
    void rejectWithF(vector<uchar>& status);
    void markFailEndpoints(const vector<uchar>& status);
    void markFailSamplePoints(const vector<uchar>& status, vector<int>& point2lineIdx);
    void checkEndpointsAndUpdateLinesInfo();
    void checkOpticalFlowEndpoints(vector<Line>& lines, const vector<Point2f>& endpoints);
    void checkAndUpdateEndpoints(vector<Line>& lines);

    void MakeALine( cv::Point2f start_pts, cv::Point2f end_pts, const int& rows, const int& cols, Line& line);
    void MakeALine(cv::Point2f start_pts, cv::Point2f end_pts, Line& line);
    void reduceLine(vector<Line> &lines, vector<int>& IDs);
    void fitLinesBySamples(std::vector<Line>& lines);
    void extractELSEDLine(const Mat &img, vector<Line>& lines, vector<Line>& lines_exist);

    bool checkMidPointDistance(const cv::Point2f& start_i, const cv::Point2f& end_i,
                               const cv::Point2f& start_j,const cv::Point2f& end_j, const float threshold = 5.0);
    float point2line(const cv::Point2f& p, const cv::Point2f& start_l, const cv::Point2f& end_l) {
        const Point2f sp(p - start_l);
        const Point2f se(end_l - start_l);
        return abs(sp.cross(se) / norm(se));
    }
    void updateTrackingLinesAndID();
    void checkGradient(std::vector<Line>& lines, const int& start_idx = 0);
    bool largeGradientAngle(const float& line_angle, const float& x, const float& y, const float& threshold = 20.0);

    void Line2KeyLine(const vector<Line>& lines_in, vector<KeyLine>& lines_out);
    void KeyLine2Line(const vector<KeyLine>&lines_in, vector<Line>&  lines_out);

    void correctNegativeXY(cv::Point2f& start_pts, cv::Point2f& end_pts);
    void correctOutsideXY(cv::Point2f& start_pts, cv::Point2f& end_pts, const int& rows, const int& cols);

    void visualize_line(const Mat &imageMat1, const FrameLinesPtr frame, const string &name, const bool show_NMS_area = false);
    void visualize_line(const Mat &imageMat1, const std::vector<Line>& lines, const string &name, const bool show_NMS_area = false);
    void visualize_line_matches(const Mat &img1, const std::vector<Line>& lines1,
                                const Mat &img2, const std::vector<Line>& lines2,
                                const string &name, const bool& show_samples = false);
    void visualize_lines_consecutive(const Mat &img1, const std::vector<Line>& lines1,
                                     const Mat &img2, const std::vector<Line>& lines2,
                                     const string &name, const bool& show_samples = false);

    void DrawRotatedRectangle(cv::Mat& image, const cv::Point2f& centerPoint, const cv::Size& rectangleSize,
                              const float& rotationDegrees, const int& val = 144);
    void fillRectangle(Mat& img, const Point2f* pts, int npts, const void* color);




    //当前帧， 新的一帧
    FrameLinesPtr curframe_, forwframe_;

    //畸变坐标映射矩阵map1和map2, 内参
    cv::Mat undist_map1_, undist_map2_ , K_;

    camodocal::CameraPtr m_camera;       // pinhole camera

    int frame_cnt;
//    vector<int> ids;                     // 每个特征点的id
    vector<int> track_cnt, tmp_track_cnt; // 记录某个特征已经跟踪多少帧了，即被多少帧看到了
    int allfeature_cnt;                  // 用来统计整个地图中有了多少条线，它将用来赋值

    double sum_time;
    double mean_time;


    cv::Mat dxImg, dyImg, gradient_dir; // degree
    int last_feature_count = -1;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;//线特征端点坐标
    vector<Line> lines_predict;
    vector< int > prev_lineID; //当前帧索引 --> 全局索引

    //NMS extend length in ELSED(vertical)
    int nms_extend = 15;
    //max lines in every frame
    int max_lines_num = 50;
};
