#include "linefeature_tracker.h"
// #include "line_descriptor/src/precomp_custom.hpp"
#include "ELSED/src/ELSED.h"

unsigned int frame_count = 0;
const int LINE_MAX_UNTRACKED = 5;

bool inBorder(const cv::Point2f &pt, const int& row, const int& col)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

KeyLine MakeKeyLine( cv::Point2f start_pts, cv::Point2f end_pts, size_t cols ){
    KeyLine keyLine;
    //    keyLine.class_id = 0;
    //    keyLine.numOfPixels;

    // Set start point(and octave)
    if(start_pts.x > end_pts.x)
    {
        cv::Point2f tmp_pts;
        tmp_pts = start_pts;
        start_pts = end_pts;
        end_pts = tmp_pts;
    }

    keyLine.startPointX = (int)start_pts.x;
    keyLine.startPointY = (int)start_pts.y;
    keyLine.sPointInOctaveX = start_pts.x;
    keyLine.sPointInOctaveY = start_pts.y;

    // Set end point(and octave)
    keyLine.endPointX = (int)end_pts.x;
    keyLine.endPointY = (int)end_pts.y;
    keyLine.ePointInOctaveX = end_pts.x;
    keyLine.ePointInOctaveY = end_pts.y;

    // Set angle
    keyLine.angle = atan2((end_pts.y-start_pts.y),(end_pts.x-start_pts.x));

    // Set line length & response
    keyLine.lineLength = keyLine.numOfPixels = norm( Mat(end_pts), Mat(start_pts));
    keyLine.response = norm( Mat(end_pts), Mat(start_pts))/cols;

    // Set octave
    keyLine.octave = 0;

    // Set pt(mid point)
    keyLine.pt = (start_pts + end_pts)/2;

    // Set size
    keyLine.size = fabs((end_pts.x-start_pts.x) * (end_pts.y-start_pts.y));

    return keyLine;
}


LineFeatureTracker::LineFeatureTracker()
{
    allfeature_cnt = 0;
    frame_cnt = 0;
    sum_time = 0.0;
}

void LineFeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());

    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    K_ = m_camera->initUndistortRectifyMap(undist_map1_,undist_map2_);//获取畸变坐标映射矩阵mapx和mapy

}

//根据内参，从像素坐标反算相机系下归一化平面的坐标
vector<Line> LineFeatureTracker::undistortedLineEndPoints()
{
    vector<Line> un_lines;
    un_lines = curframe_->lines;//当前帧储存的直线
    float fx = K_.at<float>(0, 0);
    float fy = K_.at<float>(1, 1);
    float cx = K_.at<float>(0, 2);
    float cy = K_.at<float>(1, 2);
    for (unsigned int i = 0; i <curframe_->lines.size(); i++)
    {
        un_lines[i].StartPt.x = (curframe_->lines[i].StartPt.x - cx) / fx;
        un_lines[i].StartPt.y = (curframe_->lines[i].StartPt.y - cy) / fy;
        un_lines[i].EndPt.x = (curframe_->lines[i].EndPt.x - cx) / fx;
        un_lines[i].EndPt.y = (curframe_->lines[i].EndPt.y - cy) / fy;
    }
    return un_lines;
}

void LineFeatureTracker::visualize_line(const Mat &imageMat1, const FrameLinesPtr frame, const string &name, const bool show_NMS_area)
{
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < frame->lines.size(); ++k)
    {
        const Line& line1 =  frame->lines[k];
        const cv::Point2f& startPoint = line1.start_xy;
        const cv::Point2f& endPoint = line1.end_xy;

        if (show_NMS_area) {
            Point2f dir = endPoint - startPoint;
            float angle = atan2(dir.y, dir.x) / M_PI * 180.0;
            float len = norm((startPoint - endPoint));
//            ROS_DEBUG("image size: %u x %u", img1.cols, img1.rows);
//            ROS_DEBUG("start point: (%f, %f)", startPoint.x, startPoint.y);
//            ROS_DEBUG("end point: (%f, %f)", endPoint.x, endPoint.y);
            TicToc t_rect;
            DrawRotatedRectangle(img1, ((startPoint + endPoint) / 2.0), cv::Size(len + nms_extend, nms_extend), angle);
//            ROS_INFO("DrawRotatedRectangle costs: %fms", t_rect.toc());
        }
    }

    for (int k = 0; k <  frame->lines.size(); ++k) {

        const Line& line1 =  frame->lines[k];  // trainIdx
        const int& id = frame->line_ID[k];

        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);

        //line
        const cv::Point2f& startPoint = line1.start_xy;
        const cv::Point2f& endPoint = line1.end_xy;

//        if (line1.new_detect)
        if (id > last_feature_count) //new line in the forward frame
            cv::line(img1, startPoint, endPoint, cv::Scalar(0, 150, 255),2 ,8);
        else if (line1.updated_forwframe) //old lines updated in the forward frame (green)
            cv::line(img1, startPoint, endPoint, cv::Scalar(0, 255, 0),2 ,8);
        else if (line1.num_untracked >= LINE_MAX_UNTRACKED) // large num_untracked, to remove(red)
            cv::line(img1, startPoint, endPoint, cv::Scalar(0, 0, 255),2 ,8);
        else //old lines are not updated in the forward frame (blue)
            cv::line(img1, startPoint, endPoint, cv::Scalar(255, 0, 0),2 ,8);

        //start mid end point
        // b g r
        cv::circle(img1, startPoint, 1, cv::Scalar(255, 0, 0), 5);
//        cv::circle(img1, endPoint, 1, cv::Scalar(0, 0, 255), 5);
//        cv::circle(img1, 0.5 * startPoint + 0.5 * endPoint, 1, cv::Scalar(0, 255, 0), 5);
    }
    imshow(name, img1);
    waitKey(1);
}

void LineFeatureTracker::visualize_line(const Mat &imageMat1, const std::vector<Line>& lines, const string &name, const bool show_NMS_area)
{
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < lines.size(); ++k)
    {
        const Line& line1 =  lines[k];
        const cv::Point2f& startPoint = line1.start_xy;
        const cv::Point2f& endPoint = line1.end_xy;

        if (show_NMS_area) {
            Point2f dir = endPoint - startPoint;
            float angle = atan2(dir.y, dir.x) / M_PI * 180.0;
            float len = norm((startPoint - endPoint));
            TicToc t_rect;
            DrawRotatedRectangle(img1, ((startPoint + endPoint) / 2.0), cv::Size(len + nms_extend, nms_extend), angle);
//            DrawRotatedRectangle(img1, Point2f(200, 200), cv::Size(200, 30), 87);
//            ROS_INFO("DrawRotatedRectangle costs: %fms", t_rect.toc());
        }
    }

    for (int k = 0; k <  lines.size(); ++k) {

        const Line& line1 =  lines[k];  // trainIdx
        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);

        //line
        const cv::Point2f& startPoint = line1.start_xy;
        const cv::Point2f& endPoint = line1.end_xy;

        cv::line(img1, startPoint, endPoint, cv::Scalar(255, 0, 0),2 ,8);

        //start mid end point
        // b g r
        cv::circle(img1, startPoint, 1, cv::Scalar(255, 0, 0), 5);
    }
    imshow(name, img1);
    waitKey(1);
}

void LineFeatureTracker::NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines,
                                            vector<pair<int, int> > &lineMatches) {

    float th = 3.1415926/9;
    float dth = 30 * 30;
    for (size_t i = 0; i < forw_lines.size(); ++i) {
        Line lf = forw_lines.at(i);
        Line best_match;
        size_t best_j = 100000;
        size_t best_i = 100000;
        float grad_err_min_j = 100000;
        float grad_err_min_i = 100000;
        vector<Line> candidate;

        // 从 forw --> cur 查找
        for(size_t j = 0; j < cur_lines.size(); ++j) {
            Line lc = cur_lines.at(j);
            // condition 1
            Point2f d = lf.Center - lc.Center;
            float dist = d.dot(d);
            if( dist > dth) continue;  //
            // condition 2
            float delta_theta1 = fabs(lf.theta - lc.theta);
            float delta_theta2 = 3.1415926 - delta_theta1;
            if( delta_theta1 < th || delta_theta2 < th)
            {
                //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                candidate.push_back(lc);
                //float cost = fabs(lf.image_dx - lc.image_dx) + fabs( lf.image_dy - lc.image_dy) + 0.1 * dist;
                float cost = fabs(lf.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                //std::cout<< "line match cost: "<< cost <<" "<< cost - sqrt( dist )<<" "<< sqrt( dist ) <<"\n\n";
                if(cost < grad_err_min_j)
                {
                    best_match = lc;
                    grad_err_min_j = cost;
                    best_j = j;
                }
            }

        }
        if(grad_err_min_j > 50) continue;  // 没找到

        //std::cout<< "!!!!!!!!! minimal cost: "<<grad_err_min_j <<"\n\n";

        // 如果 forw --> cur 找到了 best, 那我们反过来再验证下
        if(best_j < cur_lines.size())
        {
            // 反过来，从 cur --> forw 查找
            Line lc = cur_lines.at(best_j);
            for (int k = 0; k < forw_lines.size(); ++k)
            {
                Line lk = forw_lines.at(k);

                // condition 1
                Point2f d = lk.Center - lc.Center;
                float dist = d.dot(d);
                if( dist > dth) continue;  //
                // condition 2
                float delta_theta1 = fabs(lk.theta - lc.theta);
                float delta_theta2 = 3.1415926 - delta_theta1;
                if( delta_theta1 < th || delta_theta2 < th)
                {
                    //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                    //candidate.push_back(lk);
                    //float cost = fabs(lk.image_dx - lc.image_dx) + fabs( lk.image_dy - lc.image_dy) + dist;
                    float cost = fabs(lk.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                    if(cost < grad_err_min_i)
                    {
                        grad_err_min_i = cost;
                        best_i = k;
                    }
                }

            }
        }

        if( grad_err_min_i < 50 && best_i == i){

            //std::cout<< "line match cost: "<<grad_err_min_j<<" "<<grad_err_min_i <<"\n\n";
            lineMatches.push_back(make_pair(best_j,i));
        }
        /*
        vector<Line> l;
        l.push_back(lf);
        vector<Line> best;
        best.push_back(best_match);
        visualizeLineTrackCandidate(l,forwframe_->img,"forwframe_");
        visualizeLineTrackCandidate(best,curframe_->img,"curframe_best");
        visualizeLineTrackCandidate(candidate,curframe_->img,"curframe_");
        cv::waitKey(0);
        */
    }

}

//#define NLT
#ifdef  NLT
void LineFeatureTracker::readImage(const cv::Mat &_img)
{
    cv::Mat img;
    TicToc t_p;
    frame_cnt++;
    cv::remap(_img, img, undist_map1_, undist_map2_, CV_INTER_LINEAR);
    //ROS_INFO("undistortImage costs: %fms", t_p.toc());
    if (EQUALIZE)   // 直方图均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }

    bool first_img = false;
    if (forwframe_ == nullptr) // 系统初始化的第一帧图像
    {
        forwframe_.reset(new FrameLines);
        curframe_.reset(new FrameLines);
        forwframe_->img = img;
        curframe_->img = img;
        first_img = true;
    }
    else
    {
        forwframe_.reset(new FrameLines);  // 初始化一个新的帧
        forwframe_->img = img;
    }

    // step 1: line extraction
    TicToc t_li;
    int lineMethod = 2;
    bool isROI = false;
    lineDetector ld(lineMethod, isROI, 0, (float)img.cols, 0, (float)img.rows);
    //ROS_INFO("ld inition costs: %fms", t_li.toc());
    TicToc t_ld;
    forwframe_->lines = ld.detect(img);

    for (size_t i = 0; i < forwframe_->lines.size(); ++i) {
        if(first_img)
            forwframe_->line_ID.push_back(allfeature_cnt++);
        else
            forwframe_->line_ID.push_back(-1);   // give a negative id
    }
    ROS_INFO("line detect costs: %fms", t_ld.toc());

    // step 3: junction & line matching
    if(curframe_->lines.size() > 0)
    {
        TicToc t_nlt;
        vector<pair<int, int> > linetracker;
        NearbyLineTracking(forwframe_->lines, curframe_->lines, linetracker);
        ROS_INFO("line match costs: %fms", t_nlt.toc());

        // 对新图像上的line赋予id值
        for(int j = 0; j < linetracker.size(); j++)
        {
            forwframe_->line_ID[linetracker[j].second] = curframe_->line_ID[linetracker[j].first];
        }

        // show NLT match
        //visualizeLineMatch(curframe_->lines, forwframe_->lines, linetracker,
                           curframe_->img, forwframe_->img, "NLT Line Matches", 10, true,
                           "frame");
        //visualizeLinewithID(forwframe_->lines,forwframe_->line_ID,forwframe_->img,"forwframe_");
        //visualizeLinewithID(curframe_->lines,curframe_->line_ID,curframe_->img,"curframe_");
        stringstream ss;
        ss <<"/home/hyj/datasets/line/" <<frame_cnt<<".jpg";
        // SaveFrameLinewithID(forwframe_->lines,forwframe_->line_ID,forwframe_->img,ss.str().c_str());
        waitKey(5);


        vector<Line> vecLine_tracked, vecLine_new;
        vector< int > lineID_tracked, lineID_new;
        // 将跟踪的线和没跟踪上的线进行区分
        for (size_t i = 0; i < forwframe_->lines.size(); ++i)
        {
            if( forwframe_->line_ID[i] == -1)
            {
                forwframe_->line_ID[i] = allfeature_cnt++;
                vecLine_new.push_back(forwframe_->lines[i]);
                lineID_new.push_back(forwframe_->line_ID[i]);
            }
            else
            {
                vecLine_tracked.push_back(forwframe_->lines[i]);
                lineID_tracked.push_back(forwframe_->line_ID[i]);
            }
        }
        int diff_n = 30 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
        if( diff_n > 0)    // 补充线条
        {
            for (int k = 0; k < vecLine_new.size(); ++k) {
                vecLine_tracked.push_back(vecLine_new[k]);
                lineID_tracked.push_back(lineID_new[k]);
            }
        }

        forwframe_->lines = vecLine_tracked;
        forwframe_->line_ID = lineID_tracked;

    }
    curframe_ = forwframe_;
}
#endif
int frame_num = 0;
#define MATCHES_DIST_THRESHOLD 30
void visualize_line_match(Mat imageMat1, Mat imageMat2,
                          std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                          std::vector<DMatch> good_matches)
{
    //	Mat img_1;
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    if (imageMat2.channels() != 3){
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    }
    else{
        img2 = imageMat2;
    }

    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( good_matches.size(), 1 );
    drawLineMatches( img1, octave0_1, img2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ),Scalar::all( -1 ), lsd_mask,DrawLinesMatchesFlags::DEFAULT );
    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < good_matches.size(); ++k) {
        DMatch mt = good_matches[k];

        KeyLine line1 = octave0_1[mt.queryIdx];  // trainIdx
        KeyLine line2 = octave0_2[mt.trainIdx];  //queryIdx


        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
        cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

        cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
        cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
        cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
        cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255),1, 8);
        cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255),1, 8);

    }
    /* plot matches */
    // cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);

    namedWindow("LSD matches", CV_WINDOW_NORMAL);
    imshow( "LSD matches", lsd_outImg );
    string name = to_string(frame_num);
    string path = "/home/dragon/ros_ws/p_ws/src/PL-VIO/feature_tracker/src/image/";
    name = path + name + ".jpg";
    frame_num ++;
    imwrite(name, lsd_outImg);
    // namedWindow("LSD matches1", CV_WINDOW_NORMAL);
    namedWindow("LSD matches2", CV_WINDOW_NORMAL);
    // imshow("LSD matches1", img1);
    imshow("LSD matches2", img2);
    waitKey(1);
}

void visualize_line_match(Mat imageMat1, Mat imageMat2,
                          std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                          std::vector<bool> good_matches)
{
    //	Mat img_1;
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    if (imageMat2.channels() != 3){
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    }
    else{
        img2 = imageMat2;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < good_matches.size(); ++k) {

        if(!good_matches[k]) continue;

        KeyLine line1 = octave0_1[k];  // trainIdx
        KeyLine line2 = octave0_2[k];  //queryIdx

        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
        cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

        cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
        cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
        cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
        cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255),1, 8);
        cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255),1, 8);

    }
    /* plot matches */
    /*
    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( lsd_matches.size(), 1 );
    drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
    DrawLinesMatchesFlags::DEFAULT );

    imshow( "LSD matches", lsd_outImg );
    */
   namedWindow("LSD matches1", CV_WINDOW_NORMAL);
   namedWindow("LSD matches2", CV_WINDOW_NORMAL);
    imshow("LSD matches1", img1);
    imshow("LSD matches2", img2);
    waitKey(1);
}
void visualize_line(Mat imageMat1,std::vector<KeyLine> octave0_1)
{
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < octave0_1.size(); ++k) {

        unsigned int r = 255; //lowest + int(rand() % range);
        unsigned int g = 255; //lowest + int(rand() % range);
        unsigned int b = 0;  //lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(octave0_1[k].startPointX), int(octave0_1[k].startPointY));
        cv::Point endPoint = cv::Point(int(octave0_1[k].endPointX), int(octave0_1[k].endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);
        // cv::circle(img1, startPoint, 2, cv::Scalar(255, 0, 0), 5);
        // cv::circle(img1, endPoint, 2, cv::Scalar(0, 255, 0), 5);


    }
    /* plot matches */
    /*
    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( lsd_matches.size(), 1 );
    drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
    DrawLinesMatchesFlags::DEFAULT );

    imshow( "LSD matches", lsd_outImg );
    */
    //namedWindow("LSD_C", CV_WINDOW_NORMAL);
    //imshow("LSD_C", img1);
    //waitKey(1);
}

cv::Mat last_unsuccess_image;
vector< KeyLine > last_unsuccess_keylsd;
vector< int >  last_unsuccess_id;
Mat last_unsuccess_lbd_descr;
void LineFeatureTracker::readImage(const cv::Mat &_img)
{
    cv::Mat img;//畸变纠正后影像
    TicToc t_p;
    frame_cnt++;

    //畸变纠正
    cv::remap(_img, img, undist_map1_, undist_map2_, CV_INTER_LINEAR);
//    cv::imshow("lineimg",img);
//    cv::waitKey(1);

    //ROS_INFO("undistortImage costs: %fms", t_p.toc());
    if (EQUALIZE)   // 直方图均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }

    bool first_img = false;
    if (forwframe_ == nullptr) // 系统初始化的第一帧图像
    {
        forwframe_.reset(new FrameLines);
        curframe_.reset(new FrameLines);
        forwframe_->img = img;
        curframe_->img = img;
        first_img = true;
    }
    else
    {
        forwframe_.reset(new FrameLines);  // 初始化一个新的帧
        forwframe_->img = img;//记录去畸变、均衡化后的image
    }
    TicToc t_li;


    TicToc t_gr;
    cv::Sobel(forwframe_->img, dxImg, CV_16SC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(forwframe_->img, dyImg, CV_16SC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
//    ROS_DEBUG("cv::Sobel cost: %f ms", t_gr.toc());

    //上一帧已探测到直线
    if(curframe_->lines.size()>0)//todo curr_line
    {
        last_feature_count = allfeature_cnt - 1; //上一帧最大特征id

        //get current frame endpoints
        int num_lines_curr = curframe_->lines.size();
        getUndistortEndpointsXY(curframe_->lines, cur_pts);
        ROS_DEBUG("cur_pts.size(): %d", cur_pts.size());
//        visualize_line_samples_undistort(curr_img, curr_line, "last frame");
//        visualize_line_samples_undistort(forw_img, curr_line, "curr_line forw frame");

        // add sample points to cur_pts
        std::vector<Point2f> cur_samples;
        std::vector<int> point2line_idx; //point  --->  line local index
        getUndistortSample(curframe_->lines, cur_samples, point2line_idx);
        cur_pts.insert(cur_pts.end(), cur_samples.begin(), cur_samples.end());

        TicToc t_o; // optical flow predict line edpoints
        vector<uchar> status;
        vector<float> err;
        //predict endpoints using LK optical flow
        if (!cur_pts.empty())
        {
            forw_pts.clear();
            cv::calcOpticalFlowPyrLK(curframe_->img, forwframe_->img, cur_pts, forw_pts, status, err,
                                     cv::Size(21, 21), 3);
        }
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());

        initLinesPredict(); // set lines_predict
        vector<uchar> status_f;
        rejectWithF(status_f);
        ROS_ASSERT(status.size() == status_f.size());
        for (int i = 0; i < (int)status.size(); ++i) {
            if (!status_f[i])
                status[i] = 0;
        }
        markFailEndpoints(status);
        ROS_DEBUG("markFailSamplePoints");
        markFailSamplePoints(status, point2line_idx);
        ROS_DEBUG("markFailSamplePoints, done");
        checkEndpointsAndUpdateLinesInfo();
//        visualize_line_samples_undistort(forw_img, lines_predict, "optical before reduce line");

        //Remove lines with insufficient sampling points
        reduceLine(lines_predict, prev_lineID);
//        visualize_line_samples_undistort(forw_img, lines_predict, "optical after reduce line");

        fitLinesBySamples(lines_predict);
//        visualize_line_samples_undistort(forw_img, lines_predict, "fit line curr --> forw");


        //    lineExtraction(img, lsd);
        ROS_DEBUG("ELSED: begin.");
        TicToc t_elsed;
        //enough num of lines
//    if (lines_predict.size() < max_lines_num)
//        visualize_line_samples_undistort(forw_img, lines_predict, "optical");

        extractELSEDLine(forwframe_->img, forwframe_->lines, lines_predict);
        ROS_DEBUG("line ELSED costs: %fms, new lines: %d.", t_elsed.toc(), forwframe_->lines.size());
//        visualize_line(forw_img, forw_line, "new lines forwframe");
//        visualize_line(forw_img, lines_predict, "lines predict", false);

        //reduce lines with larage num_untracked
        reduceLine(lines_predict, prev_lineID);

        ROS_DEBUG("updateTrackingLinesAndID");
        updateTrackingLinesAndID();
//        TicToc t_vis_line_new;
        visualize_line(forwframe_->img, forwframe_, "lines forward");
//        visual_time += t_vis_line_new.toc();
        ROS_DEBUG("updateTrackingLinesAndID, done");

        TicToc t_gr;
        ROS_DEBUG("check Sample Gradient");
        checkGradient(forwframe_->lines);
//        reduceLine(lines_predict, prev_lineID);
        ROS_DEBUG("check Sample Gradient: %f ms", t_gr.toc());
        visualize_line_matches(curframe_->img, curframe_->lines, forwframe_->img, forwframe_->lines, "line matches");

//        Line2KeyLine(forwframe_->lines, forwframe_->keylsd);
        curframe_ = forwframe_;
//        curr_descriptor = forw_descriptor.clone();
    }
    else
    {
        curframe_->img = forwframe_->img.clone();

        ROS_DEBUG("ELSED: begin.");
        TicToc t_elsed;
//        lineExtraction(curr_img, curr_keyLine, curr_descriptor);
//        KeyLine2Line(curr_keyLine, curr_line);
        lines_predict.clear();
        extractELSEDLine(curframe_->img, curframe_->lines, lines_predict);
        if (curframe_->lines.size() > max_lines_num)
            curframe_->lines.resize(max_lines_num);
//        Line2KeyLine(curframe_->lines, curframe_->keylsd);
        ROS_DEBUG("line ELSED costs: %fms, lines: %d.", t_elsed.toc(), curframe_->lines.size());

        TicToc t_gr;
        ROS_DEBUG("check Sample Gradient");
        checkGradient(curframe_->lines);
//        reduceLine(lines_predict, prev_lineID);
        ROS_DEBUG("check Sample Gradient: %f ms", t_gr.toc());
//        visualize_line_samples_undistort(curr_img, curr_line, "first frame");

        track_cnt.resize(curframe_->lines.size());
        curframe_->line_ID.resize(curframe_->lines.size());
        for(int i=0; i< curframe_->lines.size(); i++)
        {
            //initialize tmp_index
            track_cnt[i] = 1;
            curframe_->line_ID[i] = allfeature_cnt++;
        }
        ROS_DEBUG("first frame, done");
    }




//
//
//
//    Ptr<line_descriptor::LSDDetectorC> lsd_ = line_descriptor::LSDDetectorC::createLSDDetectorC();
//    // lsd parameters
//    line_descriptor::LSDDetectorC::LSDOptions opts;
//    opts.refine       = 1;     //1     	The way found lines will be refined
//    opts.scale        = 0.5;   //0.8   	The scale of the image that will be used to find the lines. Range (0..1].
//    opts.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
//    opts.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
//    opts.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
//    opts.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
//    opts.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
//    opts.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
//    double min_line_length = 0.125;  // Line segments shorter than that are rejected
//    // opts.refine       = 1;
//    // opts.scale        = 0.5;
//    // opts.sigma_scale  = 0.6;
//    // opts.quant        = 2.0;
//    // opts.ang_th       = 22.5;
//    // opts.log_eps      = 1.0;
//    // opts.density_th   = 0.6;
//    // opts.n_bins       = 1024;
//    // double min_line_length = 0.125;
//    opts.min_length   = min_line_length*(std::min(img.cols,img.rows));
//
//    std::vector<KeyLine> lsd, keylsd;
//	//void LSDDetectorC::detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, int scale, int numOctaves, const std::vector<Mat>& masks ) const
//    lsd_->detect( img, lsd, 2, 1, opts);
//    // visualize_line(img,lsd);
//    // step 1: line extraction
//    // TicToc t_li;
//    // std::vector<KeyLine> lsd, keylsd;
//    // Ptr<LSDDetector> lsd_;
//    // lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
//    // lsd_->detect( img, lsd, 2, 2 );
//
//    sum_time += t_li.toc();
//   ROS_INFO("line detect costs: %fms", t_li.toc());
//
//    Mat lbd_descr, keylbd_descr;//矩阵内每一行记录一条直线的lbd描述
//    // step 2: lbd descriptor
//    TicToc t_lbd;
//    Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor(  );
//
//
//    bd_->compute( img, lsd, lbd_descr );
//    // std::cout<<"lbd_descr = "<<lbd_descr.size()<<std::endl;
////////////////////////////
//    for ( int i = 0; i < (int) lsd.size(); i++ )
//    {
//        if( lsd[i].octave == 0 && lsd[i].lineLength >= 60)/** octave (pyramid layer), from which the keyline has been extracted */
//        {
//            keylsd.push_back( lsd[i] );
//            keylbd_descr.push_back( lbd_descr.row( i ) );
//        }
//    }
//    // std::cout<<"lbd_descr = "<<lbd_descr.size()<<std::endl;
////    ROS_INFO("lbd_descr detect costs: %fms", keylsd.size() * t_lbd.toc() / lsd.size() );
//    sum_time += keylsd.size() * t_lbd.toc() / lsd.size();
/////////////////
//
//    forwframe_->keylsd = keylsd;
//    forwframe_->lbd_descr = keylbd_descr;
//
//    for (size_t i = 0; i < forwframe_->keylsd.size(); ++i) {
//        if(first_img)
//            forwframe_->line_ID.push_back(allfeature_cnt++);//第一帧直接累计所有直线索引
//        else
//            forwframe_->line_ID.push_back(-1);   // give a negative id
//    }
//
//    // if(!first_img)
//    // {
//    //     std::vector<DMatch> lsd_matches;
//    //     Ptr<BinaryDescriptorMatcher> bdm_;
//    //     bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
//    //     bdm_->match(forwframe_->lbd_descr, curframe_->lbd_descr, lsd_matches);
//    //     visualize_line_match(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd, curframe_->keylsd, lsd_matches);
//    //     // std::cout<<"lsd_matches = "<<lsd_matches.size()<<" forwframe_->keylsd = "<<keylbd_descr.size()<<" curframe_->keylsd = "<<keylbd_descr.size()<<std::endl;
//    // }
//
//
//    if(curframe_->keylsd.size() > 0)
//    {
//        /* compute matches */
//        TicToc t_match;
//        std::vector<DMatch> lsd_matches;
//        Ptr<BinaryDescriptorMatcher> bdm_;
//        bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
//        bdm_->match(forwframe_->lbd_descr, curframe_->lbd_descr, lsd_matches);
////        ROS_INFO("lbd_macht costs: %fms", t_match.toc());
//        sum_time += t_match.toc();
//        mean_time = sum_time/frame_cnt;
//        // ROS_INFO("line feature tracker mean costs: %fms", mean_time);
//        /* select best matches */
//        std::vector<DMatch> good_matches;
//        std::vector<KeyLine> good_Keylines;
//        good_matches.clear();
//        ///根据起点终点距离再次筛选匹配直线，存在遮挡时可能有问题
//        for ( int i = 0; i < (int) lsd_matches.size(); i++ )
//        {
//            if( lsd_matches[i].distance < 30 ){
//
//                DMatch mt = lsd_matches[i];
//                KeyLine line1 =  forwframe_->keylsd[mt.queryIdx] ;
//                KeyLine line2 =  curframe_->keylsd[mt.trainIdx] ;
//                Point2f serr = line1.getStartPoint() - line2.getEndPoint();
//                Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
//                // std::cout<<"11111111111111111 = "<<abs(line1.angle-line2.angle)<<std::endl;
//                if((serr.dot(serr) < 200 * 200) && (eerr.dot(eerr) < 200 * 200)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远
//                    good_matches.push_back( lsd_matches[i] );
//            }
//        }
//
//
//        vector< int > success_id;
//        // std::cout << forwframe_->line_ID.size() <<" " <<curframe_->line_ID.size();
//        for (int k = 0; k < good_matches.size(); ++k) {
//            DMatch mt = good_matches[k];
//            forwframe_->line_ID[mt.queryIdx] = curframe_->line_ID[mt.trainIdx];
//            success_id.push_back(curframe_->line_ID[mt.trainIdx]);
//        }
//
//
//
//        //visualize_line_match(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd, curframe_->keylsd, good_matches);
//
//        //把没追踪到的线存起来
//
//        vector<KeyLine> vecLine_tracked, vecLine_new;
//        vector< int > lineID_tracked, lineID_new;
//        Mat DEscr_tracked, Descr_new;
//
//        // 将跟踪的线和没跟踪上的线进行区分
//        for (size_t i = 0; i < forwframe_->keylsd.size(); ++i)
//        {
//            if(forwframe_->line_ID[i] == -1)//未跟踪到的直线
//            {
//                forwframe_->line_ID[i] = allfeature_cnt++;
//                vecLine_new.push_back(forwframe_->keylsd[i]);
//                lineID_new.push_back(forwframe_->line_ID[i]);
//                Descr_new.push_back( forwframe_->lbd_descr.row( i ) );
//            }
//
//            else
//            {
//                vecLine_tracked.push_back(forwframe_->keylsd[i]);
//                lineID_tracked.push_back(forwframe_->line_ID[i]);
//                DEscr_tracked.push_back( forwframe_->lbd_descr.row( i ) );
//            }
//        }
//
//        vector<KeyLine> h_Line_new, v_Line_new;
//        vector< int > h_lineID_new,v_lineID_new;
//        Mat h_Descr_new,v_Descr_new;
//        for (size_t i = 0; i < vecLine_new.size(); ++i)
//        {
//            if((((vecLine_new[i].angle >= 3.14/4 && vecLine_new[i].angle <= 3*3.14/4))||(vecLine_new[i].angle <= -3.14/4 && vecLine_new[i].angle >= -3*3.14/4)))
//            {
//                h_Line_new.push_back(vecLine_new[i]);
//                h_lineID_new.push_back(lineID_new[i]);
//                h_Descr_new.push_back(Descr_new.row( i ));
//            }
//            else
//            {
//                v_Line_new.push_back(vecLine_new[i]);
//                v_lineID_new.push_back(lineID_new[i]);
//                v_Descr_new.push_back(Descr_new.row( i ));
//            }
//        }
//        int h_line,v_line;
//        h_line = v_line =0;
//        for (size_t i = 0; i < vecLine_tracked.size(); ++i)
//        {
//            if((((vecLine_tracked[i].angle >= 3.14/4 && vecLine_tracked[i].angle <= 3*3.14/4))||(vecLine_tracked[i].angle <= -3.14/4 && vecLine_tracked[i].angle >= -3*3.14/4)))
//            {
//                h_line ++;
//            }
//            else
//            {
//                v_line ++;
//            }
//        }
//        int diff_h = 35 - h_line;
//        int diff_v = 35 - v_line;
//
//        // std::cout<<"h_line = "<<h_line<<" v_line = "<<v_line<<std::endl;
//        if( diff_h > 0)    // 补充线条
//        {
//            int kkk = 1;
//            if(diff_h > h_Line_new.size())
//                diff_h = h_Line_new.size();
//            else
//                kkk = int(h_Line_new.size()/diff_h);
//            for (int k = 0; k < diff_h; ++k)
//            {
//                vecLine_tracked.push_back(h_Line_new[k]);
//                lineID_tracked.push_back(h_lineID_new[k]);
//                DEscr_tracked.push_back(h_Descr_new.row(k));
//            }
//            // std::cout  <<"h_kkk = " <<kkk<<" diff_h = "<<diff_h<<" h_Line_new.size() = "<<h_Line_new.size()<<std::endl;
//        }
//        if( diff_v > 0)    // 补充线条
//        {
//            int kkk = 1;
//            if(diff_v > v_Line_new.size())
//                diff_v = v_Line_new.size();
//            else
//                kkk = int(v_Line_new.size()/diff_v);
//            for (int k = 0; k < diff_v; ++k)
//            {
//                vecLine_tracked.push_back(v_Line_new[k]);
//                lineID_tracked.push_back(v_lineID_new[k]);
//                DEscr_tracked.push_back(v_Descr_new.row(k));
//            }            // std::cout  <<"v_kkk = " <<kkk<<" diff_v = "<<diff_v<<" v_Line_new.size() = "<<v_Line_new.size()<<std::endl;
//        }
//        // int diff_n = 50 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
//        // if( diff_n > 0)    // 补充线条
//        // {
//        //     for (int k = 0; k < vecLine_new.size(); ++k) {
//        //         vecLine_tracked.push_back(vecLine_new[k]);
//        //         lineID_tracked.push_back(lineID_new[k]);
//        //         DEscr_tracked.push_back(Descr_new.row(k));
//        //     }
//        // }
//
//        forwframe_->keylsd = vecLine_tracked;
//        forwframe_->line_ID = lineID_tracked;
//        forwframe_->lbd_descr = DEscr_tracked;
//
//    }
//
//    // 将opencv的KeyLine数据转为Line，重新记录了起点，终点，长度
//    for (int j = 0; j < forwframe_->keylsd.size(); ++j) {
//        Line l;
//        KeyLine lsd = forwframe_->keylsd[j];
//        l.StartPt = lsd.getStartPoint();
//        l.EndPt = lsd.getEndPoint();
//        l.length = lsd.lineLength;
//        forwframe_->lines.push_back(l);
//    }
//    curframe_ = forwframe_;


}

void LineFeatureTracker::getUndistortEndpointsXY(const vector<Line> &lines, vector<cv::Point2f> &endpoints) {
    endpoints.clear();
    int lines_size = lines.size();
    endpoints.resize(2 * lines_size);
    for (int i = 0; i < lines_size; ++i)
    {
        endpoints[2 * i] = lines[i].start_xy;
        endpoints[2 * i + 1] = lines[i].end_xy;
    }
}

void LineFeatureTracker::getUndistortSample(const vector<Line> &lines, vector<cv::Point2f> &points,
                                            vector<int> &point2lineIdx) {
    points.clear();
    point2lineIdx.clear();
    int lines_size = lines.size();
    for (int i = 0; i < lines_size; ++i)
        for (int j = 0; j < lines[i].sample_points_undistort.size(); ++j) {
            points.emplace_back(lines[i].sample_points_undistort[j]);
            point2lineIdx.emplace_back(i); //record local line index
        }
//    for (int i = 0; i < point2lineIdx.size(); ++i)
//        ROS_DEBUG("point %d  --->  line local id %d", i, point2lineIdx[i]);

}

void LineFeatureTracker::initLinesPredict() {
    lines_predict = curframe_->lines;
    for (Line& line : lines_predict)
    {
        line.start_predict_fail = false;
        line.end_predict_fail = false;
        line.updated_forwframe = false; //非新检测
        line.extended = false;
        line.sample_points_undistort.clear();
    }
    prev_lineID = curframe_->line_ID;
    tmp_track_cnt = track_cnt;
//    ROS_DEBUG("lines_predict: %d, prev_lineID: %d, tmp_track_cnt: %d", lines_predict.size(), prev_lineID.size(), tmp_track_cnt.size());
//
//    ROS_ASSERT(lines_predict.size() == prev_lineID.size());

//    ROS_WARN("lines_predict");
//    for (int i = 0; i < lines_predict.size(); ++i)
//    {
//        const Line& l = lines_predict[i];
//        ROS_DEBUG("line local: %d , global: %d", i, prev_lineID[i]);
//    }
}

void LineFeatureTracker::rejectWithF(vector <uchar> &status) {
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("line endpoints FM ransac begins");
        status.clear();
        TicToc t_f;
        cv::Mat matrix_f =  cv::findFundamentalMat(cur_pts, forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        ROS_DEBUG("line endpoints FM ransac costs: %fms", t_f.toc());
    }
}

void LineFeatureTracker::markFailEndpoints(const vector <uchar> &status) {
    for (int i = 0; i < int(forw_pts.size()) && i < curframe_->lines.size() * 2; i++)
        if (!status[i] || !inBorder(forw_pts[i], forwframe_->img.rows, forwframe_->img.cols))
        {
//                ROS_WARN("status[%d] (%f, %f), (%f, %f)", i, forw_pts_tmp[i].x, forw_pts_tmp[i].y, forw_pts[i].x,forw_pts[i].y);
            if (i % 2 == 0) {
                lines_predict[i / 2].start_predict_fail = true;
//                ROS_DEBUG("line %d start point (%f, %f) predict fail", i / 2, lines_predict[i/2].start_xy.x,lines_predict[i/2].start_xy.y);
            }
            else {
                lines_predict[i / 2].end_predict_fail = true;
//                ROS_DEBUG("line %d end point (%f, %f)predict fail", i / 2, lines_predict[i/2].start_xy.x,lines_predict[i/2].start_xy.y);
            }
            lines_predict[i / 2].is_valid = Line::bad_prediction;
//            lines_predict[i / 2].is_valid = Line::endpoint_out_of_image;
//            ROS_DEBUG("line %d predict fail", i / 2);
        }
}

void LineFeatureTracker::markFailSamplePoints(const vector<uchar> &status, vector<int> &point2lineIdx) {
    const int num_line_endpoints = lines_predict.size() * 2;
    for (int i = 0; i < (int)point2lineIdx.size(); i++) {

        int point_id = i + num_line_endpoints;
        if (!status[point_id] || !inBorder(forw_pts[point_id], forwframe_->img.rows, forwframe_->img.cols))
            continue;
        const int& local_id = point2lineIdx[i];
//        ROS_DEBUG("forw_pts %d (%f, %f) ---> line local id: %d", i, forw_pts[point_id].x, forw_pts[point_id].y, local_id);
        lines_predict[local_id].sample_points_undistort.emplace_back(forw_pts[point_id]);
    }
}

void LineFeatureTracker::checkEndpointsAndUpdateLinesInfo() {
    //更新有效光流端点、直线信息
    ROS_DEBUG("check Optical Flow Endpoints");
    checkOpticalFlowEndpoints(lines_predict, forw_pts);

    //处理顶点光流跟丟的直线
    TicToc t_update;
    ROS_DEBUG("check And Update Endpoints");
    checkAndUpdateEndpoints(lines_predict);
    ROS_DEBUG("check And Update Endpoints: %fms", t_update.toc());
}

void LineFeatureTracker::checkOpticalFlowEndpoints(vector<Line> &lines, const vector<Point2f> &endpoints) {
    ROS_ASSERT(lines.size() * 2 <= endpoints.size());

    for (int i = 0; i < lines.size(); ++i)
    {
        const Point2f& start = endpoints[2 * i];
        const Point2f& end = endpoints[2 * i + 1];
        //不一定准确，因为光流预测失败不能说明端点真的在图像外
        if ((start.x < 0 && end.x < 0) || (start.x >= curframe_->img.cols && end.x >= curframe_->img.cols) ||
            (start.y < 0 && end.y < 0) || (start.y >= curframe_->img.rows && end.y >= curframe_->img.rows))
        {
            lines[i].is_valid = Line::line_out_of_view;
//            ROS_DEBUG("line[%d} out_of_view", i);
            continue;
        }

        if (!lines[i].start_predict_fail && !lines[i].end_predict_fail)
        {
            //延伸顶点
//            Point2f start_new, end_new;
//            if (extendEndpoints(forwframe_->img, start, end, start_new, end_new))
//                lines[i].extended = true;
//            MakeALine(start_new, end_new, lines[i]); //update lines information
            MakeALine(start, end, lines[i]); //update lines information
//            ROS_DEBUG("line %d, start (%f, %f), end (%f, %f)", i, start.x, start.y, end.x, end.y);
            lines[i].updated_forwframe = false;
        }
    }
}

void LineFeatureTracker::checkAndUpdateEndpoints(vector<Line> &lines) {

    const float threshold = 0.85;
    int i = 0;
    for (Line& line : lines)
    {
        //todo 采样点和端点数量, 样本点太少尝试延伸线特征
        if (line.sample_points_undistort.size() < 3) {
            line.is_valid = Line::few_samples;
            continue;
        }

//        ROS_DEBUG("line[%d]", i);

        bool need_update = false;
        //check start point
//        if (line.start_predict_fail)
        if (line.start_predict_fail || !inBorder(line.start_xy, forwframe_->img.rows, forwframe_->img.cols))
        {
            if (!line.sample_points_undistort.empty()) {
                line.start_xy = line.sample_points_undistort.front();
                need_update = true;
            }

////            for (int i = 1; i < line.sample_zncc.size(); ++i)
//            if (!line.sample_points_undistort.empty())
//                line.popFront();
//            while (!line.sample_points_undistort.empty())
//            {
////                ROS_DEBUG("line.sample_zncc.front(): %f", line.sample_zncc.front());
//                if (line.sample_zncc.front() < threshold)
//                    line.popFront();
//                else
//                {
//                    //                line.StartUV = line.sample_points_undistort[i];
//                    line.StartUV = line.sample_points_undistort.front();
//                    line.start_predict_fail = false;
//                    break;
//                }
//            }
        }

        //check end point
//        if (line.end_predict_fail)
        if (line.end_predict_fail || !inBorder(line.end_xy, forwframe_->img.rows, forwframe_->img.cols))
        {
            if (!line.sample_points_undistort.empty()) {
                line.end_xy = line.sample_points_undistort.back();
                need_update = true;
            }

////            for (int i = line.sample_zncc.size() - 1; i >= 0; --i)
////            ROS_DEBUG("line.sample_zncc.back(): %f", line.sample_zncc.back());
//            if (!line.sample_points_undistort.empty())
//                line.popBack();
//
//            while (!line.sample_points_undistort.empty())
//            {
//                if (line.sample_zncc.back() < threshold)
//                    line.popBack();
//                else
//                {
//                    //line.EndUV = line.sample_points_undistort[i];
//                    line.EndUV = line.sample_points_undistort.back();
//                    line.end_predict_fail = false;
//                    break;
//                }
//            }
        }

        if (need_update)
        {
//            MakeALine(line.start_xy, line.end_xy, line);
            line.is_valid = Line::valid;
            line.updated_forwframe = false;
//            ROS_DEBUG("line[%d](new_detect = %d) StartUV(%f, %f)", i, line.new_detect, line.StartUV.x, line.StartUV.y);
//            ROS_DEBUG("line[%d](new_detect = %d) EndUV(%f, %f)", i,  line.new_detect, line.EndUV.x, line.EndUV.y);
        }
//        //check new line
        if (line.length < 30)
            line.is_valid = Line::too_short;
        ++i;
    }
}

void LineFeatureTracker::MakeALine(cv::Point2f start_pts, cv::Point2f end_pts, Line &line) {
//    // Set start point(and octave)
    if(start_pts.x > end_pts.x)
        swap(start_pts, end_pts);

    line.start_xy = start_pts;

//    keyLine.sPointInOctaveX = start_pts.x;
//    keyLine.sPointInOctaveY = start_pts.y;

    // Set end point(and octave)
    line.end_xy = end_pts;
//    keyLine.ePointInOctaveX = end_pts.x;
//    keyLine.ePointInOctaveY = end_pts.y;

    line.StartPt = line.start_xy;
    line.EndPt = line.end_xy;

    // Set angle
    line.theta = atan2((end_pts.y-start_pts.y),(end_pts.x-start_pts.x));

    // Set line length & response
//    keyLine.lineLength = keyLine.numOfPixels = norm( Mat(end_pts), Mat(start_pts));
    line.length = norm( Mat(end_pts), Mat(start_pts));

//    keyLine.response = norm( Mat(end_pts), Mat(start_pts))/cols;

    // Set octave
//    keyLine.octave = 0;

    // Set pt(mid point)
    line.Center = (start_pts + end_pts)/2;

    // Set size
    line.unitDir = (end_pts - start_pts) / line.length;
}

void LineFeatureTracker::MakeALine(cv::Point2f start_pts, cv::Point2f end_pts, const int &rows, const int &cols,
                                   Line &line) {
// Set start point(and octave)
    if(start_pts.x > end_pts.x)
        swap(start_pts, end_pts);

    // correct endpoints
    if (start_pts.x < 0 || start_pts.y < 0)
    {
//        ROS_DEBUG("correctNegativeXY: (%f. %f)", start_pts.x, start_pts.y);
        correctNegativeXY(start_pts, end_pts);
//        ROS_DEBUG("after correctNegativeXY: (%f. %f)", start_pts.x, start_pts.y);
    }

    if (end_pts.x >= cols || end_pts.y >= rows)
    {
//        ROS_DEBUG("correctOutsideXY end: (%f. %f)", end_pts.x, end_pts.y);
        correctOutsideXY(start_pts, end_pts, rows, cols);
//        ROS_DEBUG("after correctOutsideXY end: (%f. %f)", end_pts.x, end_pts.y);
    }

    if (start_pts.x >= cols || start_pts.y >= rows)
    {
//        ROS_DEBUG("correctOutsideXY start: (%f. %f)", start_pts.x, start_pts.y);
        correctOutsideXY(end_pts, start_pts, rows, cols);
//        ROS_DEBUG("after correctOutsideXY start: (%f. %f)", start_pts.x, start_pts.y);
    }

    line.start_xy = start_pts;
//    keyLine.sPointInOctaveX = start_pts.x;
//    keyLine.sPointInOctaveY = start_pts.y;

    // Set end point(and octave)
    line.end_xy = end_pts;
//    keyLine.ePointInOctaveX = end_pts.x;
//    keyLine.ePointInOctaveY = end_pts.y;

    line.StartPt = line.start_xy;
    line.EndPt = line.end_xy;

    // Set angle
    line.theta = atan2((end_pts.y-start_pts.y),(end_pts.x-start_pts.x));
//    ROS_DEBUG(" line.theta  = %f",  line.theta / M_PI * 180);

    // Set line length & response
//    keyLine.lineLength = keyLine.numOfPixels = norm( Mat(end_pts), Mat(start_pts));
    line.length = norm( Mat(end_pts), Mat(start_pts));

//    keyLine.response = norm( Mat(end_pts), Mat(start_pts))/cols;

    // Set octave
//    keyLine.octave = 0;

    // Set pt(mid point)
    line.Center = (start_pts + end_pts)/2;

    // Set size
    line.unitDir = (end_pts - start_pts) / line.length;

//    line.new_detect = true;
}

void LineFeatureTracker::reduceLine(vector<Line> &lines, vector<int> &IDs) {
    ROS_ASSERT(lines.size() == IDs.size());
    ROS_ASSERT(lines.size() == tmp_track_cnt.size());
//    ROS_WARN("reduceLine");

    int j = 0;
    for (int i = 0; i < int(lines.size()); i++)
    {
        if (lines[i].is_valid == Line::valid)
        {
            lines[j] = lines[i];
            tmp_track_cnt[j] = tmp_track_cnt[i];
            forw_pts[2 * j] = forw_pts[2 * i];
            forw_pts[2 * j + 1] = forw_pts[2 * i + 1];
            IDs[j++] = IDs[i];
        }
//        else
//            ROS_WARN("lose line local: %d, global: %d, valid type = %d", i, IDs[i], lines[i].is_valid);
    }

    forw_pts.resize(2 * j);
    tmp_track_cnt.resize(j);
    lines.resize(j);
    IDs.resize(j);
}

void LineFeatureTracker::fitLinesBySamples(vector<Line> &lines) {
    //todo 样本太少则不拟合，直接采用。。。。
    cv::Vec4f line4f;
    for (Line& line : lines) {
        std::vector<Point2f> points(line.sample_points_undistort.begin(), line.sample_points_undistort.end());
        cv::fitLine(points, line4f, CV_DIST_HUBER, 1.0, 0.1, 0.01);
        Point2f dir(line4f[0], line4f[1]);
        Point2f x0y0(line4f[2], line4f[3]);
        Point2f line_start = x0y0 + (line.start_xy - x0y0).dot(dir) * dir;
        Point2f line_end = x0y0 + (line.end_xy - x0y0).dot(dir) * dir;
        MakeALine(line_start, line_end, forwframe_->img.rows, forwframe_->img.cols, line);
    }
}

void LineFeatureTracker::extractELSEDLine(const Mat &img, vector<Line> &lines, vector<Line> &lines_exist) {
    upm::ELSED elsed;
    elsed.setNMSExtend(nms_extend);

    lines.clear();
    TicToc t_r;

    if (!lines_exist.empty())
    {
        int line_size = lines_exist.size();
        elsed.getLinesExist().resize(line_size);
        int i = 0;
        for (const Line& line : lines_exist)
        {
            upm::lineInfo& line_info = elsed.getLinesExist()[i];
            line_info.centerPoint = line.Center;
            line_info.angle =  line.theta / M_PI * 180.0;
            line_info.length = line.length;
            line_info.startPoint = line.start_xy;
            line_info.endPoint = line.end_xy;
            line_info.unitDir = line.unitDir;
//            line_info.need_detect = !line.extended;
            line_info.need_detect = true;
            std::vector<cv::Point2f> samples(line.sample_points_undistort.begin(), line.sample_points_undistort.end());
            swap(samples, line_info.sample_points);
            ++i;
        }
    }

    if (lines_exist.size() >= max_lines_num)
        elsed.enough_lines = true;

    //    upm::Segments segs = elsed.detect(img);
    ROS_DEBUG("process Image");
    elsed.processImage(img);
    ROS_DEBUG("process Image, done");

//    computeGradientDirection(elsed.getImgInfoPtr()->dxImg, elsed.getImgInfoPtr()->dyImg, gradient_dir);


//    segs = elsed.getELSEDSegments();
    int line_id = 0;
    const upm::Segments& segs = elsed.outside_nms_lines;
    lines.resize(segs.size());
//    ROS_DEBUG("new lines: %d", segs.size());
    for(unsigned int i = 0; i < segs.size(); i++)
    {
        Line& line = lines[i];
        MakeALine(cv::Point2f(segs[i][0], segs[i][1]), cv::Point2f(segs[i][2], segs[i][3]), img.rows, img.cols, line);
        line.start_xy_visual = line.start_xy;
        line.end_xy_visual = line.end_xy;
        line.updated_forwframe = true;
        line.id = line_id;
        line.num_untracked = 0;
        line_id++;
//        ROS_DEBUG("line[%d] length: %f", i, line.length);
    }

    std::vector<upm::Segments> nms_lines;
    nms_lines = elsed.getNMSSegments();
    std::vector<std::vector<Line>> lines_predict_extract;
    lines_predict_extract.resize(nms_lines.size());
//    line_update.clear();
    for(unsigned int i = 0; i < nms_lines.size(); i++)
    {
        lines_predict_extract[i].resize(nms_lines[i].size());
        for (int j = 0; j < nms_lines[i].size(); ++j) {
            Line& line = lines_predict_extract[i][j];
            MakeALine(cv::Point2f(nms_lines[i][j][0], nms_lines[i][j][1]), cv::Point2f(nms_lines[i][j][2], nms_lines[i][j][3]), img.rows, img.cols, line);
            line.updated_forwframe = true;
            line.id = line_id;
            line_id++;
//            line_update.emplace_back(line);
        }
//        ROS_DEBUG("nms[%d] lines: %d", i, nms_lines[i].size());
    }

    //check lines similarity
    ROS_DEBUG("update lines_exist");
    for (int i = 0; i < lines_exist.size(); ++i)
    {
//        ROS_DEBUG("i = %d", i);
        Line& line_old = lines_exist[i];
//        line_old.Center;
//        float angle_old = line_old.theta / M_PI * 180.0;
//        line_old.length;
//        line_old.StartUV;
//        line_old.EndUV;
        const float mid_distance_threshold = 8.0;

        int id_min = -1;
        float dist_min = 9999.999;
        bool merge_line = true;//todo
        vector<int> id_merge;
//        ROS_DEBUG("lines_predict_extract[%d].size() = %lu", i, lines_predict_extract[i].size());
        ROS_ASSERT(lines_predict_extract.size() == lines_exist.size());
        //Traverse all lines in the NMS region to update the line
        for (int j = 0; j < lines_predict_extract[i].size(); ++j)
        {
//            ROS_DEBUG("j = %d", j);
            const Line& line_new = lines_predict_extract[i][j];

            //same line
            if (norm((line_old.start_xy - line_new.start_xy)) < mid_distance_threshold &&
                norm((line_old.end_xy - line_new.end_xy)) < mid_distance_threshold)
            {
//                ROS_DEBUG("line %d, find same line", j);
                line_old = line_new;
                line_old.updated_forwframe = true;
                line_old.start_xy_visual = line_old.start_xy;
                line_old.end_xy_visual = line_old.end_xy;
                line_old.num_untracked = 0;
                id_merge.clear();
                break;
            }

            //similar length
            if (line_old.length >= line_new.length) {
                if (line_old.length * 0.5 > line_new.length)
                    continue;
            } else {
                if (line_new.length * 0.5 > line_old.length)
                    continue;
            }

            //angle
            if (acos(abs(line_old.unitDir.dot(line_new.unitDir))) > 8.0 / 180.0 * M_PI)
                continue;
            //distance

            //check midpoint distance
            if (!checkMidPointDistance(line_old.start_xy, line_old.end_xy, line_new.start_xy, line_new.end_xy, mid_distance_threshold))
                continue;

            float d1 = point2line(line_old.start_xy, line_new.start_xy, line_new.end_xy);
            float d2 = point2line(line_old.end_xy, line_new.start_xy, line_new.end_xy);
//            float d3 = point2line(line_old.Center, line_new.start_xy, line_new.end_xy);
//            float d_mean = (d1 + d2 + d3) / 3.0;
            float d_mean = (d1 + d2) / 2.0;
            if (merge_line) {
                if (d_mean < 5)
                    id_merge.emplace_back(j);
            } else {
                if (dist_min > d_mean)
                    dist_min = d_mean;
                id_min = j;
            }
        }
//        ROS_DEBUG("compute similarity, done");

        //no lines candidate
        if ((merge_line && id_merge.empty()) || (!merge_line && id_min == -1)) {
            line_old.num_untracked++;
            if (line_old.num_untracked >= LINE_MAX_UNTRACKED)
                line_old.is_valid = Line::large_untracked;
            line_old.updated_forwframe = false;
            continue;
        }

        if (!merge_line)
            line_old = lines_predict_extract[i][id_min];
        else
//        if (!id_merge.empty())
        {
            Line line = lines_predict_extract[i][id_merge[0]];

            for (int j = 1; j < id_merge.size(); ++j)
            {
                const Line& line_j = lines_predict_extract[i][id_merge[j]];

                cv::Point2f start_new, end_new;
                if (line.start_xy.x < line_j.start_xy.x)
                    start_new = line.start_xy;
                else
                    start_new = line_j.start_xy;


                if (line.end_xy.x > line_j.end_xy.x)
                    end_new = line.end_xy;
                else
                    end_new = line_j.end_xy;

                MakeALine(start_new, end_new, img.rows, img.cols, line);
            }
            line_old = line;
        }
        line_old.updated_forwframe = true;
        line_old.start_xy_visual = line_old.start_xy;
        line_old.end_xy_visual = line_old.end_xy;
        line_old.num_untracked = 0;
//        ROS_DEBUG("merge, done");
    }
    ROS_DEBUG("update lines_exist, done");
}

bool LineFeatureTracker::checkMidPointDistance(const Point2f &start_i, const Point2f &end_i, const Point2f &start_j,
                                               const Point2f &end_j, const float threshold) {
    const Point2f mid_i(start_i * 0.5 + end_i * 0.5);
    const Point2f mid_j(start_j * 0.5 + end_j * 0.5);
    float d1 = point2line(mid_i, start_j, end_j);
    float d2 = point2line(mid_j, start_i, end_i);
//    ROS_DEBUG("d1 = %f, d2 = %f", d1, d2);
    if(d1 <= threshold && d2 <= threshold)
        return true;
    return false;
}

void LineFeatureTracker::updateTrackingLinesAndID() {
    ROS_ASSERT(lines_predict.size() == prev_lineID.size());
    ROS_ASSERT(lines_predict.size() == tmp_track_cnt.size());

    for (int i = 0; i < lines_predict.size(); ++i)
        tmp_track_cnt[i]++;

    int num_lines_predict = lines_predict.size();
    int new_lines_needed = max_lines_num - num_lines_predict;
    int num_lines_new = forwframe_->lines.size();
    if (new_lines_needed > 0)
    {
        // set new lines ID
        for (size_t i = 0; i < new_lines_needed && i < num_lines_new; ++i)
        {
            lines_predict.emplace_back(forwframe_->lines[i]);
            prev_lineID.emplace_back(allfeature_cnt++);
            tmp_track_cnt.emplace_back(1);
        }
    }
//    swap(lines_predict, forwframe_->mergedLine);
//    swap(prev_lineID, forwframe_->lineID);

    forwframe_->lines = lines_predict;
    forwframe_->line_ID = prev_lineID;
    track_cnt = tmp_track_cnt;
//    for (const int& id : forwframe_->lineID)
//        ROS_INFO("line id: %d", id);
}

void LineFeatureTracker::checkGradient(vector<Line> &lines, const int &start_idx) {
    for (unsigned int i = start_idx; i <lines.size(); i++)
    {
        Line& line = lines[i];

        line.sample_points_undistort.clear();

        float line_angle = line.theta / M_PI * 180.0;
        if (line_angle < 0)
            line_angle += 180.0;
        float angle_threshold = 30.0; //angle between line and sample points gradient direction

//        //鱼眼重投影 无畸变 像素坐标
        Eigen::Vector2f undistort_start, undistort_end;
        undistort_start(0) = line.start_xy.x;
        undistort_start(1) = line.start_xy.y;
        undistort_end(0) = line.end_xy.x;
        undistort_end(1) = line.end_xy.y;
        const float &length = line.length;

//        //sample points from start to end
        deque<Point2f>& sample_pts = line.sample_points_undistort;

        //todo 按像素距离、间隔较小值采样 调参
        Eigen::Vector2f start_end = undistort_end - undistort_start;
        double times = max(10.0,  start_end.norm() / 10.0);
        Eigen::Vector2f interval = (undistort_end - undistort_start) / times;
        Eigen::Vector2f undistort_p = undistort_start + interval;
        int pts_total = 0, pts_good = 0;
        for (int step = 0; step < times - 1; ++step, undistort_p += interval)
        {
            ++pts_total;
//            if (step == times) //end point
//            {
//                undistort_p(0) =  undistort_end(0);
//                undistort_p(1) =  undistort_end(1);
//            }

            //非端点加入梯度方向条件，不满足不采样
//            if (step != 0 || step != times) // not start point
            {
                if (largeGradientAngle(line_angle, undistort_p(0), undistort_p(1), angle_threshold))
                    continue;
            }

            ++pts_good;
            sample_pts.push_back(Point2f(undistort_p(0), undistort_p(1)));
        }

        //todo
        if ((pts_good * 1.0) / (pts_total * 1.0) <= 0.5)
        {
            line.is_valid = Line::bad_gradient_direction;
        }
    }
}

bool LineFeatureTracker::largeGradientAngle(const float &line_angle, const float &x, const float &y,
                                            const float &threshold) {
    ROS_ASSERT(dxImg.ptr<int16_t>());
    ROS_ASSERT(dyImg.ptr<int16_t>());

    const int16_t& dx = dxImg.at<int16_t>(y, x);
    const int16_t& dy = dyImg.at<int16_t>(y, x);
    if (sqrt(dx * dx + dy *dy) < 30) //gradient too small
        return true;
    float gradient_angle = atan2(dy, dx) / M_PI * 180.0; //degree
//                float gradient_angle = gradient_dir.at<float>(undistort_p(1), undistort_p(0));//degree
    if (gradient_angle < 0.0)
        gradient_angle += 180.0;
    else if (gradient_angle > 180.0)
        gradient_angle -= 180.0;
    float delta_angle = abs(gradient_angle - line_angle);
    if (delta_angle < 90.0 - threshold ||  delta_angle > 90 + threshold)
        return true;
    return false;
}

void LineFeatureTracker::Line2KeyLine(const vector<Line> &lines_in, vector<KeyLine> &lines_out) {
    int num_lines = lines_in.size();
    lines_out.resize(num_lines);
    int line_id = 0;
    for (int i = 0; i < num_lines; ++i) {
        const Line& line = lines_in[i];
        lines_out[i] = MakeKeyLine(line.start_xy, line.end_xy, forwframe_->img.cols);
//        ROS_DEBUG("lines_out[%d]: start(%f, %f) end(%f, %f)", i, lines_out[i].startPointX,lines_out[i].startPointY
//                  , lines_out[i].endPointX, lines_out[i].endPointY);
//        lines_out[i].class_id = line_id;
//        ++line_id;
    }
}

void LineFeatureTracker::correctNegativeXY(Point2f &start_pts, Point2f &end_pts) {
    cv::Point2f dir = end_pts - start_pts;
    float length = norm(dir);
    dir = dir / length;
//    ROS_DEBUG("dir: (%f. %f)", dir.x, dir.y);
    if (start_pts.x < 0)
    {
        float delta_x = 0 - start_pts.x;
        start_pts.x = 0;
        if (dir.x != 0)
            start_pts.y = start_pts.y + delta_x / dir.x * dir.y;
    }
    if (start_pts.y < 0)
    {
        float delta_y = 0 - start_pts.y;
        start_pts.y = 0;
        if (dir.y != 0)
            start_pts.x = start_pts.x + delta_y / dir.y * dir.x;
    }
}

void LineFeatureTracker::correctOutsideXY(Point2f &start_pts, Point2f &end_pts, const int &rows, const int &cols) {
    cv::Point2f dir = end_pts - start_pts;
    float length = norm(dir);
    dir = dir / length;
//    ROS_DEBUG("dir after: (%f, %f)", dir.x, dir.y);

    if (end_pts.x >= cols)
    {
        float delta_x = cols - 1 - end_pts.x;
        end_pts.x = cols - 1;
        if (dir.x != 0)
            end_pts.y = end_pts.y + delta_x / dir.x * dir.y;
//        ROS_WARN("end_pts X: (%f. %f)", end_pts.x, end_pts.y);

    }
//    ROS_WARN("end_pts X: (%f. %f)", end_pts.x, end_pts.y);

    if (end_pts.y >= rows)
    {
        float delta_y = rows - 1 - end_pts.y;
        end_pts.y = rows - 1;
        if (dir.y != 0)
            end_pts.x = end_pts.x + delta_y / dir.y * dir.x;
//        ROS_WARN("end_pts Y: (%f. %f)", end_pts.x, end_pts.y);
    }
}

void LineFeatureTracker::DrawRotatedRectangle(Mat &image, const Point2f &centerPoint, const Size &rectangleSize,
                                              const float &rotationDegrees, const int &val) {
    //    cv::Scalar color = cv::Scalar(255.0, 255.0, 255.0); // white

    // Create the rotated rectangle
    cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

    // We take the edges that OpenCV calculated for us
    cv::Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);

//    vertices2f[0] = Point2f(0, 300);
//    vertices2f[1] = Point2f(300, 300);
//    vertices2f[2] = Point2f(300, 200);
//    vertices2f[3] = Point2f(0, 200);

//    cv::circle(image, vertices2f[0], 1, cv::Scalar(0, 0, 255), 5);
//    cv::circle(image, vertices2f[1], 1, cv::Scalar(0, 0, 255), 5);
//    cv::circle(image, vertices2f[2], 1, cv::Scalar(0, 0, 255), 5);
//    cv::circle(image, vertices2f[3], 1, cv::Scalar(0, 0, 255), 5);

    {
        // Convert them so we can use them in a fillConvexPoly
        cv::Point vertices[4];
        for (int i = 0; i < 4; ++i)
            vertices[i] = vertices2f[i];

        TicToc t_fill;
        // Now we can fill the rotated rectangle with our specified color
        cv::fillConvexPoly(image, vertices, 4, val);
        ROS_DEBUG("cv::fillConvexPoly cost: %fms", t_fill.toc());
    }

//    {
//        TicToc t_fill;
//        uchar *buf = new uchar[4]{255, 255, 255, 0};
//        fillRectangle(image, vertices2f, 4, buf);
////        ROS_DEBUG("fill Rectangle hand writing cost: %fms", t_fill.toc());
//    }
}

void LineFeatureTracker::visualize_line_matches(const Mat &img1, const std::vector<Line>& lines1,
                                                 const Mat &img2, const std::vector<Line>& lines2,
                                                 const string &name)
{
    Size size( img1.cols + img2.cols, MAX(img1.rows, img2.rows) );
    Mat outImg;
    outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
    outImg = Scalar::all(0);
    Mat outImg1, outImg2;
    outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
    outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );

    if( img1.type() == CV_8U )
        cvtColor( img1, outImg1, CV_GRAY2BGR );
    else
        img1.copyTo( outImg1 );

    if( img2.type() == CV_8U )
        cvtColor( img2, outImg2, CV_GRAY2BGR );
    else
        img2.copyTo( outImg2 );

    //get matched lines in curframe and forwframe, global id --> {curframe local, forwframe local}
    unordered_map<int, std::vector<int>> lines_matches;
    for (int i = 0; i < curframe_->lines.size(); ++i) {
        const int& global_id = curframe_->line_ID[i];
        lines_matches[global_id].emplace_back(i);
    }
    std::vector<int> new_lines_local;
    for (int i = 0; i < forwframe_->lines.size(); ++i) {
        const int& global_id = forwframe_->line_ID[i];
        if (lines_matches.find(global_id) != lines_matches.end()) //仅添加能匹配的线对
            lines_matches[global_id].emplace_back(i);
        else
            new_lines_local.emplace_back(i);
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;

    Mat _outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
    Mat _outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
    for (const auto& m : lines_matches) {
        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);

        if (m.second.size() == 1) //no match candidate
        {
            const int& line_local_id = m.second[0];
            const Line& l = curframe_->lines[line_local_id];
            cv::line(_outImg1, l.start_xy, l.end_xy, cv::Scalar(r, g, b),2 ,8);
        }
        else if (m.second.size() == 2)
        {
            //line in curframe
            const int& line_local_id1 = m.second[0];
            const Line& l1 = curframe_->lines[line_local_id1];
            cv::line(_outImg1, l1.start_xy, l1.end_xy, cv::Scalar(r, g, b),2 ,8);

            //line in forwframe
            const int& line_local_id2 = m.second[1];
            const Line& l2 = forwframe_->lines[line_local_id2];
            cv::line(_outImg2, l2.start_xy, l2.end_xy, cv::Scalar(r, g, b),2 ,8);
        }
        else
            ROS_WARN("one global id for 3 more lines");
    }

    for (int i = 0; i < new_lines_local.size(); ++i) {
        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);
        //line in forwframe
        const int& line_local_id2 = new_lines_local[i];
        const Line& l2 = forwframe_->lines[line_local_id2];
        cv::line(_outImg2, l2.start_xy, l2.end_xy, cv::Scalar(r, g, b),2 ,8);
    }

    imshow(name, outImg);
    waitKey(1);
}
