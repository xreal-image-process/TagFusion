#include "ua_tag/AprilTagVisualizer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

namespace ORB_SLAM3
{
namespace ua_tag
{

namespace
{

void DrawObservations(cv::Mat &vis, const tag::TagFrameData::Observations &tags,
                      float imageScale = 1.f)
{
    const cv::Scalar edgeColors[4] = {
        cv::Scalar(0, 255, 0),    // 0->1 green
        cv::Scalar(0, 255, 255),  // 1->2 yellow
        cv::Scalar(0, 128, 255),  // 2->3 orange
        cv::Scalar(255, 0, 0)     // 3->0 blue
    };
    const float invScale = (imageScale != 1.f && imageScale > 0.f) ? (1.f / imageScale) : 1.f;

    for(size_t i = 0; i < tags.size(); ++i)
    {
        const tag::TagObservation &tag = tags[i];
        cv::Point2f corners[4];
        for(int k = 0; k < 4; ++k)
            corners[k] = tag.corners_raw[k] * invScale;

        for(int k = 0; k < 4; ++k)
        {
            const cv::Point2f &p0 = corners[k];
            const cv::Point2f &p1 = corners[(k + 1) % 4];
            cv::line(vis, p0, p1, edgeColors[k], 2, cv::LINE_AA);
            cv::circle(vis, p0, 3, edgeColors[k], -1, cv::LINE_AA);
        }

        cv::Point2f center(0.f, 0.f);
        for(int k = 0; k < 4; ++k)
            center += corners[k];
        center *= 0.25f;

        // v: 位姿有效；a: 歧义已消解
        const bool has_pose = tag.pose_estimate.has_value();
        const tag::TagPoseCandidate *selected =
            has_pose ? tag.pose_estimate->Selected() : nullptr;
        const int valid = selected ? 1 : 0;
        const int amb_resolved =
            (has_pose && tag.pose_estimate->selected_candidate >= 0) ? 1 : 0;

        char line1[32];
        char line2[32];
        std::snprintf(line1, sizeof(line1), "id:%d", tag.tag_id);
        std::snprintf(line2, sizeof(line2), "v:%d a:%d", valid, amb_resolved);

        // 有效：蓝；无效：红（BGR）
        const cv::Scalar textColor = valid ? cv::Scalar(255, 64, 0) : cv::Scalar(0, 0, 255);
        cv::putText(vis, line1, center + cv::Point2f(4.f, -8.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, textColor, 2, cv::LINE_AA);
        cv::putText(vis, line2, center + cv::Point2f(4.f, 10.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, textColor, 2, cv::LINE_AA);
    }
}

cv::Mat ToBgrClone(const cv::Mat &image)
{
    cv::Mat vis;
    if(image.empty())
        return vis;

    if(image.channels() == 1)
        cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
    else if(image.channels() == 4)
        cv::cvtColor(image, vis, cv::COLOR_BGRA2BGR);
    else
        vis = image.clone();
    return vis;
}

void PutPanelLabel(cv::Mat &vis, const char *label)
{
    if(vis.empty() || !label)
        return;
    cv::putText(vis, label, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(vis, label, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
}

}  // namespace

bool EnsureDir(const std::string &dirPath)
{
    if(dirPath.empty())
        return false;

    struct stat st;
    if(stat(dirPath.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);

    return mkdir(dirPath.c_str(), 0755) == 0;
}

cv::Mat DrawTags(const cv::Mat &image, const tag::TagFrameData &frame_data,
                 tag::CameraId camera_id)
{
    cv::Mat vis = ToBgrClone(image);
    if(vis.empty())
        return vis;

    if(camera_id == tag::CameraId::RIGHT)
        DrawObservations(vis, frame_data.right);
    else
        DrawObservations(vis, frame_data.left);
    return vis;
}

void OverlayTags(cv::Mat &image, const tag::TagFrameData &frame_data,
                 tag::CameraId camera_id, float imageScale)
{
    if(image.empty())
        return;
    if(camera_id == tag::CameraId::RIGHT)
        DrawObservations(image, frame_data.right, imageScale);
    else
        DrawObservations(image, frame_data.left, imageScale);
}

cv::Mat DrawTagsStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                       const tag::TagFrameData &frame_data)
{
    cv::Mat left = DrawTags(imLeft, frame_data, tag::CameraId::LEFT_OR_MONO);
    cv::Mat right = DrawTags(imRight, frame_data, tag::CameraId::RIGHT);
    if(left.empty() || right.empty())
        return cv::Mat();

    // 对齐高度后水平拼接
    if(left.rows != right.rows)
    {
        const int h = std::max(left.rows, right.rows);
        if(left.rows != h)
        {
            const int w = std::max(1, left.cols * h / left.rows);
            cv::resize(left, left, cv::Size(w, h));
        }
        if(right.rows != h)
        {
            const int w = std::max(1, right.cols * h / right.rows);
            cv::resize(right, right, cv::Size(w, h));
        }
    }

    PutPanelLabel(left, "L");
    PutPanelLabel(right, "R");

    cv::Mat stereo;
    cv::hconcat(left, right, stereo);
    return stereo;
}

bool SaveTagsVis(const cv::Mat &image,
                 const tag::TagFrameData &frame_data,
                 const std::string &savePath)
{
    if(savePath.empty() || image.empty())
        return false;

    cv::Mat vis = DrawTags(image, frame_data, tag::CameraId::LEFT_OR_MONO);
    if(vis.empty())
        return false;

    return cv::imwrite(savePath, vis);
}

bool SaveTagsVis(const cv::Mat &imLeft, const cv::Mat &imRight,
                 const tag::TagFrameData &frame_data,
                 const std::string &savePath)
{
    if(savePath.empty() || imLeft.empty() || imRight.empty())
        return false;

    cv::Mat vis = DrawTagsStereo(imLeft, imRight, frame_data);
    if(vis.empty())
        return false;

    return cv::imwrite(savePath, vis);
}

}  // namespace ua_tag
}  // namespace ORB_SLAM3
