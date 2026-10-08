#ifndef UA_TAG_APRILTAG_VISUALIZER_H
#define UA_TAG_APRILTAG_VISUALIZER_H

#include "ua_tag/TagFrameData.h"

#include <opencv2/core/core.hpp>
#include <string>

namespace ORB_SLAM3
{
namespace ua_tag
{

// 在单张图上绘制指定相机观测（默认左目/单目）
cv::Mat DrawTags(const cv::Mat &image, const tag::TagFrameData &frame_data,
                 tag::CameraId camera_id = tag::CameraId::LEFT_OR_MONO);

// 在已有 BGR 图上叠加 Tag（imageScale 与 FrameDrawer 一致：角点坐标除以 scale）
void OverlayTags(cv::Mat &image, const tag::TagFrameData &frame_data,
                 tag::CameraId camera_id = tag::CameraId::LEFT_OR_MONO,
                 float imageScale = 1.f);

// 左右目分别绘制后水平拼接（同步并列可视化）
cv::Mat DrawTagsStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                       const tag::TagFrameData &frame_data);

// 单目/仅左目：绘制并保存
bool SaveTagsVis(const cv::Mat &image,
                 const tag::TagFrameData &frame_data,
                 const std::string &savePath);

// 双目：左右检测帧并排保存
bool SaveTagsVis(const cv::Mat &imLeft, const cv::Mat &imRight,
                 const tag::TagFrameData &frame_data,
                 const std::string &savePath);

// 确保输出目录存在；成功返回 true
bool EnsureDir(const std::string &dirPath);

}  // namespace ua_tag
}  // namespace ORB_SLAM3

#endif
