#ifndef UA_TAG_TAG_VIEWER_H
#define UA_TAG_TAG_VIEWER_H

#include "ua_tag/TagFrameData.h"

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Map;

namespace tag {

// Tag 专用可视化：左 AprilTag 检测图，右 Tag 地图 + 相机轨迹（独立 Pangolin 窗口）
class TagViewer
{
public:
    explicit TagViewer(const std::string &settingsFile = std::string());
    ~TagViewer();

    TagViewer(const TagViewer &) = delete;
    TagViewer &operator=(const TagViewer &) = delete;

    void Start();
    void RequestFinish();
    bool isFinished() const;

    // 冻结后忽略后续 Update（用于只展示初始化两帧 pose + MapTag）
    void Freeze();
    bool IsFrozen() const;

    // 内部拷贝图像/位姿/MapTag 快照，不持有外部指针
    // tracking_state: Tracking::eTrackingState
    void Update(const cv::Mat &image,
                const TagFrameData &frame_data,
                bool has_pose,
                const Sophus::SE3f &Tcw,
                unsigned long frame_id,
                Map *pMap,
                int tracking_state);

private:
    struct TagVis
    {
        int id = -1;
        bool fixed = false;
        bool has_pose = false;
        bool has_corners = false;
        Sophus::SE3f T_wt;
        std::array<Eigen::Vector3f, 4> corners{};
    };

    void Run();
    bool CheckFinish();
    void SetFinish();

    void DrawCameraFrustum(const Sophus::SE3f &Twc, float r, float g, float b,
                           float size) const;
    void DrawTagSquare(const TagVis &tag) const;
    void DrawTagId(const TagVis &tag) const;
    void DrawAxes(const Sophus::SE3f &Tw, float len) const;

    mutable std::mutex mutex_data_;
    cv::Mat mImage;
    bool mbHasPose = false;
    Sophus::SE3f mTcw;
    unsigned long mnFrameId = 0;
    int mTrackingState = 1;  // Tracking::NOT_INITIALIZED
    std::vector<TagVis> mTags;
    std::vector<Sophus::SE3f> mKfTwc;

    std::atomic<bool> mbFinishRequested{false};
    std::atomic<bool> mbFinished{true};
    std::atomic<bool> mbFrozen{false};
    std::thread mThread;

    float mCameraSize = 0.08f;
    float mViewpointX = 0.0f;
    float mViewpointY = -0.7f;
    float mViewpointZ = -1.8f;
    float mViewpointF = 500.0f;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
