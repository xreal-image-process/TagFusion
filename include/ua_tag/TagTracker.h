#ifndef UA_TAG_TAG_TRACKER_H
#define UA_TAG_TAG_TRACKER_H

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Frame;
class KeyFrame;
class Map;

namespace tag {

class MapTagData;
class TagMapExporter;

class TagTracker
{
public:
    TagTracker();

    // 从 settings yaml 读取 Tag.verbose、Tag.export_dir 等
    explicit TagTracker(const std::string &settingsFile);
    ~TagTracker();

    TagTracker(const TagTracker &) = delete;
    TagTracker &operator=(const TagTracker &) = delete;

    // 休眠跟踪接口（尚未接入 Tracking 主循环；签名已切到 Map）
    bool Track(Frame &frame, Map &map);
    bool Relocalize(Frame &frame, Map &map);

    // 初始化成功后写入首帧位姿缓存（尚无速度）
    void SeedLastPose(const Sophus::SE3f &Tcw);

    // 清空位姿/速度/参考关键帧临时缓存（跟踪失败或重置时调用）
    void ClearMotionCache();

    // 地图 Reset 后截断 Tag 导出文件（轨迹/检测/首次注册）
    void ResetExports();

    // 若开启导出：记录当前帧相机 Pose 轨迹
    void LogCameraPose(const Frame &frame);

    // 若开启导出：记录本帧 AprilTag 检测（含观测面积）到 tag_detections.csv
    void LogTagDetections(const Frame &frame);

    // 若开启导出：MapTag 首次注册时追加 tag_first_registration.csv
    void LogMapTagFirstRegistration(KeyFrame *pKF, int tag_id, int left_idx,
                                    int right_idx, MapTagData *pTag);

    // 若开启导出：初始化 Commit 后，对初始化 KF 上全部 MapTag 记首次注册几何
    void LogKeyFrameMapTagFirstRegistrations(KeyFrame *pKF);

    // 若开启导出：写出终局 Tag+ORB 地图（CSV/PLY）并关闭轨迹文件
    void SaveExports(Map &map);

    // 若开启导出：在线覆盖写入 tag_map_corners.csv 与 tag_map_mirror_corners.csv（静默）
    void SaveTagMapOnline(Map &map);

    // 若开启导出：定标后立刻写出 MapTag + ORB 初始化点云/关键帧（尺度对比用）
    bool SaveInitMaps(Map &map,
                      const std::vector<Eigen::Vector3f> &orb_points,
                      const std::vector<std::pair<double, Sophus::SE3f>> &orb_kf_tcw);

    long unsigned int GetReferenceKeyFrameId() const noexcept { return mnReferenceKFId; }
    bool HasReferenceKeyFrame() const noexcept { return mbHasReferenceKF; }

    // 是否启用条件 3：可选参考 KF baseline 判定（休眠路径）
    void SetCheckRefKFBaseline(bool enable) noexcept { mbCheckRefKFBaseline = enable; }
    bool GetCheckRefKFBaseline() const noexcept { return mbCheckRefKFBaseline; }

    void SetMaxVisibleFramesPerMarker(int n) noexcept { mnMaxVisibleFramesPerMarker = n; }
    void SetMinBaseLine(float meters) noexcept { mMinBaseLine = meters; }

private:
    std::optional<Sophus::SE3f> PredictPoseWithMotionModel() const;
    void SelectReferenceKeyFrame(const Frame &frame, Map &map);
    bool NeedNewKeyFrame(const Frame &frame, const Map &map) const;
    void UpdateMotionCache(const Sophus::SE3f &Tcw_current);

    Sophus::SE3f mLastTcw;
    bool mbHasLastTcw{false};

    Sophus::SE3f mVelocity;
    bool mbHasVelocity{false};

    long unsigned int mnReferenceKFId{0};
    bool mbHasReferenceKF{false};

    int mnMaxVisibleFramesPerMarker{10};
    float mMinBaseLine{0.07f};
    bool mbCheckRefKFBaseline{false};

    bool mbVerbose{false};

    std::unique_ptr<TagMapExporter> mpExporter;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
