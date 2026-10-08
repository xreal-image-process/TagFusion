#ifndef UA_TAG_TAG_MAP_EXPORTER_H
#define UA_TAG_TAG_MAP_EXPORTER_H

#include "ua_tag/TagPoseConstraints.h"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class KeyFrame;
class Map;

namespace tag {

class MapTagData;
class TagFrameData;

// 导出 Map 中 MapTag 世界系四角点与关联关键帧相机 Pose
class TagMapExporter
{
public:
    // output_dir 下写入 tag_map_corners.csv / tag_map_mirror_corners.csv
    //                    / tag_camera_trajectory.txt(TUM)
    //                    / tag_detections.csv（每帧检测）
    //                    / tag_first_registration.csv（每个 MapTag 首次注册时几何）
    explicit TagMapExporter(const std::string &output_dir,
                            TagFactorWeightParams factor_weight = {});
    ~TagMapExporter();

    TagMapExporter(const TagMapExporter &) = delete;
    TagMapExporter &operator=(const TagMapExporter &) = delete;

    bool IsEnabled() const noexcept { return mbEnabled; }
    const std::string &OutputDir() const noexcept { return mOutputDir; }

    // 追加一帧相机轨迹（输入 Tcw；落盘 TUM：timestamp tx ty tz qx qy qz qw，位姿为 Twc）
    void AppendCameraPose(unsigned long frame_id,
                          double timestamp,
                          const Sophus::SE3f &Tcw);

    // 追加本帧 AprilTag 检测日志（左右目各行）：面积/周长/hamming/中心
    void AppendTagDetections(unsigned long frame_id,
                             double timestamp,
                             const TagFrameData &frame_data);

    // 某个 MapTag 首次注册进地图时：记录该 KF 上观测面积、距离、光轴–平面/法向夹角
    // left_idx / right_idx 为 KF.mTagFrameData 下标，无观测则为 -1
    void AppendMapTagFirstRegistration(KeyFrame *pKF,
                                       int tag_id,
                                       int left_idx,
                                       int right_idx,
                                       MapTagData *pTag);

    // 初始化 Commit 成功后：对初始化 KF 上已关联的全部 MapTag 各记一行（首次注册）
    void LogKeyFrameMapTagFirstRegistrations(KeyFrame *pKF);

    // 导出当前 Map 中全部 MapTag：tag_id + 四角点世界坐标 + 因子权重汇总（可重复调用覆盖）
    // 同时写出 tag_map_mirror_corners.csv（首次赋位姿时未选中的 IPPE 镜像解）
    // verbose=false 用于在线更新，避免刷屏
    bool SaveTagMapCorners(Map &map, bool verbose = true) const;

    // 导出有 MapTag 关联的关键帧位姿（Twc TUM）
    bool SaveTagInitKeyFrames(Map &map) const;

    // 导出定标后的 ORB 初始化地图点 + 两帧关键帧位姿（与 Tag 导出同目录，便于尺度对比）
    // orb_kf_tcw: (timestamp, Tcw)
    bool SaveOrbInitMap(
        const std::vector<Eigen::Vector3f> &points,
        const std::vector<std::pair<double, Sophus::SE3f>> &orb_kf_tcw) const;

    // Shutdown：终局 Tag 角点/平面 + 全部 ORB 点 + KF 轨迹 + ASCII PLY（含 Tag 四边形面）
    bool SaveFinalMap(Map &map, bool verbose = true) const;

    // 关闭轨迹 / 检测 / 首次注册日志文件
    void CloseTrajectory();

    // 地图 Reset 后截断并重开导出文件，避免新旧世界系轨迹拼在同一份文件里
    void ResetSession();

private:
    bool OpenTrajectoryFile();
    bool OpenDetectionLogFile();
    bool OpenFirstRegistrationLogFile();
    bool WriteTagCornersCsv(Map &map, const std::string &filename, bool mirror,
                            bool verbose) const;
    bool WriteKeyFramesTum(Map &map, const std::string &filename,
                           bool tag_associated_only, bool verbose) const;
    bool WriteOrbPointsCsv(const std::vector<Eigen::Vector3f> &points,
                           const std::string &filename, bool verbose) const;
    bool WriteFinalMapPly(Map &map,
                          const std::vector<Eigen::Vector3f> &orb_points,
                          bool verbose) const;

    std::string mOutputDir;
    TagFactorWeightParams mFactorWeight;
    bool mbEnabled{false};
    mutable std::ofstream mTrajFile;
    mutable std::ofstream mDetFile;
    mutable std::ofstream mFirstRegFile;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
