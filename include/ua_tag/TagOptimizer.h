#ifndef UA_TAG_TAG_OPTIMIZER_H
#define UA_TAG_TAG_OPTIMIZER_H

#include <memory>
#include <optional>
#include <unordered_map>

#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Frame;

namespace tag {

class MapTagData;

// Tag 优化 API（无独立线程；由 Tracking / TagTracker 同步调用）
class TagOptimizer
{
public:
    using MapTagPtr = std::shared_ptr<MapTagData>;
    using TagContainer = std::unordered_map<int, MapTagPtr>;

    TagOptimizer() = delete;

    // Motion-only BA（GTSAM）核心接口：直接使用 FIXED Tag 及其世界系 3D 角点。
    //   1) FIXED Tag 世界系角点作为 Point3 状态，NonlinearEquality 钉死
    //   2) 去畸变角点 + GenericProjectionFactor<Pose3,Point3,Cal3_S2>（Huber）
    //   3) 优化后按 chi2 标外点
    //   4) 仅内点、去 Huber 再优化一轮
    // 初值优先 Tcw_pred，否则用 frame 已有 Pose；成功则 SetPose 并返回 true。
    // 失败不清除 frame 上已有 Pose（保留匀速预测兜底），返回 false。
    static bool PoseOptimization(
        Frame &frame,
        const TagContainer &fixed_tags,
        const std::optional<Sophus::SE3f> &Tcw_pred = std::nullopt,
        bool verbose = false);

    // 单帧 Tag 初始化后估计第二帧位姿：复用 PoseOptimization(TagContainer)。
    // fixed_tags：本优化中临时固定世界位姿的 Tag（通常来自 Result::tags，状态可为 ACTIVE）。
    // 共视 Tag 门槛由 TagInitializer::TryInitialize 预先检查，此处不再重复判断。
    // 若未给 Tcw_pred 且 frame 无 Pose，则用共视 Tag 的 IPPE 估初值。
    // 成功则写入 Tcw_out（与 frame.GetPose() 一致）；失败不修改 Tcw_out。
    static bool PoseOptimizationForSecondFrame(
        Frame &second_frame,
        const TagContainer &fixed_tags,
        Sophus::SE3f &Tcw_out,
        const std::optional<Sophus::SE3f> &Tcw_pred = std::nullopt,
        bool verbose = false);
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
