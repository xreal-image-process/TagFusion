#ifndef UA_TAG_TAG_POSE_CONSTRAINTS_H
#define UA_TAG_TAG_POSE_CONSTRAINTS_H

#include "ua_tag/TagObservation.h"

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace ORB_SLAM3 {

class Frame;
class Map;

namespace tag {

class MapTagData;

// Tag.enable：总开关。缺省 true；0 时不检测、不建图、不进优化、不因 Tag 插 KF
bool ModuleEnabledFromSettings(const std::string &settings_path);

// 只读快照：优化期间不持有 Map/MapTag 锁
struct TagPoseConstraint
{
    int tagId = -1;
    std::array<Eigen::Vector3d, 4> worldCorners{};
};

// Tag 角点因子自适应权重（PoseOpt / LocalBA / GBA / 回环 4DoF 共用 yaml）
// Ω_tag = (α * w_bar / σ²) I₂，w_bar = max(w_S w_θ w_amb, w_min)
// Tag.factor_weight.enable=0 时退回 Ω = I/σ²
struct TagFactorWeightParams
{
    bool enable = false;
    bool verbose = false;
    float alpha = 3.0f;
    float s_sat = 400.0f;
    float lambda_s = 3.0f;
    float theta0_deg = 60.0f;
    float lambda_theta = 1.0f;
    float phi0_deg = 15.0f;
    float e0 = 1.0f;
    float lambda_amb = 0.5f;
    float w_min = 1e-3f;

    static TagFactorWeightParams FromSettings(const std::string &settings_path);
};

struct TagFactorWeight
{
    float area = 0.0f;
    float theta_rad = 0.0f;
    float dphi_rad = 0.0f;
    float de = 0.0f;
    float w_s = 1.0f;
    float w_theta = 1.0f;
    float w_amb = 1.0f;
    float w_obs = 1.0f;
    float w_bar = 1.0f;
    float alpha_w = 1.0f;
    bool has_theta = false;
    bool has_amb = false;
};

TagFactorWeight EvaluateTagFactorWeight(const TagObservation &obs,
                                        const TagFactorWeightParams &params);

// 对 MapTag 全部检测有效 KF 观测汇总（与 [TagW] 同一套 EvaluateTagFactorWeight）
struct TagFactorWeightSummary
{
    int n_obs = 0;
    float mean_area = 0.0f;
    float mean_w_s = 1.0f;
    float mean_w_theta = 1.0f;
    float mean_w_amb = 1.0f;
    float mean_w_obs = 1.0f;
    float mean_w_bar = 1.0f;
    float min_w_bar = 1.0f;
    float max_w_bar = 0.0f;
    float mean_alpha_w = 1.0f;
    bool valid = false;
};

TagFactorWeightSummary SummarizeMapTagFactorWeight(
    const MapTagData &tag, const TagFactorWeightParams &params);

// enable=0：I/σ²；enable=1：(α w_bar / σ²) I。verbose 时每观测打一行（调用方每观测调用一次）
Eigen::Matrix2d TagCornerInformation(float corner_sigma,
                                     const TagObservation &obs,
                                     const TagFactorWeightParams &params,
                                     const char *log_src = nullptr,
                                     unsigned long log_id = 0);

struct TagPoseOptParams
{
    bool enable = false;                  // Tag.pose_optimization：PoseOptimization 与 PoseInertial* 共用
    bool inertial_enable = false;         // 与 enable 同步；仅旧 yaml 单独写 Tag.inertial_pose_optimization 时生效
    float corner_sigma = 2.0f;            // 纯视觉 / 惯性共用
    int min_inlier_corners = 3;           // 每相机每 Tag：至少 N/4 角点内点才保留；共用
    bool fixed_only = true;               // 仅纯视觉快照使用；惯性路径固定筛 ACTIVE
    bool verbose = false;                 // 共用
    TagFactorWeightParams factor_weight;

    // Tag.pose_optimization 同时控制纯视觉 PoseOptimization 与惯性 PoseInertial*
    // （两套代码分支，运行时只会走其中一条）。旧键 Tag.inertial_pose_optimization 仅作兼容。
    static TagPoseOptParams FromSettings(const std::string &settings_path);
};

// Local BA 中 Tag 刚体位姿 + 四角点重投影（独立于 Tracking 的 TagPoseOptParams）
struct TagLocalBAParams
{
    bool enable = false;             // Tag.local_ba.enable，默认关闭
    bool optimize_active = true;     // 1=ACTIVE 可优化；0=仅 FIXED_ANCHOR 入图
    float corner_sigma = 2.0f;
    int min_inlier_corners = 3;
    bool verbose = false;
    TagFactorWeightParams factor_weight;

    static TagLocalBAParams FromSettings(const std::string &settings_path);
};

// Tag 回环：独立于 BoW 的检测分支，最终复用 CorrectLoop 主流程
struct TagLoopParams
{
    bool enable = false;

    int min_history_observations = 2;
    int min_keyframe_gap = 30;
    int min_reobservation_gap_kfs = 30;     // 最近一次有效观测间隔；过小视为连续可见
    int min_historical_kf_map_points = 20;  // 历史 KF 地图点质量门槛

    float max_ippe_reprojection_error = 5.0f;
    float max_stereo_translation_error = 0.20f;
    float max_stereo_rotation_error = 0.20f;

    float max_roll_pitch_correction = 0.15f;
    float max_yaw_correction = 1.57f;

    int min_consistent_tags = 1;
    int reference_timeout_kfs = 4;  // 连续若干 KF 未见则丢弃冻结参考
    bool verbose = false;

    static TagLoopParams FromSettings(const std::string &settings_path);
};

struct TagLocalBAStats
{
    int num_tags = 0;
    int num_tag_edges = 0;
    int num_tag_outliers = 0;
};

// 在进入 PoseOptimization / PoseInertial* 前查询 MapTag，生成与当前帧观测对齐的约束快照。
// 只使用检测有效观测（IsDetectValid）；忽略 is_opt_outlier。
// 联合优化入口会 ResetOptOutliers，同帧可再次加入 Tag。
// inertial=false：看 params.enable；筛选受 params.fixed_only 控制。
// inertial=true：看 params.inertial_enable；仅 ACTIVE（不读 fixed_only）。
std::vector<TagPoseConstraint> BuildTagMapSnapshot(
    Map *pMap, Frame *pFrame, const TagPoseOptParams &params,
    bool inertial = false);

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
