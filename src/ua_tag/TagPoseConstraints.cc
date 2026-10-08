#include "ua_tag/TagPoseConstraints.h"

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagFrameData.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_set>

#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
namespace tag {

bool ModuleEnabledFromSettings(const std::string &settings_path)
{
    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return true;
    const cv::FileNode node = fs["Tag.enable"];
    if(node.empty())
        return true;
    return static_cast<int>(node) != 0;
}

namespace {

constexpr float kDeg2Rad = 0.017453292519943295f;

float Clamp01(float x)
{
    return std::min(1.0f, std::max(0.0f, x));
}

float ReadFloatNode(const cv::FileStorage &fs, const char *key, float fallback)
{
    const cv::FileNode node = fs[key];
    if(node.empty())
        return fallback;
    return static_cast<float>(static_cast<double>(node));
}

TagFactorWeightParams LoadFactorWeightParams(const cv::FileStorage &fs)
{
    TagFactorWeightParams p;
    cv::FileNode node = fs["Tag.factor_weight.enable"];
    if(!node.empty())
        p.enable = static_cast<int>(node) != 0;

    node = fs["Tag.factor_weight.verbose"];
    if(!node.empty())
        p.verbose = static_cast<int>(node) != 0;

    p.alpha = ReadFloatNode(fs, "Tag.factor_weight.alpha", p.alpha);
    p.s_sat = ReadFloatNode(fs, "Tag.factor_weight.s_sat", p.s_sat);
    p.lambda_s = ReadFloatNode(fs, "Tag.factor_weight.lambda_s", p.lambda_s);
    p.theta0_deg = ReadFloatNode(fs, "Tag.factor_weight.theta0_deg", p.theta0_deg);
    p.lambda_theta = ReadFloatNode(fs, "Tag.factor_weight.lambda_theta", p.lambda_theta);
    p.phi0_deg = ReadFloatNode(fs, "Tag.factor_weight.phi0_deg", p.phi0_deg);
    p.e0 = ReadFloatNode(fs, "Tag.factor_weight.e0", p.e0);
    p.lambda_amb = ReadFloatNode(fs, "Tag.factor_weight.lambda_amb", p.lambda_amb);
    p.w_min = ReadFloatNode(fs, "Tag.factor_weight.w_min", p.w_min);

    if(p.alpha <= 0.0f)
        p.alpha = 3.0f;
    if(p.s_sat <= 1e-6f)
        p.s_sat = 400.0f;
    if(p.lambda_s < 0.0f)
        p.lambda_s = 3.0f;
    if(p.theta0_deg <= 1e-6f)
        p.theta0_deg = 60.0f;
    if(p.lambda_theta < 0.0f)
        p.lambda_theta = 1.0f;
    if(p.phi0_deg <= 1e-6f)
        p.phi0_deg = 15.0f;
    if(p.e0 <= 1e-6f)
        p.e0 = 1.0f;
    if(p.lambda_amb < 0.0f)
        p.lambda_amb = 0.5f;
    if(p.w_min < 0.0f)
        p.w_min = 1e-3f;
    if(p.w_min > 1.0f)
        p.w_min = 1.0f;
    return p;
}

const char *CameraName(CameraId cam)
{
    return cam == CameraId::RIGHT ? "R" : "L";
}

}  // namespace

TagFactorWeightParams TagFactorWeightParams::FromSettings(const std::string &settings_path)
{
    TagFactorWeightParams params;
    if(!ModuleEnabledFromSettings(settings_path))
        return params;
    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return params;
    return LoadFactorWeightParams(fs);
}

TagFactorWeight EvaluateTagFactorWeight(const TagObservation &obs,
                                        const TagFactorWeightParams &params)
{
    TagFactorWeight w;

    w.area = std::max(0.0f, obs.observed_area);
    const float s_sat = std::max(params.s_sat, 1e-6f);
    const float xs = w.area / s_sat;
    w.w_s = Clamp01(1.0f - std::exp(-params.lambda_s * xs * xs));

    const TagPoseCandidate *pose_src = nullptr;
    if(obs.pose_estimate.has_value())
    {
        pose_src = obs.pose_estimate->Selected();
        if(!pose_src)
        {
            float best_err = std::numeric_limits<float>::infinity();
            for(const TagPoseCandidate &c : obs.pose_estimate->candidates)
            {
                if(!c.valid)
                    continue;
                if(c.reprojection_error >= 0.0f && c.reprojection_error < best_err)
                {
                    best_err = c.reprojection_error;
                    pose_src = &c;
                }
            }
        }
    }
    if(pose_src)
    {
        Eigen::Vector3f n = pose_src->T_ct.rotationMatrix().col(2);
        const float n_norm = n.norm();
        if(n_norm > 1e-8f)
        {
            n /= n_norm;
            const float abs_cos = std::min(1.0f, std::abs(n.z()));
            w.theta_rad = std::acos(abs_cos);
            w.has_theta = true;
            const float theta0 = std::max(params.theta0_deg * kDeg2Rad, 1e-6f);
            const float xt = w.theta_rad / theta0;
            w.w_theta = Clamp01(std::exp(-params.lambda_theta * xt * xt));
        }
    }

    if(obs.pose_estimate.has_value())
    {
        const TagPoseCandidate &c0 = obs.pose_estimate->candidates[0];
        const TagPoseCandidate &c1 = obs.pose_estimate->candidates[1];
        if(c0.valid && c1.valid)
        {
            const Eigen::Matrix3f R1 = c0.T_ct.rotationMatrix();
            const Eigen::Matrix3f R2 = c1.T_ct.rotationMatrix();
            const float tr = (R1.transpose() * R2).trace();
            const float c = std::min(1.0f, std::max(-1.0f, 0.5f * (tr - 1.0f)));
            w.dphi_rad = std::acos(c);
            w.de = std::abs(c0.reprojection_error - c1.reprojection_error);
            w.has_amb = true;
            const float phi0 = std::max(params.phi0_deg * kDeg2Rad, 1e-6f);
            const float e0 = std::max(params.e0, 1e-6f);
            const float num = (w.dphi_rad / phi0) * (w.dphi_rad / phi0);
            const float den = 1.0f + (w.de / e0) * (w.de / e0);
            w.w_amb = Clamp01(std::exp(-params.lambda_amb * num / den));
        }
    }

    w.w_obs = w.w_s * w.w_theta * w.w_amb;
    w.w_bar = std::max(w.w_obs, params.w_min);
    w.alpha_w = params.alpha * w.w_bar;
    return w;
}

TagFactorWeightSummary SummarizeMapTagFactorWeight(
    const MapTagData &tag, const TagFactorWeightParams &params)
{
    TagFactorWeightSummary summary;
    const MapTagData::KeyFrameObservations observations = tag.GetObservations();

    double sum_area = 0.0;
    double sum_w_s = 0.0;
    double sum_w_theta = 0.0;
    double sum_w_amb = 0.0;
    double sum_w_obs = 0.0;
    double sum_w_bar = 0.0;
    double sum_alpha_w = 0.0;
    float min_w_bar = std::numeric_limits<float>::infinity();
    float max_w_bar = 0.0f;
    int n = 0;

    auto consider = [&](const TagObservation *obs) {
        if(!obs || !obs->IsDetectValid())
            return;
        const TagFactorWeight w = EvaluateTagFactorWeight(*obs, params);
        sum_area += w.area;
        sum_w_s += w.w_s;
        sum_w_theta += w.w_theta;
        sum_w_amb += w.w_amb;
        sum_w_obs += w.w_obs;
        sum_w_bar += w.w_bar;
        sum_alpha_w += w.alpha_w;
        min_w_bar = std::min(min_w_bar, w.w_bar);
        max_w_bar = std::max(max_w_bar, w.w_bar);
        ++n;
    };

    for(const auto &kv : observations)
    {
        KeyFrame *pKF = kv.first;
        if(!pKF || pKF->isBad())
            continue;
        const MapTagData::KeyFrameObservation &idx = kv.second;
        if(idx.leftIndex >= 0 &&
           idx.leftIndex < static_cast<int>(pKF->mTagFrameData.left.size()))
            consider(&pKF->mTagFrameData.left[static_cast<std::size_t>(idx.leftIndex)]);
        if(idx.rightIndex >= 0 &&
           idx.rightIndex < static_cast<int>(pKF->mTagFrameData.right.size()))
            consider(&pKF->mTagFrameData.right[static_cast<std::size_t>(idx.rightIndex)]);
    }

    if(n <= 0)
        return summary;

    const float inv_n = 1.0f / static_cast<float>(n);
    summary.n_obs = n;
    summary.mean_area = static_cast<float>(sum_area) * inv_n;
    summary.mean_w_s = static_cast<float>(sum_w_s) * inv_n;
    summary.mean_w_theta = static_cast<float>(sum_w_theta) * inv_n;
    summary.mean_w_amb = static_cast<float>(sum_w_amb) * inv_n;
    summary.mean_w_obs = static_cast<float>(sum_w_obs) * inv_n;
    summary.mean_w_bar = static_cast<float>(sum_w_bar) * inv_n;
    summary.min_w_bar = min_w_bar;
    summary.max_w_bar = max_w_bar;
    summary.mean_alpha_w = static_cast<float>(sum_alpha_w) * inv_n;
    summary.valid = true;
    return summary;
}

Eigen::Matrix2d TagCornerInformation(float corner_sigma,
                                     const TagObservation &obs,
                                     const TagFactorWeightParams &params,
                                     const char *log_src,
                                     unsigned long log_id)
{
    const float sigma = (corner_sigma > 1e-6f) ? corner_sigma : 2.0f;
    const float inv_sigma2 = 1.0f / (sigma * sigma);
    if(!params.enable)
        return Eigen::Matrix2d::Identity() * inv_sigma2;

    const TagFactorWeight w = EvaluateTagFactorWeight(obs, params);
    if(params.verbose && log_src)
    {
        constexpr float kRad2Deg = 57.29577951308232f;
        std::cout << "[TagW] src=" << log_src
                  << " id=" << log_id
                  << " tag=" << obs.tag_id
                  << " cam=" << CameraName(obs.camera_id)
                  << " S=" << w.area
                  << " theta_deg=" << (w.has_theta ? w.theta_rad * kRad2Deg : -1.0f)
                  << " dphi_deg=" << (w.has_amb ? w.dphi_rad * kRad2Deg : -1.0f)
                  << " de=" << (w.has_amb ? w.de : -1.0f)
                  << " wS=" << w.w_s
                  << " wTh=" << w.w_theta
                  << " wAmb=" << w.w_amb
                  << " wObs=" << w.w_obs
                  << " wBar=" << w.w_bar
                  << " alpha_w=" << w.alpha_w
                  << std::endl;
    }
    return Eigen::Matrix2d::Identity() * (w.alpha_w * inv_sigma2);
}

TagPoseOptParams TagPoseOptParams::FromSettings(const std::string &settings_path)
{
    TagPoseOptParams params;
    if(!ModuleEnabledFromSettings(settings_path))
        return params;

    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return params;

    cv::FileNode node = fs["Tag.pose_optimization"];
    if(!node.empty())
    {
        params.enable = static_cast<int>(node) != 0;
        params.inertial_enable = params.enable;
    }

    // 兼容旧 yaml：仅写了 inertial 键时仍生效；与 pose_optimization 同时存在则以 pose_optimization 为准
    node = fs["Tag.inertial_pose_optimization"];
    if(!node.empty() && fs["Tag.pose_optimization"].empty())
        params.inertial_enable = static_cast<int>(node) != 0;

    node = fs["Tag.corner_sigma"];
    if(!node.empty())
        params.corner_sigma = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.min_inlier_corners"];
    if(!node.empty())
        params.min_inlier_corners = static_cast<int>(node);

    node = fs["Tag.fixed_only"];
    if(!node.empty())
        params.fixed_only = static_cast<int>(node) != 0;

    node = fs["Tag.verbose"];
    if(!node.empty())
        params.verbose = static_cast<int>(node) != 0;

    if(params.corner_sigma <= 1e-6f)
        params.corner_sigma = 2.0f;
    if(params.min_inlier_corners < 1)
        params.min_inlier_corners = 1;
    if(params.min_inlier_corners > 4)
        params.min_inlier_corners = 4;

    params.factor_weight = LoadFactorWeightParams(fs);
    return params;
}

TagLocalBAParams TagLocalBAParams::FromSettings(const std::string &settings_path)
{
    TagLocalBAParams params;
    if(!ModuleEnabledFromSettings(settings_path))
        return params;

    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return params;

    cv::FileNode node = fs["Tag.local_ba.enable"];
    if(!node.empty())
        params.enable = static_cast<int>(node) != 0;

    node = fs["Tag.local_ba.optimize_active"];
    if(!node.empty())
        params.optimize_active = static_cast<int>(node) != 0;

    node = fs["Tag.local_ba.corner_sigma"];
    if(!node.empty())
        params.corner_sigma = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.local_ba.min_inlier_corners"];
    if(!node.empty())
        params.min_inlier_corners = static_cast<int>(node);

    node = fs["Tag.local_ba.verbose"];
    if(!node.empty())
        params.verbose = static_cast<int>(node) != 0;

    if(params.corner_sigma <= 1e-6f)
        params.corner_sigma = 2.0f;
    if(params.min_inlier_corners < 1)
        params.min_inlier_corners = 1;
    if(params.min_inlier_corners > 4)
        params.min_inlier_corners = 4;

    params.factor_weight = LoadFactorWeightParams(fs);
    return params;
}

TagLoopParams TagLoopParams::FromSettings(const std::string &settings_path)
{
    TagLoopParams params;
    if(!ModuleEnabledFromSettings(settings_path))
        return params;

    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return params;

    cv::FileNode node = fs["Tag.loop.enable"];
    if(!node.empty())
        params.enable = static_cast<int>(node) != 0;

    node = fs["Tag.loop.min_history_observations"];
    if(!node.empty())
        params.min_history_observations = static_cast<int>(node);

    node = fs["Tag.loop.min_keyframe_gap"];
    if(!node.empty())
        params.min_keyframe_gap = static_cast<int>(node);

    node = fs["Tag.loop.min_reobservation_gap_kfs"];
    if(!node.empty())
        params.min_reobservation_gap_kfs = static_cast<int>(node);

    node = fs["Tag.loop.min_historical_kf_map_points"];
    if(!node.empty())
        params.min_historical_kf_map_points = static_cast<int>(node);

    node = fs["Tag.loop.max_ippe_reprojection_error"];
    if(!node.empty())
        params.max_ippe_reprojection_error = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.loop.max_stereo_translation_error"];
    if(!node.empty())
        params.max_stereo_translation_error = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.loop.max_stereo_rotation_error"];
    if(!node.empty())
        params.max_stereo_rotation_error = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.loop.max_roll_pitch_correction"];
    if(!node.empty())
        params.max_roll_pitch_correction = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.loop.max_yaw_correction"];
    if(!node.empty())
        params.max_yaw_correction = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.loop.min_consistent_tags"];
    if(!node.empty())
        params.min_consistent_tags = static_cast<int>(node);

    node = fs["Tag.loop.reference_timeout_kfs"];
    if(!node.empty())
        params.reference_timeout_kfs = static_cast<int>(node);

    node = fs["Tag.loop.verbose"];
    if(!node.empty())
        params.verbose = static_cast<int>(node) != 0;

    if(params.min_history_observations < 1)
        params.min_history_observations = 1;
    if(params.min_keyframe_gap < 1)
        params.min_keyframe_gap = 1;
    if(params.min_reobservation_gap_kfs < 1)
        params.min_reobservation_gap_kfs = 1;
    if(params.min_historical_kf_map_points < 0)
        params.min_historical_kf_map_points = 0;
    if(params.min_consistent_tags < 1)
        params.min_consistent_tags = 1;
    if(params.reference_timeout_kfs < 1)
        params.reference_timeout_kfs = 1;
    if(params.max_ippe_reprojection_error <= 0.0f)
        params.max_ippe_reprojection_error = 5.0f;

    return params;
}

namespace {

bool ObservationEligible(const TagObservation &obs)
{
    // 仅看检测有效性；优化外点不得阻止同帧再次加入约束
    return obs.IsDetectValid();
}

bool CollectNeededTagIds(const Frame *pFrame, std::unordered_set<int> &ids)
{
    ids.clear();
    if(!pFrame)
        return false;

    for(const TagObservation &obs : pFrame->mTagFrameData.left)
    {
        if(ObservationEligible(obs))
            ids.insert(obs.tag_id);
    }
    for(const TagObservation &obs : pFrame->mTagFrameData.right)
    {
        if(ObservationEligible(obs))
            ids.insert(obs.tag_id);
    }
    return !ids.empty();
}

bool MapTagEligible(const MapTagData *pTag, const TagPoseOptParams &params,
                    bool inertial)
{
    if(!pTag || pTag->IsBad() || !pTag->HasWorldCorners())
        return false;
    if(inertial)
        return pTag->GetState() == MapTagState::ACTIVE;
    if(params.fixed_only)
        return pTag->GetState() == MapTagState::FIXED_ANCHOR;
    return pTag->GetState() == MapTagState::FIXED_ANCHOR ||
           pTag->GetState() == MapTagState::ACTIVE;
}

}  // namespace

std::vector<TagPoseConstraint> BuildTagMapSnapshot(
    Map *pMap, Frame *pFrame, const TagPoseOptParams &params, bool inertial)
{
    std::vector<TagPoseConstraint> out;
    const bool enabled = inertial ? params.inertial_enable : params.enable;
    if(!enabled || !pMap || !pFrame || !pMap->IsTagInitialized())
        return out;

    std::unordered_set<int> needed;
    if(!CollectNeededTagIds(pFrame, needed))
        return out;

    out.reserve(needed.size());
    for(int tag_id : needed)
    {
        const Map::MapTagPtr pTag = pMap->GetMapTag(tag_id);
        if(!MapTagEligible(pTag.get(), params, inertial))
            continue;

        const auto corners_f = pTag->GetWorldCorners();
        TagPoseConstraint c;
        c.tagId = tag_id;
        for(int i = 0; i < 4; ++i)
            c.worldCorners[i] = corners_f[i].cast<double>();
        out.push_back(c);
    }

    if(params.verbose)
    {
        std::cout << (inertial ? "[TagInertialPoseOpt]" : "[TagPoseOpt]")
                  << " snapshot tags=" << out.size()
                  << " needed=" << needed.size()
                  << " frame_id=" << pFrame->mnId << std::endl;
    }
    return out;
}

}  // namespace tag
}  // namespace ORB_SLAM3
