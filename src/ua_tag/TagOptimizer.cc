#include "ua_tag/TagOptimizer.h"

#include "Frame.h"
#include "CameraModels/GeometricCamera.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagObservation.h"

#include <iostream>
#include <unordered_set>
#include <vector>

#include <cstddef>

#ifdef HAS_GTSAM
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3Fisheye.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#endif

namespace ORB_SLAM3 {
namespace tag {

namespace {

#ifdef HAS_GTSAM

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;
using ProjectionFactorPinhole =
    gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;
using ProjectionFactorFisheye =
    gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3Fisheye>;

constexpr double kPixelSigma = 1.0;
constexpr double kChi2Mono = 5.991;  // 2 DoF, 95%
constexpr double kHuberDelta = 2.44774674741;  // sqrt(5.991)
constexpr int kMinInlierCorners = 3;
constexpr int kMaxIterations = 10;

gtsam::Pose3 ToGtsamPose(const Sophus::SE3f &T)
{
    return gtsam::Pose3(gtsam::Rot3(T.rotationMatrix().cast<double>()),
                        gtsam::Point3(T.translation().cast<double>()));
}

Sophus::SE3f FromGtsamPose(const gtsam::Pose3 &T)
{
    return Sophus::SE3f(T.rotation().matrix().cast<float>(),
                        T.translation().cast<float>());
}

bool IsFisheyeCamera(GeometricCamera *camera)
{
    return camera && camera->GetType() == GeometricCamera::CAM_FISHEYE;
}

boost::shared_ptr<gtsam::Cal3_S2> MakePinholeCalib(GeometricCamera *camera)
{
    if(!camera)
        return nullptr;
    const Eigen::Matrix3f K = camera->toK_();
    return boost::make_shared<gtsam::Cal3_S2>(
        static_cast<double>(K(0, 0)), static_cast<double>(K(1, 1)), 0.0,
        static_cast<double>(K(0, 2)), static_cast<double>(K(1, 2)));
}

boost::shared_ptr<gtsam::Cal3Fisheye> MakeFisheyeCalib(GeometricCamera *camera)
{
    if(!camera || camera->size() < 8)
        return nullptr;
    return boost::make_shared<gtsam::Cal3Fisheye>(
        static_cast<double>(camera->getParameter(0)),
        static_cast<double>(camera->getParameter(1)),
        0.0,
        static_cast<double>(camera->getParameter(2)),
        static_cast<double>(camera->getParameter(3)),
        static_cast<double>(camera->getParameter(4)),
        static_cast<double>(camera->getParameter(5)),
        static_cast<double>(camera->getParameter(6)),
        static_cast<double>(camera->getParameter(7)));
}

struct CornerMeas
{
    gtsam::Point3 pw;
    gtsam::Point2 z;  // 鱼眼：corners_raw；针孔：corners_undistorted
    bool use_fisheye = false;
    boost::shared_ptr<gtsam::Cal3_S2> K_pin;
    boost::shared_ptr<gtsam::Cal3Fisheye> K_fish;
    boost::optional<gtsam::Pose3> body_P_sensor;
    TagObservation *obs = nullptr;
};

// init 第二帧 motion-only：容器内 Tag 世界位姿在本优化中临时当作锚点使用，
// 不要求 MapTagState::FIXED_ANCHOR（初始化 Tag 现为 ACTIVE）。
const MapTagData *FindFixedTagWithCorners(
    const TagOptimizer::TagContainer &fixed_tags, int tag_id)
{
    const auto it = fixed_tags.find(tag_id);
    if(it == fixed_tags.end() || !it->second)
        return nullptr;
    const MapTagData *map_tag = it->second.get();
    if(!map_tag->HasPose() || !map_tag->HasWorldCorners())
        return nullptr;
    return map_tag;
}

void CollectFixedTagCornerMeasurements(Frame &frame,
                                       const TagOptimizer::TagContainer &fixed_tags,
                                       std::vector<CornerMeas> &meas)
{
    meas.clear();

    if(!frame.mpCamera)
    {
        std::cerr << "[TagOptimizer] ERROR: left camera model missing" << std::endl;
        return;
    }
    if(!frame.mTagFrameData.right.empty() && !frame.mpCamera2)
    {
        std::cerr << "[TagOptimizer] ERROR: right Tag measurements require Camera2 "
                     "(no left-camera fallback)"
                  << std::endl;
        return;
    }

    const bool fish_left = IsFisheyeCamera(frame.mpCamera);
    const bool fish_right = IsFisheyeCamera(frame.mpCamera2);

    const auto K_left_pin = MakePinholeCalib(frame.mpCamera);
    const auto K_right_pin = MakePinholeCalib(frame.mpCamera2);
    const auto K_left_fish = MakeFisheyeCalib(frame.mpCamera);
    const auto K_right_fish = MakeFisheyeCalib(frame.mpCamera2);

    boost::optional<gtsam::Pose3> T_body_right = boost::none;
    if(frame.mpCamera2)
        T_body_right = ToGtsamPose(frame.GetRelativePoseTlr());

    auto append_obs = [&](TagObservation &obs, bool use_fisheye,
                          const boost::shared_ptr<gtsam::Cal3_S2> &K_pin,
                          const boost::shared_ptr<gtsam::Cal3Fisheye> &K_fish,
                          const boost::optional<gtsam::Pose3> &body_P_sensor) {
        if(use_fisheye && !K_fish)
            return;
        if(!use_fisheye && !K_pin)
            return;

        const MapTagData *map_tag = FindFixedTagWithCorners(fixed_tags, obs.tag_id);
        if(!map_tag)
            return;

        if(!obs.IsDetectValid())
            return;
        const auto &pw = map_tag->GetWorldCorners();
        // 鱼眼：原图像素 + Cal3Fisheye（与 ORB KB / BA 一致）
        // 针孔：去畸变像素 + Cal3_S2
        const auto &uv = use_fisheye ? obs.corners_raw : obs.corners_undistorted;

        for(int i = 0; i < 4; ++i)
        {
            CornerMeas m;
            m.pw = gtsam::Point3(pw[i].x(), pw[i].y(), pw[i].z());
            m.z = gtsam::Point2(uv[i].x, uv[i].y);
            m.use_fisheye = use_fisheye;
            m.K_pin = K_pin;
            m.K_fish = K_fish;
            m.body_P_sensor = body_P_sensor;
            m.obs = &obs;
            meas.push_back(m);
        }
    };

    for(TagObservation &obs : frame.mTagFrameData.left)
        append_obs(obs, fish_left, K_left_pin, K_left_fish, boost::none);
    for(TagObservation &obs : frame.mTagFrameData.right)
        append_obs(obs, fish_right, K_right_pin, K_right_fish, T_body_right);
}

gtsam::SharedNoiseModel MakeHuberNoise()
{
    const auto gaussian = gtsam::noiseModel::Isotropic::Sigma(2, kPixelSigma);
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(kHuberDelta), gaussian);
}

gtsam::SharedNoiseModel MakeGaussianNoise()
{
    return gtsam::noiseModel::Isotropic::Sigma(2, kPixelSigma);
}

void BuildGraphAndValues(const std::vector<CornerMeas> &meas,
                         const std::vector<char> &use_mask,
                         const gtsam::SharedNoiseModel &proj_noise,
                         const gtsam::Pose3 &Twc0,
                         gtsam::NonlinearFactorGraph &graph,
                         gtsam::Values &values)
{
    graph = gtsam::NonlinearFactorGraph();
    values.clear();
    values.insert(X(0), Twc0);

    for(std::size_t i = 0; i < meas.size(); ++i)
    {
        if(!use_mask.empty() && !use_mask[i])
            continue;

        const CornerMeas &m = meas[i];
        const gtsam::Key point_key = L(static_cast<int>(i));

        values.insert(point_key, m.pw);
        graph.emplace_shared<gtsam::NonlinearEquality<gtsam::Point3>>(point_key,
                                                                      m.pw);

        if(m.use_fisheye)
        {
            graph.emplace_shared<ProjectionFactorFisheye>(
                m.z, proj_noise, X(0), point_key, m.K_fish, m.body_P_sensor);
        }
        else
        {
            graph.emplace_shared<ProjectionFactorPinhole>(
                m.z, proj_noise, X(0), point_key, m.K_pin, m.body_P_sensor);
        }
    }
}

double CornerChi2(const CornerMeas &m, std::size_t idx, const gtsam::Pose3 &Twc)
{
    gtsam::Values values;
    values.insert(X(0), Twc);
    values.insert(L(static_cast<int>(idx)), m.pw);

    gtsam::Vector e;
    if(m.use_fisheye)
    {
        const ProjectionFactorFisheye factor(m.z, MakeGaussianNoise(), X(0),
                                             L(static_cast<int>(idx)), m.K_fish,
                                             m.body_P_sensor);
        e = factor.unwhitenedError(values);
    }
    else
    {
        const ProjectionFactorPinhole factor(m.z, MakeGaussianNoise(), X(0),
                                             L(static_cast<int>(idx)), m.K_pin,
                                             m.body_P_sensor);
        e = factor.unwhitenedError(values);
    }
    return e.squaredNorm() / (kPixelSigma * kPixelSigma);
}

#endif  // HAS_GTSAM

// 用第二帧上与 fixed_tags 共视的 Tag IPPE，估计 Tcw 初值：Tcw = T_ct * T_wt^{-1}
std::optional<Sophus::SE3f> EstimateInitialTcwFromFixedTags(
    const Frame &frame, const TagOptimizer::TagContainer &fixed_tags)
{
    auto try_obs = [&](const TagObservation &obs) -> std::optional<Sophus::SE3f> {
        if(!obs.IsDetectValid() || !obs.pose_estimate.has_value())
            return std::nullopt;

        const auto it = fixed_tags.find(obs.tag_id);
        if(it == fixed_tags.end() || !it->second || !it->second->HasPose())
            return std::nullopt;

        const TagPoseEstimate &est = *obs.pose_estimate;
        const TagPoseCandidate *cand = est.Selected();
        if(!cand || !cand->valid)
        {
            if(est.candidates[0].valid)
                cand = &est.candidates[0];
            else if(est.candidates[1].valid)
                cand = &est.candidates[1];
            else
                return std::nullopt;
        }

        // Tcw 为左目位姿：T_cL_t * T_wt^{-1}；右目 IPPE 先经 T_lr 转到左目
        const Sophus::SE3f T_cL_t = ExpressTagPoseInLeftCamera(
            obs.camera_id, cand->T_ct, frame.GetRelativePoseTlr());
        return T_cL_t * it->second->GetPose().inverse();
    };

    for(const TagObservation &obs : frame.mTagFrameData.left)
    {
        if(auto Tcw = try_obs(obs))
            return Tcw;
    }
    for(const TagObservation &obs : frame.mTagFrameData.right)
    {
        if(auto Tcw = try_obs(obs))
            return Tcw;
    }
    return std::nullopt;
}

}  // namespace

bool TagOptimizer::PoseOptimization(Frame &frame,
                                    const TagContainer &fixed_tags,
                                    const std::optional<Sophus::SE3f> &Tcw_pred,
                                    bool verbose)
{
#ifndef HAS_GTSAM
    (void)fixed_tags;
    (void)Tcw_pred;
    if(verbose)
        std::cout << "[TagOptimizer] fail step=no_gtsam frame_id=" << frame.mnId
                  << std::endl;
    return false;
#else
    Sophus::SE3f Tcw0;
    if(Tcw_pred)
        Tcw0 = *Tcw_pred;
    else if(frame.HasPose())
        Tcw0 = frame.GetPose();
    else
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=no_initial_pose frame_id="
                      << frame.mnId << std::endl;
        return false;
    }

    // 同帧多次优化前清临时外点；检测外点保持不变
    frame.mTagFrameData.ResetOptOutliers();

    std::vector<CornerMeas> meas;
    CollectFixedTagCornerMeasurements(frame, fixed_tags, meas);
    if(static_cast<int>(meas.size()) < kMinInlierCorners)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=too_few_measurements frame_id="
                      << frame.mnId << " n_meas=" << meas.size()
                      << " min_meas=" << kMinInlierCorners << std::endl;
        return false;
    }

    std::unordered_set<int> covis_tag_ids;
    for(const CornerMeas &m : meas)
    {
        if(m.obs)
            covis_tag_ids.insert(m.obs->tag_id);
    }
    const int n_covis_tags = static_cast<int>(covis_tag_ids.size());
    if(n_covis_tags <= 0)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=no_covis_tags frame_id="
                      << frame.mnId << " n_meas=" << meas.size() << std::endl;
        return false;
    }
    const int min_inliers_to_accept = 3 * n_covis_tags;

    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setMaxIterations(kMaxIterations);
    lm_params.setVerbosityLM("SILENT");

    const gtsam::Pose3 Twc_init = ToGtsamPose(Tcw0.inverse());  // Tcw -> Twc

    std::vector<char> use_all(meas.size(), 1);
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    BuildGraphAndValues(meas, use_all, MakeHuberNoise(), Twc_init, graph, values);

    values = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();
    gtsam::Pose3 Twc = values.at<gtsam::Pose3>(X(0));

    std::vector<char> inlier(meas.size(), 0);
    int n_inliers = 0;
    for(std::size_t i = 0; i < meas.size(); ++i)
    {
        if(CornerChi2(meas[i], i, Twc) <= kChi2Mono)
        {
            inlier[i] = 1;
            ++n_inliers;
        }
    }

    if(n_inliers < kMinInlierCorners)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=huber_too_few_inliers frame_id="
                      << frame.mnId << " n_inliers=" << n_inliers
                      << " min_inliers=" << kMinInlierCorners
                      << " n_meas=" << meas.size()
                      << " n_covis_tags=" << n_covis_tags << std::endl;
        return false;
    }

    BuildGraphAndValues(meas, inlier, MakeGaussianNoise(), Twc, graph, values);
    values = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();
    Twc = values.at<gtsam::Pose3>(X(0));

    n_inliers = 0;
    for(std::size_t i = 0; i < meas.size(); ++i)
    {
        if(!inlier[i])
            continue;
        if(CornerChi2(meas[i], i, Twc) <= kChi2Mono)
            ++n_inliers;
        else
            inlier[i] = 0;
    }

    for(TagObservation &obs : frame.mTagFrameData.left)
        obs.ResetOptOutliers();
    for(TagObservation &obs : frame.mTagFrameData.right)
        obs.ResetOptOutliers();

    for(std::size_t i = 0; i < meas.size();)
    {
        TagObservation *obs = meas[i].obs;
        int n_bad = 0;
        std::size_t j = i;
        while(j < meas.size() && meas[j].obs == obs)
        {
            if(!inlier[j])
                ++n_bad;
            ++j;
        }
        if(obs && n_bad >= 3)
            obs->is_opt_outlier = true;
        i = j;
    }

    if(n_inliers <= min_inliers_to_accept)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=final_inlier_ratio frame_id="
                      << frame.mnId << " n_inliers=" << n_inliers
                      << " min_inliers_to_accept=" << min_inliers_to_accept
                      << " n_covis_tags=" << n_covis_tags
                      << " n_meas=" << meas.size() << std::endl;
        return false;
    }

    frame.SetPose(FromGtsamPose(Twc).inverse());  // Twc -> Tcw
    if(verbose)
    {
        std::cout << "[TagOptimizer] frame_id=" << frame.mnId
                  << " n_inliers=" << n_inliers
                  << " n_covis_tags=" << n_covis_tags << std::endl;
    }
    return true;
#endif
}

bool TagOptimizer::PoseOptimizationForSecondFrame(
    Frame &second_frame,
    const TagContainer &fixed_tags,
    Sophus::SE3f &Tcw_out,
    const std::optional<Sophus::SE3f> &Tcw_pred,
    bool verbose)
{
    std::optional<Sophus::SE3f> pred = Tcw_pred;
    if(!pred && !second_frame.HasPose())
        pred = EstimateInitialTcwFromFixedTags(second_frame, fixed_tags);

    if(!PoseOptimization(second_frame, fixed_tags, pred, verbose))
        return false;

    if(!second_frame.HasPose())
        return false;

    Tcw_out = second_frame.GetPose();
    return true;
}

}  // namespace tag
}  // namespace ORB_SLAM3
