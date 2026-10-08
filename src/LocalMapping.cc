/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"
#include "Map.h"
#include "CameraModels/GeometricCamera.h"
#include "ua_tag/AprilTagDetector.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagObservation.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling=0;

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTagLocalBAParams(const tag::TagLocalBAParams &params)
{
    mTagLocalBAParams = params;
    if(mTagLocalBAParams.enable)
    {
        std::cout << "Tag LocalBA enabled"
                  << " (optimize_active=" << (mTagLocalBAParams.optimize_active ? 1 : 0)
                  << ", corner_sigma=" << mTagLocalBAParams.corner_sigma
                  << ", min_inlier_corners=" << mTagLocalBAParams.min_inlier_corners
                  << ", factor_weight=" << (mTagLocalBAParams.factor_weight.enable ? 1 : 0)
                  << ")" << std::endl;
    }
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{
    // Offline architecture: processing is driven by ProcessUntilIdle() from Track*().
    mbFinished = true;
}

bool LocalMapping::HasPendingKeyFrames()
{
    return CheckNewKeyFrames();
}

void LocalMapping::ProcessUntilIdle()
{
    while(ProcessOneKeyFrame())
    {
    }
    ResetIfRequested();
}

bool LocalMapping::ProcessOneKeyFrame()
{
    if(!CheckNewKeyFrames() || mbBadImu)
        return false;

#ifdef REGISTER_TIMES
    double timeLBA_ms = 0;
    double timeKFCulling_ms = 0;

    std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
    ProcessNewKeyFrame();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

    double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
    vdKFInsert_ms.push_back(timeProcessKF);
#endif

    std::cout << "[KF " << mpCurrentKeyFrame->mnId << "] LocalMapping begin" << std::endl;

    MapPointCulling();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

    double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
    vdMPCulling_ms.push_back(timeMPCulling);
#endif

    CreateNewMapPoints();

    mbAbortBA = false;

    if(!CheckNewKeyFrames())
    {
        SearchInNeighbors();
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

    double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
    vdMPCreation_ms.push_back(timeMPCreation);
#endif

    bool b_doneLBA = false;
    int num_FixedKF_BA = 0;
    int num_OptKF_BA = 0;
    int num_MPs_BA = 0;
    int num_edges_BA = 0;

    if(mpAtlas->KeyFramesInMap()>2)
    {
        if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
        {
            float dist = 0.f;
            if(mpCurrentKeyFrame->mPrevKF)
            {
                dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm();
                if(mpCurrentKeyFrame->mPrevKF->mPrevKF)
                    dist += (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();
            }

            if(mpCurrentKeyFrame->mPrevKF && dist>0.05)
                mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
            if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
            {
                if((mTinit<10.f) && (dist<0.02))
                {
                    cout << "Not enough motion for initializing. Reseting..." << endl;
                    unique_lock<mutex> lock(mMutexReset);
                    mbResetRequestedActiveMap = true;
                    mpMapToReset = mpCurrentKeyFrame->GetMap();
                    mbBadImu = true;
                }
            }

            bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
            tag::TagLocalBAStats tagLbaStats;
            Optimizer::LocalInertialBA(
                mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),
                num_FixedKF_BA, num_OptKF_BA, num_MPs_BA, num_edges_BA,
                bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2(),
                mTagLocalBAParams, &tagLbaStats);
            b_doneLBA = true;
        }
        else
        {
            tag::TagLocalBAStats tagLbaStats;
            Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA,
                                             mTagLocalBAParams, &tagLbaStats);
            b_doneLBA = true;
        }
    }

    if(b_doneLBA)
        std::cout << "[KF " << mpCurrentKeyFrame->mnId << "] Local BA finished" << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

    if(b_doneLBA)
    {
        timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
        vdLBA_ms.push_back(timeLBA_ms);

        nLBA_exec += 1;
        if(mbAbortBA)
        {
            nLBA_abort += 1;
        }
        vnLBA_edges.push_back(num_edges_BA);
        vnLBA_KFopt.push_back(num_OptKF_BA);
        vnLBA_KFfixed.push_back(num_FixedKF_BA);
        vnLBA_MPs.push_back(num_MPs_BA);
    }

#endif

    if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
    {
        if (mbMonocular)
            InitializeIMU(1e2, 1e10, true);
        else
            InitializeIMU(1e2, 1e5, true);
    }

    KeyFrameCulling();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

    timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
    vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif

    if ((mTinit<50.0f) && mbInertial)
    {
        if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK)
        {
            if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()){
                if (mTinit>5.0f)
                {
                    cout << "start VIBA 1" << endl;
                    mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                    if (mbMonocular)
                        InitializeIMU(1.f, 1e5, true);
                    else
                        InitializeIMU(1.f, 1e5, true);

                    cout << "end VIBA 1" << endl;
                }
            }
            else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()){
                if (mTinit>15.0f){
                    cout << "start VIBA 2" << endl;
                    mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                    if (mbMonocular)
                        InitializeIMU(0.f, 0.f, true);
                    else
                        InitializeIMU(0.f, 0.f, true);

                    cout << "end VIBA 2" << endl;
                }
            }

            if (((mpAtlas->KeyFramesInMap())<=200) &&
                    ((mTinit>25.0f && mTinit<25.5f)||
                    (mTinit>35.0f && mTinit<35.5f)||
                    (mTinit>45.0f && mTinit<45.5f)||
                    (mTinit>55.0f && mTinit<55.5f)||
                    (mTinit>65.0f && mTinit<65.5f)||
                    (mTinit>75.0f && mTinit<75.5f))){
                if (mbMonocular)
                    ScaleRefinement();
            }
        }
    }

#ifdef REGISTER_TIMES
    vdLBASync_ms.push_back(timeKFCulling_ms);
    vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

    mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

    double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
    vdLMTotal_ms.push_back(timeLocalMap);
#endif

    return true;
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map（须先于 MapTag 关联，供 CheckMapTagAssociations 等使用）
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);

    // 建立 KF ↔ MapTag 双向关联，并扩展 Tag 地图（新 ID → CANDIDATE）
    ProcessTagObservations(mpCurrentKeyFrame);
}

bool LocalMapping::InitializeMapTagPose(tag::MapTagData *pTag, KeyFrame *pKF,
                                        int leftIndex, int rightIndex) const
{
    if(!pTag || !pKF)
        return false;

    const int tag_id = pTag->Id();
    const int n_left = static_cast<int>(pKF->mTagFrameData.left.size());
    const int n_right = static_cast<int>(pKF->mTagFrameData.right.size());

    const tag::TagObservation *obs_left =
        (leftIndex >= 0 && leftIndex < n_left)
            ? &pKF->mTagFrameData.left[leftIndex]
            : nullptr;
    const tag::TagObservation *obs_right =
        (rightIndex >= 0 && rightIndex < n_right)
            ? &pKF->mTagFrameData.right[rightIndex]
            : nullptr;

    if(!obs_left && !obs_right)
    {
        std::cout << "[TagLM] InitializeMapTagPose fail"
                  << " tag_id=" << tag_id
                  << " kf_id=" << pKF->mnId
                  << " reason=no_valid_obs_index"
                  << " left_idx=" << leftIndex << "/" << n_left
                  << " right_idx=" << rightIndex << "/" << n_right
                  << std::endl;
        return false;
    }

    // 左目优先；失败且右目可用时回退右目
    auto trySetPoseFromObs = [&](const tag::TagObservation *obs,
                                 const char *cam_src) -> bool {
        if(!obs)
            return false;

        if(!obs->pose_estimate.has_value())
        {
            std::cout << "[TagLM] InitializeMapTagPose fail"
                      << " tag_id=" << tag_id
                      << " kf_id=" << pKF->mnId
                      << " reason=no_pose_estimate"
                      << " cam=" << cam_src
                      << " left_idx=" << leftIndex
                      << " right_idx=" << rightIndex
                      << " detect_outlier=" << (obs->is_detect_outlier ? 1 : 0)
                      << " opt_outlier=" << (obs->is_opt_outlier ? 1 : 0)
                      << std::endl;
            return false;
        }

        if(!obs->IsAmbiguityResolved())
        {
            const tag::TagPoseEstimate &est = *obs->pose_estimate;
            std::cout << "[TagLM] InitializeMapTagPose fail"
                      << " tag_id=" << tag_id
                      << " kf_id=" << pKF->mnId
                      << " reason=ambiguity_unresolved"
                      << " cam=" << cam_src
                      << " selected=" << est.selected_candidate
                      << " ambiguity_ratio=" << est.ambiguity_ratio
                      << " cand0_valid=" << (est.candidates[0].valid ? 1 : 0)
                      << " cand1_valid=" << (est.candidates[1].valid ? 1 : 0)
                      << " left_idx=" << leftIndex
                      << " right_idx=" << rightIndex
                      << std::endl;
            return false;
        }

        const tag::TagPoseCandidate *sel = obs->pose_estimate->Selected();
        if(!sel)
        {
            std::cout << "[TagLM] InitializeMapTagPose fail"
                      << " tag_id=" << tag_id
                      << " kf_id=" << pKF->mnId
                      << " reason=no_selected_candidate"
                      << " cam=" << cam_src
                      << std::endl;
            return false;
        }

        // 若为右目，则需要将 Tag 坐标系转换到左目
        const Sophus::SE3f T_cL_t = tag::ExpressTagPoseInLeftCamera(
            obs->camera_id, sel->T_ct, pKF->GetRelativePoseTlr());
        // T_wt = Twc * T_cL_t（左目锚定）
        pTag->SetPose(pKF->GetPoseInverse() * T_cL_t);
        if(const tag::TagPoseCandidate *mir = obs->pose_estimate->Unselected())
        {
            const Sophus::SE3f T_cL_m = tag::ExpressTagPoseInLeftCamera(
                obs->camera_id, mir->T_ct, pKF->GetRelativePoseTlr());
            pTag->SetMirrorPose(pKF->GetPoseInverse() * T_cL_m);
        }
        // 已消歧的单帧初值：直接 ACTIVE，可进 LBA / PoseOpt（勿停在 CANDIDATE）
        pTag->SetState(tag::MapTagState::ACTIVE);

        const Eigen::Vector3f tw = pTag->GetPose().translation();
        std::cout << "[TagLM] InitializeMapTagPose ok"
                  << " tag_id=" << tag_id
                  << " kf_id=" << pKF->mnId
                  << " cam=" << cam_src
                  << " state=ACTIVE"
                  << " selected=" << obs->pose_estimate->selected_candidate
                  << " reproj_err=" << sel->reprojection_error
                  << " t_w=[" << tw.x() << ", " << tw.y() << ", " << tw.z() << "]"
                  << std::endl;
        return true;
    };

    if(trySetPoseFromObs(obs_left, "left"))
        return true;
    if(trySetPoseFromObs(obs_right, "right"))
        return true;

    return false;
}

bool LocalMapping::TryResolveMapTagAmbiguityMultiFrame(tag::MapTagData *pTag) const
{
    constexpr float kMinBaseline = 0.07f;
    constexpr std::size_t kMinValidKFs = 7;
    constexpr double kTauRho = 2.0;
    constexpr double kErrorEps = 1e-9;

    if(!pTag || pTag->IsBad() || pTag->HasPose())
        return false;
    if(pTag->GetState() != tag::MapTagState::CANDIDATE)
        return false;

    const int tag_id = pTag->Id();
    const float tag_size = pTag->GetTagSize();
    if(tag_id < 0 || tag_size <= 0.0f)
        return false;

    const auto observations = pTag->GetObservations();
    if(observations.size() < kMinValidKFs)
        return false;

    // 按首次观测起的 KF id 排序
    std::vector<KeyFrame *> ordered_kfs;
    ordered_kfs.reserve(observations.size());
    for(const auto &kv : observations)
    {
        KeyFrame *pKF = kv.first;
        if(!pKF || pKF->isBad())
            continue;
        ordered_kfs.push_back(pKF);
    }
    std::sort(ordered_kfs.begin(), ordered_kfs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });
    if(ordered_kfs.empty())
        return false;

    // 按照 Tag ID 从 KF 中挑选可测量的观测
    auto pickMeasurableObs = [](KeyFrame *pKF, int tid) -> const tag::TagObservation * {
        const tag::TagObservation *left =
            pKF->mTagFrameData.Find(tid, tag::CameraId::LEFT_OR_MONO);
        if(left && left->IsDetectValid())
            return left;
        const tag::TagObservation *right =
            pKF->mTagFrameData.Find(tid, tag::CameraId::RIGHT);
        if(right && right->IsDetectValid())
            return right;
        return nullptr;
    };

    auto hasTwoValidIppe = [](const tag::TagObservation *obs) -> bool {
        if(!obs || !obs->pose_estimate.has_value())
            return false;
        const auto &est = *obs->pose_estimate;
        return est.candidates[0].valid && est.candidates[1].valid;
    };

    // 从首帧起，相邻有效 KF 基线 > 0.07m；且该 KF 上需有可测量观测
    std::vector<KeyFrame *> valid_kfs;
    valid_kfs.reserve(ordered_kfs.size());
    Eigen::Vector3f last_center = Eigen::Vector3f::Zero();
    bool has_last = false;
    for(KeyFrame *pKF : ordered_kfs)
    {
        if(!pickMeasurableObs(pKF, tag_id))
            continue;
        const Eigen::Vector3f center = pKF->GetCameraCenter();
        if(!has_last)
        {
            valid_kfs.push_back(pKF);
            last_center = center;
            has_last = true;
            continue;
        }
        if((center - last_center).norm() > kMinBaseline)
        {
            valid_kfs.push_back(pKF);
            last_center = center;
        }
    }

    if(valid_kfs.size() < kMinValidKFs)
    {
        std::cout << "[TagLM] multi-frame disambiguate skip"
                  << " tag_id=" << tag_id
                  << " reason=insufficient_valid_kfs"
                  << " n_valid=" << valid_kfs.size()
                  << " need>=" << kMinValidKFs
                  << std::endl;
        return false;
    }

    std::vector<cv::Point3f> object_pts;
    ua_tag::BuildSquareObjectPoints(static_cast<double>(tag_size), object_pts);

    struct IppeHyp
    {
        KeyFrame *pKF = nullptr;
        const tag::TagObservation *obs = nullptr;
        int cand_idx = -1;
    };

    // 所有有效 KF 上的全部 IPPE 解（每观测 2 个候选；同 KF 左目优先，否则右目）
    std::vector<IppeHyp> hyps;
    hyps.reserve(valid_kfs.size() * 2);
    for(KeyFrame *pKF : valid_kfs)
    {
        const tag::TagObservation *left =
            pKF->mTagFrameData.Find(tag_id, tag::CameraId::LEFT_OR_MONO);
        const tag::TagObservation *right =
            pKF->mTagFrameData.Find(tag_id, tag::CameraId::RIGHT);
        const tag::TagObservation *src = nullptr;
        if(hasTwoValidIppe(left))
            src = left;
        else if(hasTwoValidIppe(right))
            src = right;
        if(!src)
            continue;
        hyps.push_back({pKF, src, 0});
        hyps.push_back({pKF, src, 1});
    }
    if(hyps.size() < 2)
        return false;

    // 计算 Tag 世界坐标系下的四角点
    auto computeWorldCorners =
        [&](const IppeHyp &hyp,
            std::array<Eigen::Vector3f, 4> &corners_w) -> bool {
        const tag::TagPoseCandidate &cand =
            hyp.obs->pose_estimate->candidates[hyp.cand_idx];
        if(!cand.valid)
            return false;
        const Sophus::SE3f T_cL_t = tag::ExpressTagPoseInLeftCamera(
            hyp.obs->camera_id, cand.T_ct, hyp.pKF->GetRelativePoseTlr());
        const Sophus::SE3f T_wt = hyp.pKF->GetPoseInverse() * T_cL_t;
        for(int k = 0; k < 4; ++k)
        {
            const Eigen::Vector3f P_t(object_pts[k].x, object_pts[k].y,
                                     object_pts[k].z);
            corners_w[k] = T_wt * P_t;
        }
        return true;
    };


    auto recomputeWorldPose = [&](const IppeHyp &hyp, Sophus::SE3f &T_wt) -> bool {
        const tag::TagPoseCandidate &cand =
            hyp.obs->pose_estimate->candidates[hyp.cand_idx];
        if(!cand.valid)
            return false;
        const Sophus::SE3f T_cL_t = tag::ExpressTagPoseInLeftCamera(
            hyp.obs->camera_id, cand.T_ct, hyp.pKF->GetRelativePoseTlr());
        T_wt = hyp.pKF->GetPoseInverse() * T_cL_t;
        return true;
    };

    // 多帧联合重投影误差评估（这里包含右目观测评估）
    auto evalCrossReproj =
        [&](const std::array<Eigen::Vector3f, 4> &corners_w,
            KeyFrame *pExclude) -> double {
        double sum = 0.0;
        int n = 0;
        for(KeyFrame *pKF : valid_kfs)
        {
            if(pKF == pExclude || !pKF || pKF->isBad())
                continue;

            // 收集指定KF上指定Tag的观测
            const tag::TagObservation *obs = pickMeasurableObs(pKF, tag_id);
            if(!obs)
                continue;

            GeometricCamera *cam = nullptr;
            Sophus::SE3f T_cam_w;
            const Sophus::SE3f Tcw = pKF->GetPose();  // left <- world
            if(obs->camera_id == tag::CameraId::RIGHT)
            {
                if(!pKF->mpCamera2)
                    continue;
                cam = pKF->mpCamera2;
                // P_left = T_lr * P_right => P_right = T_rl * Tcw * P_w
                T_cam_w = pKF->GetRelativePoseTlr().inverse() * Tcw;
            }
            else
            {
                if(!pKF->mpCamera)
                    continue;
                cam = pKF->mpCamera;
                T_cam_w = Tcw;
            }

            double sum_sq = 0.0;
            bool ok = true;
            for(int k = 0; k < 4; ++k)
            {
                const Eigen::Vector3f P_c = T_cam_w * corners_w[k];
                if(P_c.z() <= 0.0f)
                {
                    ok = false;
                    break;
                }
                const cv::Point2f uv =
                    cam->project(cv::Point3f(P_c.x(), P_c.y(), P_c.z()));
                const double dx =
                    static_cast<double>(uv.x - obs->corners_raw[k].x);
                const double dy =
                    static_cast<double>(uv.y - obs->corners_raw[k].y);
                sum_sq += dx * dx + dy * dy;
            }
            if(!ok)
                continue;
            sum += std::sqrt(sum_sq / 4.0);
            ++n;
        }
        if(n <= 0)
            return std::numeric_limits<double>::infinity();
        return sum / static_cast<double>(n);
    };

    double best_err = std::numeric_limits<double>::infinity();
    int best_hyp_idx = -1;

    // 遍历所有候选IPPE解，选择重投影误差最小的
    for(std::size_t i = 0; i < hyps.size(); ++i)
    {
        // 计算候选IPPE解对应 Tag 世界坐标系下的四角点
        std::array<Eigen::Vector3f, 4> corners_w{};
        if(!computeWorldCorners(hyps[i], corners_w))
            continue;
        // 评估重投影误差
        const double err = evalCrossReproj(corners_w, hyps[i].pKF);
        if(err < best_err)
        {
            best_err = err;
            best_hyp_idx = static_cast<int>(i);
        }
    }

    if(best_hyp_idx < 0 || !std::isfinite(best_err))
    {
        std::cout << "[TagLM] multi-frame disambiguate fail"
                  << " tag_id=" << tag_id
                  << " reason=no_valid_hypothesis"
                  << " n_valid_kfs=" << valid_kfs.size()
                  << " n_hyps=" << hyps.size()
                  << std::endl;
        return false;
    }

    // 选择重投影误差最小的IPPE解，并计算镜像解的重投影误差，区分度需要较大
    const IppeHyp &best = hyps[best_hyp_idx];
    const int mirror_idx = 1 - best.cand_idx;
    IppeHyp mirror = best;
    mirror.cand_idx = mirror_idx;

    std::array<Eigen::Vector3f, 4> mirror_corners{};
    if(!computeWorldCorners(mirror, mirror_corners))
    {
        std::cout << "[TagLM] multi-frame disambiguate fail"
                  << " tag_id=" << tag_id
                  << " reason=mirror_invalid"
                  << " kf_id=" << best.pKF->mnId
                  << " best_cand=" << best.cand_idx
                  << std::endl;
        return false;
    }
    const double mirror_err = evalCrossReproj(mirror_corners, mirror.pKF);
    const double rho =
        (mirror_err + kErrorEps) / (best_err + kErrorEps);

    std::cout << "[TagLM] multi-frame disambiguate"
              << " tag_id=" << tag_id
              << " n_valid_kfs=" << valid_kfs.size()
              << " best_kf_id=" << best.pKF->mnId
              << " best_cand=" << best.cand_idx
              << " e_best=" << best_err
              << " e_mirror=" << mirror_err
              << " ratio=" << rho
              << " tau=" << kTauRho
              << std::endl;

    if(!(rho > kTauRho) || !std::isfinite(rho) || !std::isfinite(mirror_err))
    {
        std::cout << "[TagLM] multi-frame disambiguate reject"
                  << " tag_id=" << tag_id
                  << " reason=ratio_weak"
                  << " ratio=" << rho
                  << std::endl;
        return false;
    }

    Sophus::SE3f T_wt;
    if(!recomputeWorldPose(best, T_wt))
        return false;

    // 回写选中 IPPE 到源观测，切换 Tag 状态为 ACTIVE
    pTag->SetPose(T_wt);
    pTag->SetState(tag::MapTagState::ACTIVE);

    Sophus::SE3f T_wt_mirror;
    if(recomputeWorldPose(mirror, T_wt_mirror))
        pTag->SetMirrorPose(T_wt_mirror);

    const Sophus::SE3f T_ct_sel =
        best.obs->pose_estimate->candidates[best.cand_idx].T_ct;
    best.pKF->mTagFrameData.WriteBackSelectedCandidate(
        tag_id, best.obs->camera_id, best.cand_idx, &T_ct_sel);

    const Eigen::Vector3f tw = T_wt.translation();
    std::cout << "[TagLM] multi-frame disambiguate ok"
              << " tag_id=" << tag_id
              << " state=ACTIVE"
              << " ratio=" << rho
              << " e_best=" << best_err
              << " e_mirror=" << mirror_err
              << " t_w=[" << tw.x() << ", " << tw.y() << ", " << tw.z() << "]"
              << std::endl;
    return true;
}

void LocalMapping::ProcessTagObservations(KeyFrame *pKF)
{
    if(!pKF)
        return;

    Map *pMap = pKF->GetMap();
    if(!pMap || !pMap->IsTagInitialized())
        return;

    if(pKF->mTagFrameData.Empty())
        return;

    struct PendingObservation
    {
        int leftIndex = -1;
        int rightIndex = -1;
    };

    // 按照 Tag ID 聚合当前KF中的观测，同时进行简单的有效性检查
    std::unordered_map<int, PendingObservation> pending;

    for(int i = 0; i < static_cast<int>(pKF->mTagFrameData.left.size()); ++i)
    {
        const tag::TagObservation &obs = pKF->mTagFrameData.left[i];
        if(!obs.IsDetectValid())
            continue;
        pending[obs.tag_id].leftIndex = i;
    }
    for(int i = 0; i < static_cast<int>(pKF->mTagFrameData.right.size()); ++i)
    {
        const tag::TagObservation &obs = pKF->mTagFrameData.right[i];
        if(!obs.IsDetectValid())
            continue;
        pending[obs.tag_id].rightIndex = i;
    }

    if(pending.empty())
        return;

    bool tag_map_updated = false;

    for(const auto &kv : pending)
    {
        const int tag_id = kv.first;
        const int left_idx = kv.second.leftIndex;
        const int right_idx = kv.second.rightIndex;

        auto pTag = pMap->GetMapTag(tag_id);
        bool created = false;
        const bool had_pose_before = pTag && pTag->HasPose();

        if(!pTag)
        {
            // 按 tag_id 取配置边长；无配置时回退 0.16
            float tag_size = 0.16f;
            if(mpTracker)
                tag_size = static_cast<float>(mpTracker->GetConfiguredTagSize(tag_id));

            pTag = std::make_shared<tag::MapTagData>();
            pTag->SetId(tag_id);
            pTag->SetTagSize(tag_size);
            pTag->SetState(tag::MapTagState::CANDIDATE);
            InitializeMapTagPose(pTag.get(), pKF, left_idx, right_idx);

            if(!pMap->AddMapTag(pTag))
            {
                // 同 ID 已存在（并发/竞态）：改用地图中的对象
                pTag = pMap->GetMapTag(tag_id);
            }
            else
            {
                created = true;
                tag_map_updated = true;
                const char *st_name =
                    pTag->GetState() == tag::MapTagState::ACTIVE ? "ACTIVE"
                    : pTag->GetState() == tag::MapTagState::FIXED_ANCHOR
                          ? "FIXED_ANCHOR"
                          : "CANDIDATE";
                std::cout << "[TagLM] created MapTag id=" << tag_id
                          << " as " << st_name
                          << " from kf_id=" << pKF->mnId
                          << " has_pose=" << (pTag->HasPose() ? 1 : 0)
                          << std::endl;
            }
        }

        if(!pTag)
            continue;

        if(!pKF->AddMapTag(pTag.get(), left_idx, right_idx))
        {
            if(created && pTag->Observations() == 0)
                pMap->EraseMapTag(tag_id);
            continue;
        }

        // 首次成功注册进地图：记录观测面积 / 距离 / 光轴夹角
        if(created && mpTracker)
            mpTracker->LogMapTagFirstRegistration(pKF, tag_id, left_idx,
                                                  right_idx, pTag.get());

        // CANDIDATE：无 pose → 单帧消歧初值（成功即 ACTIVE）或多帧消歧；
        // 已有 pose 却仍为 CANDIDATE（旧逻辑残留）→ 直接晋升 ACTIVE
        if(pTag->GetState() == tag::MapTagState::CANDIDATE)
        {
            if(!pTag->HasPose())
            {
                if(!created)
                    InitializeMapTagPose(pTag.get(), pKF, left_idx, right_idx);
                if(!pTag->HasPose())
                    TryResolveMapTagAmbiguityMultiFrame(pTag.get());
            }
            else if(!pTag->IsFixed())
            {
                pTag->SetState(tag::MapTagState::ACTIVE);
                std::cout << "[TagLM] promote CANDIDATE→ACTIVE"
                          << " tag_id=" << tag_id
                          << " kf_id=" << pKF->mnId
                          << " reason=has_unambiguous_pose"
                          << std::endl;
            }
        }

        if(pTag->HasPose() && (!had_pose_before || created))
            tag_map_updated = true;
    }

    // MapTag 新增或首次获得 pose：在线覆盖写入 tag_map_corners.csv
    if(tag_map_updated && mpTracker)
        mpTracker->SaveTagMapOnline();
}

void LocalMapping::EmptyQueue()
{
    assert(!HasPendingKeyFrames() && "LocalMapping queue must be empty in offline sync");
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size();

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;

        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
        {
            lit++;
            borrar--;
        }
    }
}


void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // For stereo inertial case
    if(mbMonocular)
        nn=30;
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (mbInertial)
    {
        KeyFrame* pKF = mpCurrentKeyFrame;
        int count=0;
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
        {
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            if(it==vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF;
        }
    }

    float th = 0.6f;

    ORBmatcher matcher(th,false);

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        // 删除return提前结束三角化的部分
        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (mpCurrentKeyFrame->NLeft == -1 && kp1_ur>=0);
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;

            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];

            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (pKF2->NLeft == -1 && kp2_ur>=0);
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            // 鱼眼双目左右特征分桶；PinHole+Tag 的 mpCamera2 不能当作鱼眼路径
            if(mpCurrentKeyFrame->NLeft != -1 && pKF2->NLeft != -1){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);

            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            if (bStereo1 || bStereo2) totalStereoPts++;
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<0.9996 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
            {
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo)
                countStereoGoodProj++;

            if(!goodProj)
                continue;

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            if (bPointStereo)
                countStereo++;
            
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }    
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=30;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<20 && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    mbStopped = false;
    mbStopRequested = false;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular)
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }



    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mbInertial)
            {
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            else
            {
                pKF->SetBadFlag();
            }
        }
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

void LocalMapping::ResetSynchronously()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    ResetIfRequested();
}

void LocalMapping::ResetActiveMapSynchronously(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    ResetIfRequested();
}

void LocalMapping::RequestReset()
{
    ResetSynchronously();
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    ResetActiveMapSynchronously(pMap);
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested)
        return;

    float minTime;
    int nMinKF;
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }


    if(mpAtlas->KeyFramesInMap()<nMinKF)
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    if(vpKF.size()<nMinKF)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime)
        return;

    bInitializing = true;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        dirG = dirG/dirG.norm();
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        Eigen::Vector3f vzg = v*ang/nv;
        Rwg = Sophus::SO3f::exp(vzg).matrix();
        mRwg = Rwg.cast<double>();
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        if (!mpAtlas->isImuInitialized())
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
    }

    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    if (!mpAtlas->isImuInitialized())
    {
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end());

    while(!lpKFtoCheck.empty())
    {
        KeyFrame* pKF = lpKFtoCheck.front();
        const set<KeyFrame*> sChilds = pKF->GetChilds();
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
        {
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad())
                continue;

            if(pChild->mnBAGlobalForKF!=GBAid)
            {
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;

                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias();
                pChild->mnBAGlobalForKF = GBAid;

            }
            lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);

        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            cout << "KF " << pKF->mnId << " not set to inertial!! \n";
        }

        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            if(!pRefKF)
                continue;

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size();
    mIdxInit++;

    // 不要 delete 队列里的 KF：Tracking 可能仍把它们当作 mpLastKeyFrame
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    mpTracker->mState=Tracking::OK;
    bInitializing = false;

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    
    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.002)||!mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
