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


#include "LoopClosing.h"

#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "G2oTypes.h"
#include "Map.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagObservation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace ORB_SLAM3
{

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale, const bool bActiveLC):
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mbFixScale(bFixScale), mnLoopNumCoincidences(0), mnMergeNumCoincidences(0),
    mbLoopDetected(false), mbMergeDetected(false), mnLoopNumNotFound(0), mnMergeNumNotFound(0), mbActiveLC(bActiveLC)
{
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);

#ifdef REGISTER_TIMES

    vdDataQuery_ms.clear();
    vdEstSim3_ms.clear();
    vdPRTotal_ms.clear();

    vdMergeMaps_ms.clear();
    vdWeldingBA_ms.clear();
    vdMergeOptEss_ms.clear();
    vdMergeTotal_ms.clear();
    vnMergeKFs.clear();
    vnMergeMPs.clear();
    nMerges = 0;

    vdLoopFusion_ms.clear();
    vdLoopOptEss_ms.clear();
    vdLoopTotal_ms.clear();
    vnLoopKFs.clear();
    nLoop = 0;

    vdGBA_ms.clear();
    vdUpdateMap_ms.clear();
    vdFGBATotal_ms.clear();
    vnGBAKFs.clear();
    vnGBAMPs.clear();
    nFGBA_exec = 0;
    nFGBA_abort = 0;

#endif

    mstrFolderSubTraj = "SubTrajectories/";
    mnNumCorrection = 0;
    mnCorrectionGBA = 0;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void LoopClosing::SetTagLoopParams(const tag::TagLoopParams &params)
{
    mTagLoopParams = params;
    if(mTagLoopParams.enable)
    {
        std::cout << "Tag Loop enabled"
                  << " (min_history_obs=" << mTagLoopParams.min_history_observations
                  << ", min_kf_gap=" << mTagLoopParams.min_keyframe_gap
                  << ", min_reobs_gap=" << mTagLoopParams.min_reobservation_gap_kfs
                  << ", min_hist_mps=" << mTagLoopParams.min_historical_kf_map_points
                  << ", min_consistent_tags=" << mTagLoopParams.min_consistent_tags
                  << ", ref_timeout_kfs=" << mTagLoopParams.reference_timeout_kfs
                  << ")" << std::endl;
    }
}

void LoopClosing::SetTagLocalBAParams(const tag::TagLocalBAParams &params)
{
    mTagLocalBAParams = params;
}

void LoopClosing::ResetTagLoopState()
{
    if(mpTagLoopMatchedKF && mpTagLoopMatchedKF != mpCurrentKF)
        mpTagLoopMatchedKF->SetErase();
    mbTagLoopDetected = false;
    mnLoopTagId = -1;
    mpLoopTag = nullptr;
    mpTagLoopMatchedKF = nullptr;
    mnTagLoopCoincidences = 0;
    mnTagLoopNotFound = 0;
    mpTagLoopLastCurrentKF = nullptr;
}

void LoopClosing::ReleaseTagLoopHistoricalKF(KeyFrame *pKF)
{
    if(!pKF || pKF == mpCurrentKF)
        return;
    if(pKF == mpTagLoopMatchedKF || pKF == mpLoopMatchedKF)
        return;
    for(const auto &kv : mTagLoopReferences)
    {
        if(kv.second.pHistoricalKF == pKF)
            return;
    }
    pKF->SetErase();
}

void LoopClosing::EraseTagLoopReference(int tagId, const char *reason)
{
    const auto it = mTagLoopReferences.find(tagId);
    if(it == mTagLoopReferences.end())
        return;
    KeyFrame *pKF = it->second.pHistoricalKF;
    const unsigned long lastSeen = it->second.lastSeenKF;
    mTagLoopReferences.erase(it);
    std::cout << "[TagLoopRef] KF=" << (mpCurrentKF ? mpCurrentKF->mnId : 0)
              << " tag=" << tagId
              << " action=erase reason=" << (reason ? reason : "unknown")
              << " lastSeen=" << lastSeen
              << " histKF=" << (pKF ? pKF->mnId : 0)
              << std::endl;
    ReleaseTagLoopHistoricalKF(pKF);
}

void LoopClosing::ClearTagLoopReferences(const char *reason)
{
    if(mTagLoopReferences.empty())
        return;
    std::cout << "[TagLoopRef] KF=" << (mpCurrentKF ? mpCurrentKF->mnId : 0)
              << " action=clear_all reason=" << (reason ? reason : "unknown")
              << " n=" << mTagLoopReferences.size() << std::endl;
    std::vector<KeyFrame*> kfs;
    kfs.reserve(mTagLoopReferences.size());
    for(const auto &kv : mTagLoopReferences)
    {
        if(kv.second.pHistoricalKF)
            kfs.push_back(kv.second.pHistoricalKF);
    }
    mTagLoopReferences.clear();
    for(KeyFrame *pKF : kfs)
        ReleaseTagLoopHistoricalKF(pKF);
}

void LoopClosing::PruneTagLoopReferences()
{
    if(!mpCurrentKF)
        return;
    const unsigned long curId = mpCurrentKF->mnId;
    const unsigned long timeout =
        static_cast<unsigned long>(std::max(1, mTagLoopParams.reference_timeout_kfs));
    std::vector<int> expired;
    for(const auto &kv : mTagLoopReferences)
    {
        const TagLoopReference &ref = kv.second;
        if(!ref.pHistoricalKF || ref.pHistoricalKF->isBad() ||
           ref.pHistoricalKF->GetMap() != mpCurrentKF->GetMap() ||
           curId >= ref.lastSeenKF + timeout)
            expired.push_back(kv.first);
    }
    for(int tagId : expired)
    {
        const auto it = mTagLoopReferences.find(tagId);
        const char *reason = "timeout";
        if(it != mTagLoopReferences.end() &&
           (!it->second.pHistoricalKF || it->second.pHistoricalKF->isBad() ||
            it->second.pHistoricalKF->GetMap() != mpCurrentKF->GetMap()))
            reason = "hist_kf_invalid";
        EraseTagLoopReference(tagId, reason);
    }
}


void LoopClosing::Run()
{
    // Offline architecture: processing is driven by ProcessUntilIdle() from Track*().
    mbFinished = true;
}

bool LoopClosing::HasPendingKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

void LoopClosing::ProcessUntilIdle()
{
    while(ProcessOneKeyFrame())
    {
    }
    ResetIfRequested();
}

bool LoopClosing::ProcessOneKeyFrame()
{
    if(!CheckNewKeyFrames())
        return false;

    if(mpLastCurrentKF)
    {
        mpLastCurrentKF->mvpLoopCandKFs.clear();
        mpLastCurrentKF->mvpMergeCandKFs.clear();
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartPR = std::chrono::steady_clock::now();
#endif

    bool bFindedRegion = NewDetectCommonRegions();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndPR = std::chrono::steady_clock::now();

    double timePRTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPR - time_StartPR).count();
    vdPRTotal_ms.push_back(timePRTotal);
#endif

    std::cout << "[KF " << mpCurrentKF->mnId << "] LoopClosing begin" << std::endl;

    if(bFindedRegion)
    {
        if(mbMergeDetected)
        {
            if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                (!mpCurrentKF->GetMap()->isImuInitialized()))
            {
                cout << "IMU is not initilized, merge is aborted" << endl;
            }
            else
            {
                Sophus::SE3d mTmw = mpMergeMatchedKF->GetPose().cast<double>();
                g2o::Sim3 gSmw2(mTmw.unit_quaternion(), mTmw.translation(), 1.0);
                Sophus::SE3d mTcw = mpCurrentKF->GetPose().cast<double>();
                g2o::Sim3 gScw1(mTcw.unit_quaternion(), mTcw.translation(), 1.0);
                g2o::Sim3 gSw2c = mg2oMergeSlw.inverse();
                g2o::Sim3 gSw1m = mg2oMergeSlw;

                mSold_new = (gSw2c * gScw1);


                if(mpCurrentKF->GetMap()->IsInertial() && mpMergeMatchedKF->GetMap()->IsInertial())
                {
                    cout << "Merge check transformation with IMU" << endl;
                    if(mSold_new.scale()<0.90||mSold_new.scale()>1.1){
                        mpMergeLastCurrentKF->SetErase();
                        mpMergeMatchedKF->SetErase();
                        mpMergeLastCurrentKF = nullptr;
                        mpMergeMatchedKF = nullptr;
                        mnMergeNumCoincidences = 0;
                        mvpMergeMatchedMPs.clear();
                        mvpMergeMPs.clear();
                        mnMergeNumNotFound = 0;
                        mbMergeDetected = false;
                        Verbose::PrintMess("scale bad estimated. Abort merging", Verbose::VERBOSITY_NORMAL);
                        std::cout << "[KF " << mpCurrentKF->mnId << "] No loop" << std::endl;
                        return true;
                    }
                    if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                           mpCurrentKF->GetMap()->GetIniertialBA1())
                    {
                        Eigen::Vector3d phi = LogSO3(mSold_new.rotation().toRotationMatrix());
                        phi(0)=0;
                        phi(1)=0;
                        mSold_new = g2o::Sim3(ExpSO3(phi),mSold_new.translation(),1.0);
                    }
                }

                mg2oMergeSmw = gSmw2 * gSw2c * gScw1;

                mg2oMergeScw = mg2oMergeSlw;

                Verbose::PrintMess("*Merge detected", Verbose::VERBOSITY_QUIET);
                std::cout << "[KF " << mpCurrentKF->mnId << "] Merge confirmed" << std::endl;

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();

                nMerges += 1;
#endif
                if (mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
                    MergeLocal2();
                else
                    MergeLocal();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndMerge = std::chrono::steady_clock::now();

                double timeMergeTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMerge - time_StartMerge).count();
                vdMergeTotal_ms.push_back(timeMergeTotal);
#endif

                Verbose::PrintMess("Merge finished!", Verbose::VERBOSITY_QUIET);
            }

            vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
            vdPR_MatchedTime.push_back(mpMergeMatchedKF->mTimeStamp);
            vnPR_TypeRecogn.push_back(1);

            mpMergeLastCurrentKF->SetErase();
            mpMergeMatchedKF->SetErase();
            mpMergeLastCurrentKF = nullptr;
            mpMergeMatchedKF = nullptr;
            mnMergeNumCoincidences = 0;
            mvpMergeMatchedMPs.clear();
            mvpMergeMPs.clear();
            mnMergeNumNotFound = 0;
            mbMergeDetected = false;
            ClearTagLoopReferences("merge");

            if(mbLoopDetected)
            {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mpLoopLastCurrentKF = nullptr;
                mpLoopMatchedKF = nullptr;
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
                mbLoopDetected = false;
                ResetTagLoopState();
            }

        }

        if(mbLoopDetected)
        {
            bool bGoodLoop = true;
            vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
            vdPR_MatchedTime.push_back(mpLoopMatchedKF->mTimeStamp);
            vnPR_TypeRecogn.push_back(0);

            Verbose::PrintMess("*Loop detected", Verbose::VERBOSITY_QUIET);

            mg2oLoopScw = mg2oLoopSlw;
            // Tag使用不同的门控阈值
            if(!mbTagLoopDetected && mpCurrentKF->GetMap()->IsInertial())
            {
                Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
                g2o::Sim3 g2oTwc(Twc.unit_quaternion(),Twc.translation(),1.0);
                g2o::Sim3 g2oSww_new = g2oTwc*mg2oLoopScw;

                Eigen::Vector3d phi = LogSO3(g2oSww_new.rotation().toRotationMatrix());
                cout << "phi = " << phi.transpose() << endl;
                if (fabs(phi(0))<0.008f && fabs(phi(1))<0.008f && fabs(phi(2))<0.349f)
                {
                    if(mpCurrentKF->GetMap()->IsInertial())
                    {
                        if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                                mpCurrentKF->GetMap()->GetIniertialBA2())
                        {
                            phi(0)=0;
                            phi(1)=0;
                            g2oSww_new = g2o::Sim3(ExpSO3(phi),g2oSww_new.translation(),1.0);
                            mg2oLoopScw = g2oTwc.inverse()*g2oSww_new;
                        }
                    }

                }
                else
                {
                    cout << "BAD LOOP!!!" << endl;
                    bGoodLoop = false;
                }

            }

            if (bGoodLoop) {

                mvpLoopMapPoints = mvpLoopMPs;

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_StartLoop = std::chrono::steady_clock::now();

                nLoop += 1;

#endif
                CorrectLoop();
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLoop = std::chrono::steady_clock::now();

                double timeLoopTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLoop - time_StartLoop).count();
                vdLoopTotal_ms.push_back(timeLoopTotal);
#endif

                mnNumCorrection += 1;
            }
            else
            {
                std::cout << "[KF " << mpCurrentKF->mnId << "] No loop" << std::endl;
            }

            mpLoopLastCurrentKF->SetErase();
            mpLoopMatchedKF->SetErase();
            mpLoopLastCurrentKF = nullptr;
            mpLoopMatchedKF = nullptr;
            mnLoopNumCoincidences = 0;
            mvpLoopMatchedMPs.clear();
            mvpLoopMPs.clear();
            mnLoopNumNotFound = 0;
            mbLoopDetected = false;
            ResetTagLoopState();
        }

    }
    else
    {
        std::cout << "[KF " << mpCurrentKF->mnId << "] No loop" << std::endl;
    }
    mpLastCurrentKF = mpCurrentKF;
    return true;
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

namespace {

void TagLoopPoseDelta(const Sophus::SE3f &TcwA, const Sophus::SE3f &TcwB,
                      float &dtrans, float &drot)
{
    const Sophus::SE3f dT = TcwA * TcwB.inverse();
    dtrans = dT.translation().norm();
    drot = dT.so3().log().norm();
}

bool TagLoopPosesConsistent(const Sophus::SE3f &TcwA, const Sophus::SE3f &TcwB,
                            float maxTrans, float maxRot)
{
    float dtrans = 0.0f;
    float drot = 0.0f;
    TagLoopPoseDelta(TcwA, TcwB, dtrans, drot);
    return dtrans < maxTrans && drot < maxRot;
}

void PrintSE3Compact(const Sophus::SE3f &T)
{
    const Eigen::Vector3f t = T.translation();
    const Eigen::Quaternionf q = T.unit_quaternion();
    std::cout << "t=[" << t.x() << " " << t.y() << " " << t.z() << "]"
              << " q=[" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "]";
}

void PrintTagLoopPair(unsigned long kfId, int tagA, int tagB,
                      const Sophus::SE3f &TcwA, const Sophus::SE3f &TcwB,
                      float maxTrans, float maxRot, const char *src)
{
    float dtrans = 0.0f;
    float drot = 0.0f;
    TagLoopPoseDelta(TcwA, TcwB, dtrans, drot);
    const int over_trans = dtrans >= maxTrans ? 1 : 0;
    const int over_rot = drot >= maxRot ? 1 : 0;
    std::cout << "[TagLoopPair] KF=" << kfId
              << " tagA=" << tagA << " tagB=" << tagB
              << " dtrans=" << dtrans << " m"
              << " drot=" << drot << " rad"
              << " consistent=" << ((over_trans || over_rot) ? 0 : 1)
              << " over_trans=" << over_trans
              << " over_rot=" << over_rot
              << " src=" << src
              << " TcwA ";
    PrintSE3Compact(TcwA);
    std::cout << " TcwB ";
    PrintSE3Compact(TcwB);
    std::cout << std::endl;
}

bool TagObservationUsable(const tag::TagObservation &obs)
{
    return obs.IsDetectValid();
}

bool TagObservationAtIndexUsable(const std::vector<tag::TagObservation> &observations, int idx)
{
    return idx >= 0 && idx < static_cast<int>(observations.size()) &&
           TagObservationUsable(observations[idx]);
}

bool HistoricalTagObservationUsable(const KeyFrame *pKF,
                                    const tag::MapTagData::KeyFrameObservation &obs)
{
    if(!pKF)
        return false;
    return TagObservationAtIndexUsable(pKF->mTagFrameData.left, obs.leftIndex) ||
           TagObservationAtIndexUsable(pKF->mTagFrameData.right, obs.rightIndex);
}

bool CurrentTagObservationUsable(const KeyFrame *pKF,
                                 const KeyFrame::MapTagAssociation &assoc)
{
    if(!pKF)
        return false;
    return TagObservationAtIndexUsable(pKF->mTagFrameData.left, assoc.leftObservationIndex) ||
           TagObservationAtIndexUsable(pKF->mTagFrameData.right, assoc.rightObservationIndex);
}

float SelectedReprojectionError(const tag::TagObservation *obs)
{
    if(!obs || !obs->IsAmbiguityResolved() || !obs->pose_estimate)
        return -1.0f;
    const tag::TagPoseCandidate *sel = obs->pose_estimate->Selected();
    if(!sel)
        return -1.0f;
    return sel->reprojection_error;
}

float CurrentTagReprojectionError(const KeyFrame *pKF,
                                  const KeyFrame::MapTagAssociation &assoc)
{
    if(!pKF)
        return 1e9f;
    float eL = -1.0f;
    float eR = -1.0f;
    if(assoc.leftObservationIndex >= 0 &&
       assoc.leftObservationIndex < static_cast<int>(pKF->mTagFrameData.left.size()))
        eL = SelectedReprojectionError(&pKF->mTagFrameData.left[assoc.leftObservationIndex]);
    if(assoc.rightObservationIndex >= 0 &&
       assoc.rightObservationIndex < static_cast<int>(pKF->mTagFrameData.right.size()))
        eR = SelectedReprojectionError(&pKF->mTagFrameData.right[assoc.rightObservationIndex]);
    if(eL >= 0.0f && eR >= 0.0f)
        return 0.5f * (eL + eR);
    if(eL >= 0.0f)
        return eL;
    if(eR >= 0.0f)
        return eR;
    return 1e9f;
}

std::string TagIppeFailReason(const tag::TagObservation *obs, int idx, float maxReproj)
{
    if(idx < 0)
        return "no_idx";
    if(!obs)
        return "no_obs";
    if(!obs->IsDetectValid())
        return "detect_invalid";
    if(!obs->pose_estimate)
        return "no_ippe";
    if(!obs->IsAmbiguityResolved())
        return "unresolved";
    const tag::TagPoseCandidate *sel = obs->pose_estimate->Selected();
    if(!sel)
        return "no_selected";
    if(sel->reprojection_error < 0.0f || sel->reprojection_error > maxReproj)
        return "reproj";
    return "ok";
}

struct TagIppeCamDiag
{
    int idx = -1;
    std::string status = "no_idx";
    int selected = -1;
    float reproj = -1.0f;
    float cand0_e = -1.0f;
    float cand1_e = -1.0f;
    int cand0_v = 0;
    int cand1_v = 0;
    float amb_ratio = -1.0f;
    bool usable = false;
    bool hasTcw = false;
    Sophus::SE3f Tcw;
    Sophus::SE3f Tct;
};

void FillTagIppeCamDiag(TagIppeCamDiag &d, const tag::TagObservation *obs, int idx,
                        float maxReproj, const Sophus::SE3f &Tlr,
                        const Sophus::SE3f &historicalTwt)
{
    d.idx = idx;
    d.status = TagIppeFailReason(obs, idx, maxReproj);
    if(!obs || !obs->pose_estimate)
        return;

    const tag::TagPoseEstimate &est = *obs->pose_estimate;
    d.selected = est.selected_candidate;
    d.amb_ratio = est.ambiguity_ratio;
    d.cand0_e = est.candidates[0].reprojection_error;
    d.cand1_e = est.candidates[1].reprojection_error;
    d.cand0_v = est.candidates[0].valid ? 1 : 0;
    d.cand1_v = est.candidates[1].valid ? 1 : 0;

    const tag::TagPoseCandidate *sel = est.Selected();
    if(!sel && d.selected >= 0 && d.selected < 2 &&
       est.candidates[static_cast<size_t>(d.selected)].valid)
        sel = &est.candidates[static_cast<size_t>(d.selected)];
    if(sel)
        d.reproj = sel->reprojection_error;

    // 只要有已选定的 IPPE 解，就算出 Tcw 供诊断；是否用于回环仍看 status==ok。
    if(!est.Selected())
        return;
    d.Tct = tag::ExpressTagPoseInLeftCamera(obs->camera_id, est.Selected()->T_ct, Tlr);
    d.Tcw = d.Tct * historicalTwt.inverse();
    d.hasTcw = true;
    d.usable = (d.status == "ok");
}

void PrintTagIppeCam(unsigned long kfId, int tagId, char cam, const TagIppeCamDiag &d,
                     float maxReproj)
{
    std::cout << "[TagLoopIppe] KF=" << kfId
              << " tag=" << tagId
              << " cam=" << cam
              << " idx=" << d.idx
              << " status=" << d.status
              << " sel=" << d.selected
              << " reproj=" << d.reproj
              << " thr=" << maxReproj
              << " cand0_e=" << d.cand0_e << " cand0_v=" << d.cand0_v
              << " cand1_e=" << d.cand1_e << " cand1_v=" << d.cand1_v
              << " amb_ratio=" << d.amb_ratio
              << " usable=" << (d.usable ? 1 : 0);
    if(d.hasTcw)
    {
        std::cout << " Tcw ";
        PrintSE3Compact(d.Tcw);
        std::cout << " Tct ";
        PrintSE3Compact(d.Tct);
    }
    std::cout << std::endl;
}

void PrintTagIppeStereo(unsigned long kfId, int tagId,
                        const TagIppeCamDiag &L, const TagIppeCamDiag &R,
                        float maxT, float maxR,
                        bool success, const char *resultReason,
                        const char *chosen, const Sophus::SE3f *TcwTag)
{
    std::cout << "[TagLoopIppe] KF=" << kfId
              << " tag=" << tagId
              << " stereo"
              << " hasL=" << (L.usable ? 1 : 0)
              << " hasR=" << (R.usable ? 1 : 0)
              << " L=" << L.status
              << " R=" << R.status
              << " selL=" << L.selected
              << " selR=" << R.selected
              << " reprojL=" << L.reproj
              << " reprojR=" << R.reproj;

    const bool bothTcw = L.hasTcw && R.hasTcw;
    float dtrans = -1.0f;
    float drot = -1.0f;
    int over_trans = 0;
    int over_rot = 0;
    int consistent = -1;
    if(bothTcw)
    {
        TagLoopPoseDelta(L.Tcw, R.Tcw, dtrans, drot);
        over_trans = dtrans >= maxT ? 1 : 0;
        over_rot = drot >= maxR ? 1 : 0;
        consistent = (over_trans || over_rot) ? 0 : 1;
        std::cout << " dtrans=" << dtrans << " m"
                  << " drot=" << drot << " rad"
                  << " maxT=" << maxT << " m"
                  << " maxR=" << maxR << " rad"
                  << " consistent=" << consistent
                  << " over_trans=" << over_trans
                  << " over_rot=" << over_rot
                  << " gate=" << ((L.usable && R.usable) ? 1 : 0);
        std::cout << " TcwL ";
        PrintSE3Compact(L.Tcw);
        std::cout << " TcwR ";
        PrintSE3Compact(R.Tcw);
    }
    else
    {
        std::cout << " dtrans=n/a drot=n/a consistent=n/a"
                  << " other_fail=" << (L.usable ? R.status : (R.usable ? L.status : "both"));
    }

    std::cout << " result=" << (success ? "ok" : "fail")
              << " reason=" << resultReason
              << " chosen=" << chosen;
    if(TcwTag)
    {
        std::cout << " TcwTag ";
        PrintSE3Compact(*TcwTag);
    }
    std::cout << std::endl;
}

}  // namespace

bool LoopClosing::EstimateTagLoopPose(const KeyFrame::MapTagAssociation &assoc,
                                      const Sophus::SE3f &historicalTwt,
                                      Sophus::SE3f &TcwTag, Sophus::SE3f &TctLeft,
                                      bool &bStereoConsistent, int &nValidCameras,
                                      std::string *failReason)
{
    bStereoConsistent = false;
    nValidCameras = 0;
    auto setFail = [&](const char *msg) {
        if(failReason)
            *failReason = msg;
        return false;
    };
    if(!assoc.pMapTag)
        return setFail("no_historical_pose");

    const int tagId = assoc.pMapTag->Id();
    const float maxReproj = mTagLoopParams.max_ippe_reprojection_error;
    const float maxT = mTagLoopParams.max_stereo_translation_error;
    const float maxR = mTagLoopParams.max_stereo_rotation_error;
    const Sophus::SE3f Tlr = mpCurrentKF->GetRelativePoseTlr();

    const tag::TagObservation *obsL = nullptr;
    const tag::TagObservation *obsR = nullptr;
    if(assoc.leftObservationIndex >= 0 &&
       assoc.leftObservationIndex < static_cast<int>(mpCurrentKF->mTagFrameData.left.size()))
        obsL = &mpCurrentKF->mTagFrameData.left[assoc.leftObservationIndex];
    if(assoc.rightObservationIndex >= 0 &&
       assoc.rightObservationIndex < static_cast<int>(mpCurrentKF->mTagFrameData.right.size()))
        obsR = &mpCurrentKF->mTagFrameData.right[assoc.rightObservationIndex];

    TagIppeCamDiag L, R;
    FillTagIppeCamDiag(L, obsL, assoc.leftObservationIndex, maxReproj, Tlr, historicalTwt);
    FillTagIppeCamDiag(R, obsR, assoc.rightObservationIndex, maxReproj, Tlr, historicalTwt);
    PrintTagIppeCam(mpCurrentKF->mnId, tagId, 'L', L, maxReproj);
    PrintTagIppeCam(mpCurrentKF->mnId, tagId, 'R', R, maxReproj);

    auto dumpStereo = [&](bool success, const char *reason, const char *chosen,
                          const Sophus::SE3f *pTcw) {
        PrintTagIppeStereo(mpCurrentKF->mnId, tagId, L, R, maxT, maxR,
                           success, reason, chosen, pTcw);
    };

    // 检查双目Tag计算当前T_wc相机pose是否一致
    if(L.usable && R.usable)
    {
        float dtrans = 0.0f;
        float drot = 0.0f;
        TagLoopPoseDelta(L.Tcw, R.Tcw, dtrans, drot);
        const int over_trans = dtrans >= maxT ? 1 : 0;
        const int over_rot = drot >= maxR ? 1 : 0;
        if(over_trans || over_rot)
        {
            std::ostringstream oss;
            oss << "stereo_inconsistent dtrans=" << dtrans << " m drot=" << drot
                << " rad maxT=" << maxT << " maxR=" << maxR
                << " over_trans=" << over_trans << " over_rot=" << over_rot;
            dumpStereo(false, oss.str().c_str(), "none", nullptr);
            return setFail(oss.str().c_str());
        }
        TcwTag = L.Tcw;
        TctLeft = L.Tct;
        bStereoConsistent = true;
        nValidCameras = 2;
        dumpStereo(true, "stereo_ok", "L", &TcwTag);
        return true;
    }
    if(L.usable)
    {
        TcwTag = L.Tcw;
        TctLeft = L.Tct;
        nValidCameras = 1;
        dumpStereo(true, "mono_L", "L", &TcwTag);
        return true;
    }
    if(R.usable)
    {
        TcwTag = R.Tcw;
        TctLeft = R.Tct;
        nValidCameras = 1;
        dumpStereo(true, "mono_R", "R", &TcwTag);
        return true;
    }

    std::ostringstream oss;
    oss << "no_valid_ippe L=" << L.status
        << " selL=" << L.selected << " reprojL=" << L.reproj
        << " cand0L=" << L.cand0_e << " cand1L=" << L.cand1_e
        << " R=" << R.status
        << " selR=" << R.selected << " reprojR=" << R.reproj
        << " cand0R=" << R.cand0_e << " cand1R=" << R.cand1_e;
    dumpStereo(false, oss.str().c_str(), "none", nullptr);
    return setFail(oss.str().c_str());
}

bool LoopClosing::ValidateTagLoop(const g2o::Sim3 &gScwTag, g2o::Sim3 &gScwTag4DoF)
{
    gScwTag4DoF = gScwTag;
    Map *pMap = mpCurrentKF->GetMap();
    if(!pMap || !pMap->IsInertial())
        return true;

    const Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
    const g2o::Sim3 gTwcOrb(Twc.unit_quaternion(), Twc.translation(), 1.0);
    g2o::Sim3 gSwwCorrection = gTwcOrb * gScwTag;
    Eigen::Vector3d phi = LogSO3(gSwwCorrection.rotation().toRotationMatrix());

    if(std::abs(phi(0)) > mTagLoopParams.max_roll_pitch_correction ||
       std::abs(phi(1)) > mTagLoopParams.max_roll_pitch_correction ||
       std::abs(phi(2)) > mTagLoopParams.max_yaw_correction)
    {
        std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
                  << " reject 4DoF phi=" << phi.transpose() << std::endl;
        return false;
    }

    if(pMap->GetIniertialBA2())
    {
        phi(0) = 0.0;
        phi(1) = 0.0;
        gSwwCorrection = g2o::Sim3(ExpSO3(phi), gSwwCorrection.translation(), 1.0);
        gScwTag4DoF = gTwcOrb.inverse() * gSwwCorrection;
    }
    return true;
}

void LoopClosing::CollectTagLoopMapPoints(KeyFrame *pMatchedKF, std::vector<MapPoint*> &vpMPs)
{
    vpMPs.clear();
    if(!pMatchedKF)
        return;

    std::unordered_set<MapPoint*> spMPs;
    auto addKF = [&](KeyFrame *pKF) {
        if(!pKF || pKF->isBad())
            return;
        const std::vector<MapPoint*> matches = pKF->GetMapPointMatches();
        for(MapPoint *pMP : matches)
        {
            if(pMP && !pMP->isBad())
                spMPs.insert(pMP);
        }
    };

    addKF(pMatchedKF);
    const std::vector<KeyFrame*> vpCov = pMatchedKF->GetBestCovisibilityKeyFrames(10);
    for(KeyFrame *pKF : vpCov)
        addKF(pKF);

    vpMPs.assign(spMPs.begin(), spMPs.end());
}

std::vector<tag::TagPoseConstraint> LoopClosing::BuildTagLoopPoseConstraints(KeyFrame *pKF)
{
    // 回环专用快照：角点来自冻结/历史 T_wt，避免 BuildTagMapSnapshot 读到 LBA 后的实时 MapTag 位姿。
    std::vector<tag::TagPoseConstraint> out;
    if(!pKF)
        return out;

    const auto associations = pKF->GetMapTagAssociations();
    out.reserve(associations.size());

    for(const auto &kv : associations)
    {
        const KeyFrame::MapTagAssociation &assoc = kv.second;
        tag::MapTagData *pTag = assoc.pMapTag;
        if(!pTag || pTag->IsBad())
            continue;

        const tag::MapTagState state = pTag->GetState();
        if(state != tag::MapTagState::ACTIVE && state != tag::MapTagState::FIXED_ANCHOR)
            continue;
        // 左右目至少有一个有效二维角点即可，不要求 IPPE / reobservation_gap
        if(!CurrentTagObservationUsable(pKF, assoc))
            continue;
        if(!assoc.hasHistoricalTagPose)
            continue;

        const float tagSize = pTag->GetTagSize();
        if(tagSize <= 0.0f)
            continue;

        // 已冻结的回环参考优先；否则用关联时拷下的 historicalTagPose
        Sophus::SE3f historicalTwt = assoc.historicalTagPose;
        const auto itRef = mTagLoopReferences.find(pTag->Id());
        if(itRef != mTagLoopReferences.end())
            historicalTwt = itRef->second.frozenTwt;

        // 与 OpenCV IPPE_SQUARE / MapTagData::CornersFromPose 角点顺序一致
        const float h = 0.5f * tagSize;
        const Eigen::Vector3f Xt[4] = {
            Eigen::Vector3f(-h,  h, 0.f),
            Eigen::Vector3f( h,  h, 0.f),
            Eigen::Vector3f( h, -h, 0.f),
            Eigen::Vector3f(-h, -h, 0.f)};

        tag::TagPoseConstraint c;
        c.tagId = pTag->Id();
        for(int k = 0; k < 4; ++k)
            c.worldCorners[k] = (historicalTwt * Xt[k]).cast<double>();
        out.push_back(c);
    }

    return out;
}

bool LoopClosing::DetectTagLoop()
{
    if(!mTagLoopParams.enable || !mpCurrentKF)
        return false;

    PruneTagLoopReferences();

    const auto associations = mpCurrentKF->GetMapTagAssociations();
    if(associations.empty())
    {
        if(mnTagLoopCoincidences > 0)
        {
            mnTagLoopNotFound++;
            if(mnTagLoopNotFound >= 2)
                ResetTagLoopState();
        }
        return false;
    }

    // 获取一级共视关键帧局部地图
    std::set<KeyFrame*> localKFs;
    localKFs.insert(mpCurrentKF);
    const std::vector<KeyFrame*> vpCov = mpCurrentKF->GetVectorCovisibleKeyFrames();
    for(KeyFrame *pKF : vpCov)
        localKFs.insert(pKF);

    Map *pMap = mpCurrentKF->GetMap();

    struct TagLoopEstimate
    {
        tag::MapTagData *pTag = nullptr;
        KeyFrame *pMatchedKF = nullptr;
        Sophus::SE3f historicalTwt;
        Sophus::SE3f Tct;
        Sophus::SE3f TcwTag;
        bool stereoConsistent = false;
        int nValidCameras = 0;
        float currentReproj = 1e9f;
        bool frozen = false;
        bool inLocalMap = false;
    };
    std::vector<TagLoopEstimate> estimates;
    std::map<std::string, int> skipCounts;

    auto skipTag = [&](int tagId, const std::string &reason, const std::string &extra = {}) {
        skipCounts[reason]++;
        if(mTagLoopParams.verbose)
        {
            std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
                      << " skip tag=" << tagId
                      << " reason=" << reason;
            if(!extra.empty())
                std::cout << " " << extra;
            std::cout << std::endl;
        }
    };

    auto printSkipSummary = [&]() {
        std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
                  << " skip_summary assoc=" << associations.size()
                  << " estimates=" << estimates.size()
                  << " frozen_refs=" << mTagLoopReferences.size();
        for(const auto &kv : skipCounts)
            std::cout << " " << kv.first << "=" << kv.second;
        std::cout << std::endl;
    };

    auto logLiveDelta = [&](const Sophus::SE3f &frozenTwt, tag::MapTagData *pTag) {
        if(!pTag || !pTag->HasPose())
            return;
        float dtrans = 0.0f;
        float drot = 0.0f;
        TagLoopPoseDelta(frozenTwt, pTag->GetPose(), dtrans, drot);
        std::cout << " dtrans_live=" << dtrans
                  << " dang_live=" << drot;
    };

    auto findLastObservedKF = [&](tag::MapTagData *pTag) -> KeyFrame* {
        KeyFrame *pLastObservedKF = nullptr;
        const auto observations = pTag->GetObservations();
        for(const auto &obsKF : observations)
        {
            KeyFrame *pKF = obsKF.first;
            if(!pKF || pKF == mpCurrentKF || pKF->isBad())
                continue;
            if(pKF->GetMap() != pMap)
                continue;
            if(pKF->mnId >= mpCurrentKF->mnId)
                continue;
            if(!HistoricalTagObservationUsable(pKF, obsKF.second))
                continue;
            if(!pLastObservedKF || pKF->mnId > pLastObservedKF->mnId)
                pLastObservedKF = pKF;
        }
        return pLastObservedKF;
    };

    auto selectEarliestHistKF = [&](tag::MapTagData *pTag, std::string &rejectExtra) -> KeyFrame* {
        std::vector<KeyFrame*> candidates;
        int nLocal = 0, nGap = 0, nMps = 0, nUnusable = 0;
        const auto observations = pTag->GetObservations();
        for(const auto &obsKF : observations)
        {
            KeyFrame *pKF = obsKF.first;
            if(!pKF)
                continue;
            if(pKF == mpCurrentKF || pKF->mnId >= mpCurrentKF->mnId)
                continue;
            if(pKF->isBad())
                continue;
            if(pKF->GetMap() != pMap)
                continue;
            if(localKFs.count(pKF))
            {
                ++nLocal;
                continue;
            }
            const unsigned long gap = mpCurrentKF->mnId - pKF->mnId;
            if(gap < static_cast<unsigned long>(mTagLoopParams.min_keyframe_gap))
            {
                ++nGap;
                continue;
            }
            if(!HistoricalTagObservationUsable(pKF, obsKF.second))
            {
                ++nUnusable;
                continue;
            }
            if(pKF->TrackedMapPoints(1) < mTagLoopParams.min_historical_kf_map_points)
            {
                ++nMps;
                continue;
            }
            candidates.push_back(pKF);
        }

        KeyFrame *pBestHistKF = nullptr;
        for(KeyFrame *pKF : candidates)
        {
            if(!pBestHistKF || pKF->mnId < pBestHistKF->mnId ||
               (pKF->mnId == pBestHistKF->mnId &&
                pKF->TrackedMapPoints(1) > pBestHistKF->TrackedMapPoints(1)))
                pBestHistKF = pKF;
        }
        if(!pBestHistKF)
        {
            rejectExtra = "n_obs=" + std::to_string(observations.size()) +
                          " n_local=" + std::to_string(nLocal) +
                          " n_gap=" + std::to_string(nGap) +
                          " n_mps=" + std::to_string(nMps) +
                          " n_unusable=" + std::to_string(nUnusable) +
                          " min_gap=" + std::to_string(mTagLoopParams.min_keyframe_gap) +
                          " min_mps=" + std::to_string(mTagLoopParams.min_historical_kf_map_points);
        }
        return pBestHistKF;
    };

    // 遍历当前KF检测到的所有Tag
    for(const auto &kv : associations)
    {
        const int tag_id = kv.first;
        const KeyFrame::MapTagAssociation &assoc = kv.second;
        tag::MapTagData *pTag = assoc.pMapTag;

        // 跳过无效状态Tag
        if(!pTag || pTag->IsBad() || !pTag->HasPose())
        {
            if(mTagLoopReferences.count(tag_id))
                EraseTagLoopReference(tag_id, "bad_or_no_pose");
            skipTag(tag_id, "bad_or_no_pose");
            continue;
        }

        // 跳过无效状态Tag
        const tag::MapTagState state = pTag->GetState();
        if(state != tag::MapTagState::ACTIVE && state != tag::MapTagState::FIXED_ANCHOR)
        {
            skipTag(tag_id, "not_active",
                    "state=" + std::to_string(static_cast<int>(state)));
            continue;
        }

        // 跳过当前KF无效观测的Tag
        if(!CurrentTagObservationUsable(mpCurrentKF, assoc))
        {
            skipTag(tag_id, "current_obs_unusable",
                    "leftIdx=" + std::to_string(assoc.leftObservationIndex) +
                    " rightIdx=" + std::to_string(assoc.rightObservationIndex));
            continue;
        }

        // 检查回环冻结参考
        auto refIt = mTagLoopReferences.find(tag_id);
        bool hasFrozenRef = refIt != mTagLoopReferences.end();
        if(hasFrozenRef &&
           (!refIt->second.pHistoricalKF || refIt->second.pHistoricalKF->isBad() ||
            refIt->second.pHistoricalKF->GetMap() != pMap))
        {
            EraseTagLoopReference(tag_id, "hist_kf_invalid");
            refIt = mTagLoopReferences.find(tag_id);
            hasFrozenRef = false;
        }

        // Tag 是否出现在当前局部地图。找到一帧即可
        bool localContainsTag = false;
        const auto observations = pTag->GetObservations();
        for(const auto &obsKF : observations)
        {
            KeyFrame *pHistoricalKF = obsKF.first;
            if(!pHistoricalKF || pHistoricalKF == mpCurrentKF)
                continue;
            if(localKFs.count(pHistoricalKF))
            {
                localContainsTag = true;
                break;
            }
        }

        KeyFrame *pHistKF = nullptr;
        Sophus::SE3f frozenTwt;

        // 若存在回环冻结参考，则直接使用冻结参考的KF和位姿
        if(hasFrozenRef)
        {
            pHistKF = refIt->second.pHistoricalKF;
            frozenTwt = refIt->second.frozenTwt;
        }
        
        // 若不存在回环冻结参考，则需要进一步进行判断
        else
        {
            KeyFrame *pLastObservedKF = findLastObservedKF(pTag); // 找到最近一次有效观测的KF
            if(!pLastObservedKF)
            {
                skipTag(tag_id, "no_last_observation");
                continue;
            }
            // 检查最近一次有效观测间隔
            const unsigned long reobservationGap =
                mpCurrentKF->mnId - pLastObservedKF->mnId;
            if(reobservationGap <
               // 若最近一次有效观测间隔小于最小最近一次有效观测间隔，则跳过
               static_cast<unsigned long>(mTagLoopParams.min_reobservation_gap_kfs))
            {
                skipTag(tag_id, "reobservation_gap",
                        "lastKF=" + std::to_string(pLastObservedKF->mnId) +
                        " gap=" + std::to_string(reobservationGap) +
                        " min=" + std::to_string(mTagLoopParams.min_reobservation_gap_kfs) +
                        " in_local_map=" + std::to_string(localContainsTag ? 1 : 0));
                continue;
            }

            // 若不存在历史Tag位姿，则跳过
            if(!assoc.hasHistoricalTagPose)
            {
                skipTag(tag_id, "no_historical_pose",
                        "hist_obs=" + std::to_string(assoc.historicalObservationCount));
                continue;
            }
            // 若历史观测次数小于最小历史观测次数，则跳过
            if(static_cast<int>(assoc.historicalObservationCount) <
               mTagLoopParams.min_history_observations)
            {
                skipTag(tag_id, "min_history_obs",
                        "hist_obs=" + std::to_string(assoc.historicalObservationCount) +
                        " min=" + std::to_string(mTagLoopParams.min_history_observations));
                continue;
            }

            // 选择最早的历史KF作为当前帧的回环帧
            std::string rejectExtra;
            pHistKF = selectEarliestHistKF(pTag, rejectExtra);
            if(!pHistKF)
            {
                skipTag(tag_id, "no_qualified_historical_kf", rejectExtra);
                continue;
            }
            // 使用未被污染的历史Tag pose作为回环冻结参考
            frozenTwt = assoc.historicalTagPose;
        }

        // 估计Tag回环位姿
        Sophus::SE3f TcwTag, TctLeft;
        bool bStereoConsistent = false;
        int nValidCameras = 0;
        std::string poseFail;
        if(!EstimateTagLoopPose(assoc, frozenTwt, TcwTag, TctLeft, bStereoConsistent,
                                nValidCameras, &poseFail))
        {
            skipTag(tag_id, poseFail,
                    "leftIdx=" + std::to_string(assoc.leftObservationIndex) +
                    " rightIdx=" + std::to_string(assoc.rightObservationIndex) +
                    " frozen=" + std::to_string(hasFrozenRef ? 1 : 0));
            continue;
        }

        // 若不存在回环冻结参考（冻结pose计算完毕，需要进行冻结参考更新）
        if(!hasFrozenRef)
        {
            TagLoopReference ref;
            ref.frozenTwt = frozenTwt;
            ref.pHistoricalKF = pHistKF;
            ref.lastSeenKF = mpCurrentKF->mnId;
            pHistKF->SetNotErase();
            mTagLoopReferences[tag_id] = ref;
            std::cout << "[TagLoopRef] KF=" << mpCurrentKF->mnId
                      << " tag=" << tag_id
                      << " action=freeze histKF=" << pHistKF->mnId
                      << " lastSeen=" << mpCurrentKF->mnId
                      << " in_local_map=" << (localContainsTag ? 1 : 0)
                      << " Twt ";
            PrintSE3Compact(frozenTwt);
            logLiveDelta(frozenTwt, pTag);
            std::cout << std::endl;
        }
        // 若存在回环冻结参考（直接使用冻结pose）
        else
        {
            refIt->second.lastSeenKF = mpCurrentKF->mnId;
            std::cout << "[TagLoopRef] KF=" << mpCurrentKF->mnId
                      << " tag=" << tag_id
                      << " action=reuse histKF=" << pHistKF->mnId
                      << " in_local_map=" << (localContainsTag ? 1 : 0)
                      << " lastSeen=" << mpCurrentKF->mnId;
            logLiveDelta(frozenTwt, pTag);
            std::cout << std::endl;
        }

        TagLoopEstimate est;
        est.pTag = pTag;
        est.pMatchedKF = pHistKF;
        est.historicalTwt = frozenTwt;
        est.Tct = TctLeft;
        est.TcwTag = TcwTag;
        est.stereoConsistent = bStereoConsistent;
        est.nValidCameras = nValidCameras;
        est.currentReproj = CurrentTagReprojectionError(mpCurrentKF, assoc);
        est.frozen = hasFrozenRef;
        est.inLocalMap = localContainsTag;
        estimates.push_back(est);
    }

    if(!skipCounts.empty())
        printSkipSummary();

    const float maxT = mTagLoopParams.max_stereo_translation_error;
    const float maxR = mTagLoopParams.max_stereo_rotation_error;

    auto dumpPairs = [&](const std::vector<TagLoopEstimate> &groupA,
                         const std::vector<TagLoopEstimate> &groupB,
                         const char *src) {
        const bool sameGroup = &groupA == &groupB;
        for(size_t i = 0; i < groupA.size(); ++i)
        {
            if(!groupA[i].pTag)
                continue;
            const size_t jBegin = sameGroup ? i + 1 : 0;
            for(size_t j = jBegin; j < groupB.size(); ++j)
            {
                if(!groupB[j].pTag)
                    continue;
                PrintTagLoopPair(mpCurrentKF->mnId,
                                 groupA[i].pTag->Id(), groupB[j].pTag->Id(),
                                 groupA[i].TcwTag, groupB[j].TcwTag,
                                 maxT, maxR, src);
            }
        }
    };
    dumpPairs(estimates, estimates, "loop");

    if(estimates.empty())
    {
        if(skipCounts.empty())
            printSkipSummary();
        if(mnTagLoopCoincidences > 0)
        {
            mnTagLoopNotFound++;
            if(mnTagLoopNotFound >= 2)
                ResetTagLoopState();
        }
        return false;
    }

    std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
              << " valid_tags=" << estimates.size();
    for(const TagLoopEstimate &est : estimates)
    {
        const Sophus::SE3f dOrb = mpCurrentKF->GetPoseInverse() * est.TcwTag;
        std::cout << " {id=" << est.pTag->Id()
                  << " histKF=" << est.pMatchedKF->mnId
                  << " cams=" << est.nValidCameras
                  << " stereo=" << (est.stereoConsistent ? 1 : 0)
                  << " frozen=" << (est.frozen ? 1 : 0)
                  << " in_local_map=" << (est.inLocalMap ? 1 : 0)
                  << " reproj=" << est.currentReproj
                  << " dtrans_to_ORB=" << dOrb.translation().norm()
                  << " dang_to_ORB=" << dOrb.so3().log().norm()
                  << " TcwTag ";
        PrintSE3Compact(est.TcwTag);
        std::cout << "}";
    }
    std::cout << std::endl;

    auto buildCluster = [&](int seed) {
        std::vector<int> members;
        members.reserve(estimates.size());
        for(size_t j = 0; j < estimates.size(); ++j)
        {
            if(TagLoopPosesConsistent(estimates[seed].TcwTag, estimates[j].TcwTag, maxT, maxR))
                members.push_back(static_cast<int>(j));
        }
        return members;
    };

    auto clusterStereoCount = [&](const std::vector<int> &members) {
        int n = 0;
        for(int idx : members)
        {
            if(estimates[idx].stereoConsistent)
                ++n;
        }
        return n;
    };
    auto clusterCamSum = [&](const std::vector<int> &members) {
        int n = 0;
        for(int idx : members)
            n += estimates[idx].nValidCameras;
        return n;
    };
    auto clusterMinReproj = [&](const std::vector<int> &members) {
        float best = 1e9f;
        for(int idx : members)
            best = std::min(best, estimates[idx].currentReproj);
        return best;
    };
    auto clusterMinTagId = [&](const std::vector<int> &members) {
        int bestId = estimates[members.front()].pTag->Id();
        for(int idx : members)
            bestId = std::min(bestId, estimates[idx].pTag->Id());
        return bestId;
    };

    auto betterCluster = [&](const std::vector<int> &a, const std::vector<int> &b) {
        if(a.size() != b.size())
            return a.size() > b.size();
        const int stereoA = clusterStereoCount(a);
        const int stereoB = clusterStereoCount(b);
        if(stereoA != stereoB)
            return stereoA > stereoB;
        const int camA = clusterCamSum(a);
        const int camB = clusterCamSum(b);
        if(camA != camB)
            return camA > camB;
        const float reprojA = clusterMinReproj(a);
        const float reprojB = clusterMinReproj(b);
        if(reprojA != reprojB)
            return reprojA < reprojB;
        return clusterMinTagId(a) < clusterMinTagId(b);
    };

    auto betterEstimate = [&](int i, int j) {
        const TagLoopEstimate &a = estimates[i];
        const TagLoopEstimate &b = estimates[j];
        if(a.stereoConsistent != b.stereoConsistent)
            return a.stereoConsistent && !b.stereoConsistent;
        if(a.nValidCameras != b.nValidCameras)
            return a.nValidCameras > b.nValidCameras;
        if(a.currentReproj != b.currentReproj)
            return a.currentReproj < b.currentReproj;
        return a.pTag->Id() < b.pTag->Id();
    };

    // 根据不同Tag计算出当前相机pose的差异，筛选出多个计算当前相机pose差异小于指定阈值的Tag集合
    std::vector<int> cluster = buildCluster(0);
    for(size_t i = 1; i < estimates.size(); ++i)
    {
        std::vector<int> cand = buildCluster(static_cast<int>(i));
        // 只选出一个最优的Tag集合
        if(betterCluster(cand, cluster))
            cluster = std::move(cand);
    }
    if(static_cast<int>(cluster.size()) < mTagLoopParams.min_consistent_tags)
    {
        std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
                  << " reject cluster=" << cluster.size()
                  << " < min_consistent_tags="
                  << mTagLoopParams.min_consistent_tags << std::endl;
        return false;
    }

    std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
              << " cluster=" << cluster.size()
              << " stereo=" << clusterStereoCount(cluster)
              << " cams=" << clusterCamSum(cluster)
              << " tags=";
    for(size_t i = 0; i < cluster.size(); ++i)
    {
        if(i)
            std::cout << ",";
        std::cout << estimates[cluster[i]].pTag->Id();
    }
    std::cout << std::endl;

    int bestInCluster = cluster.front();
    for(int idx : cluster)
    {
        if(betterEstimate(idx, bestInCluster))
            bestInCluster = idx;
    }
    const TagLoopEstimate &best = estimates[bestInCluster];

    // 转换为固定尺度的Sim3位姿，并检查Tag计算的Sim3位姿和回环前计算的漂移pose差异是否满足指定条件
    const g2o::Sim3 gScwTag(best.TcwTag.unit_quaternion().cast<double>(),
                            best.TcwTag.translation().cast<double>(), 1.0);
    g2o::Sim3 gScwTag4DoF;
    if(!ValidateTagLoop(gScwTag, gScwTag4DoF))
    {
        std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
                  << " reject 4DoF tag=" << best.pTag->Id()
                  << " histKF=" << best.pMatchedKF->mnId << std::endl;
        return false;
    }

    // 判断本次Tag候选是否可靠
    bool bAccept = false;
    const char *acceptReason = "none";
    const unsigned long lastCoinKFId =
        mpTagLoopLastCurrentKF ? mpTagLoopLastCurrentKF->mnId : 0;
    if(static_cast<int>(cluster.size()) >= 2) // 多个Tag给出一致的当前KF位姿
    {
        bAccept = true;
        acceptReason = "multi_tag";
    }
    else if(best.stereoConsistent) // 左右目Tag计算的当前KF位姿一致
    {
        bAccept = true;
        acceptReason = "stereo";
    }
    else // 单Tag、单相机时连续KF验证
    {
        if(mnTagLoopCoincidences > 0 && mpTagLoopLastCurrentKF)
        {
            const Sophus::SE3f TcwLast(
                mg2oTagLoopLastScw.rotation().cast<float>(),
                mg2oTagLoopLastScw.translation().cast<float>());
            if(TagLoopPosesConsistent(best.TcwTag, TcwLast, maxT, maxR))
            {
                mnTagLoopCoincidences++;
                if(mnTagLoopCoincidences >= 2)
                {
                    bAccept = true;
                    acceptReason = "coincidence";
                }
                else
                {
                    acceptReason = "coincidence_wait";
                }
            }
            else
            {
                mnTagLoopCoincidences = 1;
                acceptReason = "coincidence_reset";
            }
        }
        else
        {
            mnTagLoopCoincidences = 1;
            acceptReason = "coincidence_start";
        }
    }

    std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
              << " accept_check accept=" << (bAccept ? 1 : 0)
              << " reason=" << acceptReason
              << " tag=" << best.pTag->Id()
              << " histKF=" << best.pMatchedKF->mnId
              << " cluster=" << cluster.size()
              << " stereo=" << (best.stereoConsistent ? 1 : 0)
              << " coincidences=" << mnTagLoopCoincidences
              << " lastKF=" << lastCoinKFId
              << " t_cur=" << mpCurrentKF->mTimeStamp
              << " t_hist=" << best.pMatchedKF->mTimeStamp
              << " dtrans=" << (mpCurrentKF->GetPoseInverse() * best.TcwTag).translation().norm()
              << " dang=" << (mpCurrentKF->GetPoseInverse() * best.TcwTag).so3().log().norm()
              << std::endl;

    // 保存本次Tag回环候选结果
    if(mpTagLoopMatchedKF && mpTagLoopMatchedKF != best.pMatchedKF &&
       mpTagLoopMatchedKF != mpCurrentKF)
        mpTagLoopMatchedKF->SetErase();
    best.pMatchedKF->SetNotErase();
    mpTagLoopMatchedKF = best.pMatchedKF;
    mpTagLoopLastCurrentKF = mpCurrentKF;
    mg2oTagLoopLastScw = gScwTag4DoF;
    mg2oTagLoopScw = gScwTag4DoF;
    mTagLoopTwtHistory = best.historicalTwt;
    mTagLoopTct = best.Tct;
    mnTagLoopNotFound = 0;

    if(!bAccept)
    {
        std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
                  << " candidate tag=" << best.pTag->Id()
                  << " histKF=" << best.pMatchedKF->mnId
                  << " coincidences=" << mnTagLoopCoincidences
                  << " cluster=" << cluster.size()
                  << " stereo=" << (best.stereoConsistent ? 1 : 0)
                  << std::endl;
        return false;
    }

    // 提交回环前：当前 KF 多 Tag 4DoF pose-only（单顶点 yaw+xyz，两轮 Huber）。
    // 组级剔除：每相机≥3 角点内点 / 左右合计≥6，且 Tag RMSE≤5px；至少 2 个内点 Tag 才写入 mg2oLoopScw。
    // 失败则保留 coincidence 候选、不提交。
    {
        const auto constraints = BuildTagLoopPoseConstraints(mpCurrentKF);
        // seed 来自触发 Tag 的 IPPE + ValidateTagLoop 零化 roll/pitch 后的 4DoF Sim3
        const Sophus::SE3f seedTcw(
            gScwTag4DoF.rotation().cast<float>(),
            gScwTag4DoF.translation().cast<float>());
        Sophus::SE3f refinedTcw = seedTcw;
        Optimizer::TagLoopPoseOptStats stats;
        const float cornerSigma = (mTagLocalBAParams.corner_sigma > 0.0f)
                                      ? mTagLocalBAParams.corner_sigma
                                      : 2.0f;
        const bool refined = Optimizer::OptimizeTagLoopPose4DoF(
            mpCurrentKF, constraints, seedTcw, refinedTcw, &stats, cornerSigma,
            mTagLocalBAParams.factor_weight);

        float dtrans = 0.0f;
        float drot = 0.0f;
        TagLoopPoseDelta(seedTcw, refinedTcw, dtrans, drot);

        auto joinIds = [](const std::vector<int> &ids) {
            std::ostringstream oss;
            for(size_t i = 0; i < ids.size(); ++i)
            {
                if(i)
                    oss << ",";
                oss << ids[i];
            }
            return oss.str();
        };

        if(!refined)
        {
            std::cout << "[TagLoopPoseOpt]"
                      << " KF=" << mpCurrentKF->mnId
                      << " trigger=" << best.pTag->Id()
                      << " status=rejected"
                      << " reason=min_inlier_tags"
                      << " input_tags=" << stats.inputTags
                      << " inlier_tags=" << stats.inlierTags
                      << " tags_outlier=" << joinIds(stats.outlierTagIds)
                      << std::endl;
            return false;
        }

        // 用多 Tag 优化后的 4DoF 位姿覆盖 seed，再写入 mg2oLoopScw
        gScwTag4DoF = g2o::Sim3(refinedTcw.unit_quaternion().cast<double>(),
                                refinedTcw.translation().cast<double>(), 1.0);
        mg2oTagLoopLastScw = gScwTag4DoF;
        mg2oTagLoopScw = gScwTag4DoF;

        std::cout << "[TagLoopPoseOpt]"
                  << " KF=" << mpCurrentKF->mnId
                  << " trigger=" << best.pTag->Id()
                  << " input_tags=" << stats.inputTags
                  << " edges_L=" << stats.inputEdgesLeft
                  << " edges_R=" << stats.inputEdgesRight
                  << " inlier_tags=" << stats.inlierTags
                  << " inlier_corners=" << stats.inlierCorners
                  << " rmse_before=" << stats.rmseBefore
                  << " rmse_after=" << stats.rmseAfter
                  << " seed_to_refined_trans=" << dtrans
                  << " seed_to_refined_rot=" << drot
                  << " tags_inlier=" << joinIds(stats.inlierTagIds)
                  << " tags_outlier=" << joinIds(stats.outlierTagIds)
                  << " status=accepted"
                  << std::endl;
    }

    // Drop unfinished BoW loop state. These pointers are only valid after a BoW
    // assignment; they must not be dereferenced when still null/uninitialized.
    if(mpLoopLastCurrentKF && mpLoopLastCurrentKF != mpCurrentKF)
        mpLoopLastCurrentKF->SetErase();
    mpLoopLastCurrentKF = nullptr;
    if(mpLoopMatchedKF && mpLoopMatchedKF != best.pMatchedKF &&
       mpLoopMatchedKF != mpCurrentKF)
        mpLoopMatchedKF->SetErase();
    mpLoopMatchedKF = nullptr;
    mnLoopNumCoincidences = 0;
    mvpLoopMatchedMPs.clear();
    mvpLoopMPs.clear();
    mnLoopNumNotFound = 0;

    mbLoopDetected = true;
    mbTagLoopDetected = true;
    mnLoopTagId = best.pTag->Id();
    mpLoopTag = best.pTag;
    mpLoopMatchedKF = best.pMatchedKF;
    mpLoopLastCurrentKF = mpCurrentKF;
    mg2oLoopScw = gScwTag4DoF;
    mg2oLoopSlw = gScwTag4DoF;
    CollectTagLoopMapPoints(best.pMatchedKF, mvpLoopMPs);
    mvpLoopMatchedMPs.clear();

    const char *reason = "coincidence";
    if(static_cast<int>(cluster.size()) >= 2)
        reason = "multi_tag";
    else if(best.stereoConsistent)
        reason = "stereo";

    std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
              << " valid loop confirmed"
              << " tag=" << mnLoopTagId
              << " histKF=" << mpLoopMatchedKF->mnId
              << " t_cur=" << mpCurrentKF->mTimeStamp
              << " t_hist=" << mpLoopMatchedKF->mTimeStamp
              << " cluster=" << cluster.size()
              << " stereo=" << (best.stereoConsistent ? 1 : 0)
              << " cams=" << best.nValidCameras
              << " reason=" << reason
              << " dtrans=" << (mpCurrentKF->GetPoseInverse() * best.TcwTag).translation().norm()
              << " dang=" << (mpCurrentKF->GetPoseInverse() * best.TcwTag).so3().log().norm()
              << " mps=" << mvpLoopMPs.size() << std::endl;
    return true;
}

bool LoopClosing::NewDetectCommonRegions()
{
    // To deactivate placerecognition. No loopclosing nor merging will be performed
    if(!mbActiveLC)
        return false;

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;

        mpLastMap = mpCurrentKF->GetMap();
    }

    auto logTagDetectSkipped = [&](const char *reason) {
        if(!mTagLoopParams.enable || !mpCurrentKF)
            return;
        if(mpCurrentKF->mTagFrameData.Empty())
            return;
        std::cout << "[TagLoop] KF " << mpCurrentKF->mnId
                  << " skip_detect reason=" << reason
                  << " detections=" << mpCurrentKF->mTagFrameData.Size()
                  << std::endl;
    };

    if(mpLastMap->IsInertial() && !mpLastMap->GetIniertialBA2())
    {
        logTagDetectSkipped("imu_ba2_not_ready");
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    if(mpTracker->mSensor == System::STEREO && mpLastMap->GetAllKeyFrames().size() < 5) //12
    {
        logTagDetectSkipped("map_too_small_stereo");
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    if(mpLastMap->GetAllKeyFrames().size() < 12)
    {
        logTagDetectSkipped("map_too_small");
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    //cout << "LoopClousure: Checking KF: " << mpCurrentKF->mnId << endl;

    //Check the last candidates with geometric validation
    // Loop candidates
    bool bLoopDetectedInKF = false;
    bool bCheckSpatial = false;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartEstSim3_1 = std::chrono::steady_clock::now();
#endif
    if(mnLoopNumCoincidences > 0)
    {
        bCheckSpatial = true;
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse()).cast<double>();
        g2o::Sim3 gScl(mTcl.unit_quaternion(),mTcl.translation(),1.0);
        g2o::Sim3 gScw = gScl * mg2oLoopSlw;
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs);
        if(bCommonRegion)
        {

            bLoopDetectedInKF = true;

            mnLoopNumCoincidences++;
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw = gScw;
            mvpLoopMatchedMPs = vpMatchedMPs;


            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if(!mbLoopDetected)
            {
                cout << "PR: Loop detected with Reffine Sim3" << endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false;

            mnLoopNumNotFound++;
            if(mnLoopNumNotFound >= 2)
            {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mpLoopLastCurrentKF = nullptr;
                mpLoopMatchedKF = nullptr;
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }

        }
    }

    //Merge candidates
    bool bMergeDetectedInKF = false;
    if(mnMergeNumCoincidences > 0)
    {
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse()).cast<double>();

        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3 gScw = gScl * mg2oMergeSlw;
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs);
        if(bCommonRegion)
        {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeSlw = gScw;
            mvpMergeMatchedMPs = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        }
        else
        {
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if(mnMergeNumNotFound >= 2)
            {
                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mpMergeLastCurrentKF = nullptr;
                mpMergeMatchedKF = nullptr;
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0;
            }


        }
    }  
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_1 = std::chrono::steady_clock::now();

        double timeEstSim3 = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_1 - time_StartEstSim3_1).count();
#endif

    // 检测回环候选
    DetectTagLoop();

    if(mbMergeDetected || mbLoopDetected)
    {
#ifdef REGISTER_TIMES
        vdEstSim3_ms.push_back(timeEstSim3);
#endif
        mpKeyFrameDB->add(mpCurrentKF);
        return true;
    }

    //TODO: This is only necessary if we use a minimun score for pick the best candidates
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();

    // Extract candidates from the bag of words
    vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
    if(!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
        // Search in BoW
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartQuery = std::chrono::steady_clock::now();
#endif
        mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand,3);
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndQuery = std::chrono::steady_clock::now();

        double timeDataQuery = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndQuery - time_StartQuery).count();
        vdDataQuery_ms.push_back(timeDataQuery);
#endif
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartEstSim3_2 = std::chrono::steady_clock::now();
#endif
    // Check the BoW candidates if the geometric candidate list is empty
    //Loop candidates
    if(!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, mg2oLoopSlw, mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMatchedMPs);
    }
    // Merge candidates
    if(!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF, mg2oMergeSlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMatchedMPs);
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_2 = std::chrono::steady_clock::now();

        timeEstSim3 += std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_2 - time_StartEstSim3_2).count();
        vdEstSim3_ms.push_back(timeEstSim3);
#endif

    mpKeyFrameDB->add(mpCurrentKF);

    if(mbMergeDetected || mbLoopDetected)
    {
        return true;
    }

    mpCurrentKF->SetErase();
    mpCurrentKF->mbCurrentPlaceRecognition = false;

    return false;
}

bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if(nNumProjMatches >= nProjMatches)
    {
        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(nNumProjMatches) + " initial matches ", Verbose::VERBOSITY_DEBUG);
        Sophus::SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>();
        g2o::Sim3 gSwm(mTwm.unit_quaternion(),mTwm.translation(),1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        bool bFixedScale = mbFixScale;       // TODO CHECK; Solo para el monocular inertial
        if(mpTracker->mSensor==System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
            bFixedScale=false;
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);

        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(numOptMatches) + " matches after of the optimization ", Verbose::VERBOSITY_DEBUG);

        if(numOptMatches > nProjOptMatches)
        {
            g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(),1.0);

            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF2, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                             int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    set<KeyFrame*> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

    int nNumCovisibles = 10;

    ORBmatcher matcherBoW(0.9, true);
    ORBmatcher matcher(0.75, true);

    // Varibles to select the best numbe
    KeyFrame* pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    //Verbose::PrintMess("BoW candidates: There are " + to_string(vpBowCand.size()) + " possible candidates ", Verbose::VERBOSITY_DEBUG);
    for(KeyFrame* pKFi : vpBowCand)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        // std::cout << "KF candidate: " << pKFi->mnId << std::endl;
        // Current KF against KF with covisibles version
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        if(vpCovKFi.empty())
        {
            std::cout << "Covisible list empty" << std::endl;
            vpCovKFi.push_back(pKFi);
        }
        else
        {
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }


        bool bAbortByNearKF = false;
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                bAbortByNearKF = true;
                break;
            }
        }
        if(bAbortByNearKF)
        {
            //std::cout << "Check BoW aborted because is close to the matched one " << std::endl;
            continue;
        }
        //std::cout << "Check BoW continue because is far to the matched one " << std::endl;


        std::vector<std::vector<MapPoint*> > vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<MapPoint*> spMatchedMPi;
        int numBoWMatches = 0;

        KeyFrame* pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0;

        std::vector<MapPoint*> vpMatchedPoints = std::vector<MapPoint*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
        std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));

        int nIndexMostBoWMatchesKF=0;
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            if(!vpCovKFi[j] || vpCovKFi[j]->isBad())
                continue;

            int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
            if (num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            for(int k=0; k < vvpMatchedMPs[j].size(); ++k)
            {
                MapPoint* pMPi_j = vvpMatchedMPs[j][k];
                if(!pMPi_j || pMPi_j->isBad())
                    continue;

                if(spMatchedMPi.find(pMPi_j) == spMatchedMPi.end())
                {
                    spMatchedMPi.insert(pMPi_j);
                    numBoWMatches++;

                    vpMatchedPoints[k]= pMPi_j;
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j];
                }
            }
        }

        //pMostBoWMatchesKF = vpCovKFi[pMostBoWMatchesKF];

        if(numBoWMatches >= nBoWMatches) // TODO pick a good threshold
        {
            // Geometric validation
            bool bFixedScale = mbFixScale;
            if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                bFixedScale=false;

            Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300); // at least 15 inliers

            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers;
            bool bConverge = false;
            Eigen::Matrix4f mTcm;
            while(!bConverge && !bNoMore)
            {
                mTcm = solver.iterate(20,bNoMore, vbInliers, nInliers, bConverge);
                //Verbose::PrintMess("BoW guess: Solver achieve " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
            }

            if(bConverge)
            {
                //std::cout << "Check BoW: SolverSim3 converged" << std::endl;

                //Verbose::PrintMess("BoW guess: Convergende with " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
                // Match by reprojection
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                vpCovKFi.push_back(pMostBoWMatchesKF);
                set<KeyFrame*> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                //std::cout << "There are " << vpCovKFi.size() <<" near KFs" << std::endl;

                set<MapPoint*> spMapPoints;
                vector<MapPoint*> vpMapPoints;
                vector<KeyFrame*> vpKeyFrames;
                for(KeyFrame* pCovKFi : vpCovKFi)
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches())
                    {
                        if(!pCovMPij || pCovMPij->isBad())
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                            vpKeyFrames.push_back(pCovKFi);
                        }
                    }
                }

                //std::cout << "There are " << vpKeyFrames.size() <<" KFs which view all the mappoints" << std::endl;

                g2o::Sim3 gScm(solver.GetEstimatedRotation().cast<double>(),solver.GetEstimatedTranslation().cast<double>(), (double) solver.GetEstimatedScale());
                g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0);
                g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position
                Sophus::Sim3f mScw = Converter::toSophus(gScw);

                vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                vector<KeyFrame*> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);
                //cout <<"BoW: " << numProjMatches << " matches between " << vpMapPoints.size() << " points with coarse Sim3" << endl;

                if(numProjMatches >= nProjMatches)
                {
                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    bool bFixedScale = mbFixScale;
                    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                        bFixedScale=false;

                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, mHessian7x7, true);

                    if(numOptMatches >= nSim3Inliers)
                    {
                        g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0);
                        g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position
                        Sophus::Sim3f mScw = Converter::toSophus(gScw);

                        vector<MapPoint*> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);

                        if(numProjOptMatches >= nProjOptMatches)
                        {
                            int max_x = -1, min_x = 1000000;
                            int max_y = -1, min_y = 1000000;
                            for(MapPoint* pMPi : vpMatchedMP)
                            {
                                if(!pMPi || pMPi->isBad())
                                {
                                    continue;
                                }

                                tuple<size_t,size_t> indexes = pMPi->GetIndexInKeyFrame(pKFi);
                                int index = get<0>(indexes);
                                if(index >= 0)
                                {
                                    int coord_x = pKFi->mvKeysUn[index].pt.x;
                                    if(coord_x < min_x)
                                    {
                                        min_x = coord_x;
                                    }
                                    if(coord_x > max_x)
                                    {
                                        max_x = coord_x;
                                    }
                                    int coord_y = pKFi->mvKeysUn[index].pt.y;
                                    if(coord_y < min_y)
                                    {
                                        min_y = coord_y;
                                    }
                                    if(coord_y > max_y)
                                    {
                                        max_y = coord_y;
                                    }
                                }
                            }

                            int nNumKFs = 0;
                            //vpMatchedMPs = vpMatchedMP;
                            //vpMPs = vpMapPoints;
                            // Check the Sim3 transformation with the current KeyFrame covisibles
                            vector<KeyFrame*> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);

                            int j = 0;
                            while(nNumKFs < 3 && j<vpCurrentCovKFs.size())
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j];
                                Sophus::SE3d mTjc = (pKFj->GetPose() * mpCurrentKF->GetPoseInverse()).cast<double>();
                                g2o::Sim3 gSjc(mTjc.unit_quaternion(),mTjc.translation(),1.0);
                                g2o::Sim3 gSjw = gSjc * gScw;
                                int numProjMatches_j = 0;
                                vector<MapPoint*> vpMatchedMPs_j;
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj,pMostBoWMatchesKF, gSjw,numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if(bValid)
                                {
                                    Sophus::SE3f Tc_w = mpCurrentKF->GetPose();
                                    Sophus::SE3f Tw_cj = pKFj->GetPoseInverse();
                                    Sophus::SE3f Tc_cj = Tc_w * Tw_cj;
                                    Eigen::Vector3f vector_dist = Tc_cj.translation();
                                    nNumKFs++;
                                }
                                j++;
                            }

                            if(nNumKFs < 3)
                            {
                                vnStage[index] = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }

                            if(nBestMatchesReproj < numProjOptMatches)
                            {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestScw = gScw;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP;
                            }
                        }
                    }
                }
            }
            /*else
            {
                Verbose::PrintMess("BoW candidate: it don't match with the current one", Verbose::VERBOSITY_DEBUG);
            }*/
        }
        index++;
    }

    if(nBestMatchesReproj > 0)
    {
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw;
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        return nNumCoincidences >= 3;
    }
    else
    {
        int maxStage = -1;
        int maxMatched;
        for(int i=0; i<vnStage.size(); ++i)
        {
            if(vnStage[i] > maxStage)
            {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    if(nNumProjMatches >= nProjMatches)
    {
        return true;
    }

    return false;
}

int LoopClosing::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                         set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                         vector<MapPoint*> &vpMatchedMapPoints)
{
    int nNumCovisibles = 10;
    vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<KeyFrame*> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames();
    if(nInitialCov < nNumCovisibles)
    {
        for(int i=0; i<nInitialCov; ++i)
        {
            vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
            int nInserted = 0;
            int j = 0;
            while(j < vpKFs.size() && nInserted < nNumCovisibles)
            {
                if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end())
                {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }
    set<MapPoint*> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    for(KeyFrame* pKFi : vpCovKFm)
    {
        for(MapPoint* pMPij : pKFi->GetMapPointMatches())
        {
            if(!pMPij || pMPij->isBad())
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    Sophus::Sim3f mScw = Converter::toSophus(g2oScw);
    ORBmatcher matcher(0.9, true);

    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

void LoopClosing::CorrectLoop()
{
    ClearTagLoopReferences(mbTagLoopDetected ? "tag_loop_accept" : "bow_loop");
    if(mbTagLoopDetected)
    {
        std::cout << "[KF " << mpCurrentKF->mnId << "] Loop confirmed type=tag"
                  << " tag=" << mnLoopTagId;
        if(mpLoopMatchedKF)
            std::cout << " histKF=" << mpLoopMatchedKF->mnId
                      << " t_cur=" << mpCurrentKF->mTimeStamp
                      << " t_hist=" << mpLoopMatchedKF->mTimeStamp;
        std::cout << std::endl;
    }
    else
    {
        std::cout << "[KF " << mpCurrentKF->mnId << "] Loop confirmed" << std::endl;
    }

    assert(!mpLocalMapper->HasPendingKeyFrames());

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();
    //assert(mpCurrentKF->GetMap()->CheckEssentialGraph());

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    //std::cout << "Loop: number of connected KFs -> " + to_string(mvpCurrentConnectedKFs.size()) << std::endl;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oLoopScw;
    Sophus::SE3f Twc = mpCurrentKF->GetPoseInverse();
    Sophus::SE3f Tcw = mpCurrentKF->GetPose();
    g2o::Sim3 g2oScw(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>(),1.0);
    NonCorrectedSim3[mpCurrentKF]=g2oScw;

    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
    Sophus::SE3d correctedTcw(mg2oLoopScw.rotation(),mg2oLoopScw.translation() / mg2oLoopScw.scale());
    mpCurrentKF->SetPose(correctedTcw.cast<float>());

    Map* pLoopMap = mpCurrentKF->GetMap();

#ifdef REGISTER_TIMES
    /*KeyFrame* pKF = mpCurrentKF;
    int numKFinLoop = 0;
    while(pKF && pKF->mnId > mpLoopMatchedKF->mnId)
    {
        pKF = pKF->GetParent();
        numKFinLoop += 1;
    }
    vnLoopKFs.push_back(numKFinLoop);*/

    std::chrono::steady_clock::time_point time_StartFusion = std::chrono::steady_clock::now();
#endif

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);

        const bool bImuInit = pLoopMap->isImuInitialized();

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            if(pKFi!=mpCurrentKF)
            {
                Sophus::SE3f Tiw = pKFi->GetPose();
                Sophus::SE3d Tic = (Tiw * Twc).cast<double>();
                g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oLoopScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
                pKFi->SetPose(correctedTiw.cast<float>());

                //Pose without correction
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion().cast<double>(),Tiw.translation().cast<double>(),1.0);
                NonCorrectedSim3[pKFi]=g2oSiw;
            }  
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            /*Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
            pKFi->SetPose(correctedTiw.cast<float>());*/

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw));

                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Correct velocity according to orientation correction
            if(bImuInit)
            {
                Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse()*g2oSiw.rotation()).cast<float>();
                pKFi->SetVelocity(Rcor*pKFi->GetVelocity());
            }

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        if(mbTagLoopDetected)
        {
            const std::vector<Map::MapTagPtr> vpTags = pLoopMap->GetAllMapTags();
            for(const Map::MapTagPtr &pTagPtr : vpTags)
            {
                tag::MapTagData *pTag = pTagPtr.get();
                if(!pTag || pTag->IsBad() || !pTag->HasPose())
                    continue;
                if(pTag == mpLoopTag || pTag->IsFixed())
                    continue;
                if(pTag->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                KeyFrame *pRefKF = nullptr;
                int bestScore = -1;
                const auto tagObs = pTag->GetObservations();
                for(const auto &obsKF : tagObs)
                {
                    KeyFrame *pKF = obsKF.first;
                    if(!pKF || CorrectedSim3.count(pKF) == 0)
                        continue;
                    const int score = pKF->TrackedMapPoints(1);
                    if(score > bestScore)
                    {
                        bestScore = score;
                        pRefKF = pKF;
                    }
                }
                if(!pRefKF)
                    continue;

                const g2o::Sim3 g2oCorrectedSiw = CorrectedSim3[pRefKF];
                const g2o::Sim3 g2oSiw = NonCorrectedSim3[pRefKF];
                const g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

                const Sophus::SE3f TwtOld = pTag->GetPose();
                const Eigen::Matrix3f Rcor =
                    (g2oCorrectedSiw.rotation().inverse() * g2oSiw.rotation())
                        .toRotationMatrix()
                        .cast<float>();
                const Eigen::Matrix3f Rnew = Rcor * TwtOld.rotationMatrix();
                const Eigen::Vector3f tnew =
                    g2oCorrectedSwi.map(g2oSiw.map(TwtOld.translation().cast<double>())).cast<float>();
                pTag->SetPose(Sophus::SE3f(Rnew, tnew));
                pTag->mnCorrectedByKF = mpCurrentKF->mnId;
                pTag->mnCorrectedReference = pRefKF->mnId;
            }
        }

        // TODO Check this index increasement
        mpAtlas->GetCurrentMap()->IncreaseChangeIndex();


        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpLoopMatchedMPs.size(); i++)
        {
            if(mvpLoopMatchedMPs[i])
            {
                MapPoint* pLoopMP = mvpLoopMatchedMPs[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
        //cout << "LC: end replacing duplicated" << endl;
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    if(mbTagLoopDetected && mpCurrentKF && mpLoopMatchedKF)
        LoopConnections[mpCurrentKF].insert(mpLoopMatchedKF);

    // Optimize graph
    bool bFixedScale = mbFixScale;
    // TODO CHECK; Solo para el monocular inertial
    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
        bFixedScale=false;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndFusion = std::chrono::steady_clock::now();

        double timeFusion = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndFusion - time_StartFusion).count();
        vdLoopFusion_ms.push_back(timeFusion);
#endif
    //cout << "Optimize essential graph" << endl;
    if(pLoopMap->IsInertial() && pLoopMap->isImuInitialized())
    {
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections,
                                              mbTagLoopDetected ? mpLoopTag : nullptr);
    }
    else
    {
        //cout << "Loop -> Scale correction: " << mg2oLoopScw.scale() << endl;
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale,
                                          mbTagLoopDetected ? mpLoopTag : nullptr);
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOpt = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOpt - time_EndFusion).count();
    vdLoopOptEss_ms.push_back(timeOptEss);
#endif

    mpAtlas->InformNewBigChange();

    // Add loop edge
    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);

    std::cout << "[KF " << mpCurrentKF->mnId << "] CorrectLoop finished" << std::endl;
    std::cout << "[KF " << mpCurrentKF->mnId << "] EssentialGraph finished" << std::endl;

    // Run GBA unless the inertial system has multiple maps.
    // In particular, an initialized single-map system still runs GBA even
    // when the map contains 200 or more keyframes.
    if(!pLoopMap->isImuInitialized() || mpAtlas->CountMaps()==1)
    {
        std::cout << "[KF " << mpCurrentKF->mnId << "] GBA begin" << std::endl;
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mnCorrectionGBA = mnNumCorrection;
        RunGlobalBundleAdjustment(pLoopMap, mpCurrentKF->mnId);
        std::cout << "[KF " << mpCurrentKF->mnId << "] GBA map update finished" << std::endl;
    }

    mLastLoopKFid = mpCurrentKF->mnId;
}

void LoopClosing::MergeLocal()
{
    int numTemporalKFs = 25; //Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    bool bRelaunchBA = false;

    assert(!mpLocalMapper->HasPendingKeyFrames());

    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    //std::cout << "Merge local, Active map: " << pCurrentMap->GetId() << std::endl;
    //std::cout << "Merge local, Non-Active map: " << pMergeMap->GetId() << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();
#endif

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    //Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    set<KeyFrame*> spLocalWindowKFs;
    //Get MPs in the welding area from the current map
    set<MapPoint*> spLocalWindowMPs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpCurrentKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = mpCurrentKF->mNextKF;
        while(pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());

            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spLocalWindowKFs.insert(mpCurrentKF);
    }

    vector<KeyFrame*> vpCovisibleKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spLocalWindowKFs.insert(mpCurrentKF);
    const int nMaxTries = 5;
    int nNumTries = 0;
    while(spLocalWindowKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        vpNewCovKFs.empty();
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        set<MapPoint*> spMPs = pKFi->GetMapPoints();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }

    //std::cout << "[Merge]: Ma = " << to_string(pCurrentMap->GetId()) << "; #KFs = " << to_string(spLocalWindowKFs.size()) << "; #MPs = " << to_string(spLocalWindowMPs.size()) << std::endl;

    set<KeyFrame*> spMergeConnectedKFs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpMergeMatchedKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs/2)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;
        }

        pKFi = mpMergeMatchedKF->mNextKF;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spMergeConnectedKFs.insert(mpMergeMatchedKF);
    }
    vpCovisibleKFs = mpMergeMatchedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spMergeConnectedKFs.insert(mpMergeMatchedKF);
    nNumTries = 0;
    while(spMergeConnectedKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        for(KeyFrame* pKFi : spMergeConnectedKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spMergeConnectedKFs.find(pKFcov) == spMergeConnectedKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
    }

    vector<MapPoint*> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    //std::cout << "[Merge]: Mm = " << to_string(pMergeMap->GetId()) << "; #KFs = " << to_string(spMergeConnectedKFs.size()) << "; #MPs = " << to_string(spMapPointMerge.size()) << std::endl;


    //
    Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
    g2o::Sim3 g2oNonCorrectedSwc(Twc.unit_quaternion(),Twc.translation(),1.0);
    g2o::Sim3 g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3 g2oCorrectedScw = mg2oMergeScw; //TODO Check the transformation

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[mpCurrentKF]=g2oCorrectedScw;
    vNonCorrectedSim3[mpCurrentKF]=g2oNonCorrectedScw;


#ifdef REGISTER_TIMES
    vnMergeKFs.push_back(spLocalWindowKFs.size() + spMergeConnectedKFs.size());
    vnMergeMPs.push_back(spLocalWindowMPs.size() + spMapPointMerge.size());
#endif
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
        {
            Verbose::PrintMess("Bad KF in correction", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(pKFi->GetMap() != pCurrentMap)
            Verbose::PrintMess("Other map KF, this should't happen", Verbose::VERBOSITY_DEBUG);

        g2o::Sim3 g2oCorrectedSiw;

        if(pKFi!=mpCurrentKF)
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            //Pose without correction
            vNonCorrectedSim3[pKFi]=g2oSiw;

            Sophus::SE3d Tic = Tiw*Twc;
            g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0);
            g2oCorrectedSiw = g2oSic*mg2oMergeScw;
            vCorrectedSim3[pKFi]=g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->mTcwMerge  = pKFi->GetPose();

        // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
        double s = g2oCorrectedSiw.scale();
        pKFi->mfScale = s;
        Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

        pKFi->mTcwMerge = correctedTiw.cast<float>();

        if(pCurrentMap->isImuInitialized())
        {
            Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
            pKFi->mVwbMerge = Rcor * pKFi->GetVelocity();
        }

        //TODO DEBUG to know which are the KFs that had been moved to the other map
    }

    int numPointsWithCorrection = 0;

    //for(MapPoint* pMPi : spLocalWindowMPs)
    set<MapPoint*>::iterator itMP = spLocalWindowMPs.begin();
    while(itMP != spLocalWindowMPs.end())
    {
        MapPoint* pMPi = *itMP;
        if(!pMPi || pMPi->isBad())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }

        KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
        if(vCorrectedSim3.find(pKFref) == vCorrectedSim3.end())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            numPointsWithCorrection++;
            continue;
        }
        g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
        Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
        Eigen::Quaterniond Rcor = g2oCorrectedSwi.rotation() * g2oNonCorrectedSiw.rotation();

        pMPi->mPosMerge = eigCorrectedP3Dw.cast<float>();
        pMPi->mNormalVectorMerge = Rcor.cast<float>() * pMPi->GetNormal();

        itMP++;
    }
    /*if(numPointsWithCorrection>0)
    {
        std::cout << "[Merge]: " << std::to_string(numPointsWithCorrection) << " points removed from Ma due to its reference KF is not in welding area" << std::endl;
        std::cout << "[Merge]: Ma has " << std::to_string(spLocalWindowMPs.size()) << " points" << std::endl;
    }*/

    {
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        //std::cout << "Merge local window: " << spLocalWindowKFs.size() << std::endl;
        //std::cout << "[Merge]: init merging maps " << std::endl;
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            if(!pKFi || pKFi->isBad())
            {
                //std::cout << "Bad KF in correction" << std::endl;
                continue;
            }

            //std::cout << "KF id: " << pKFi->mnId << std::endl;

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(pKFi->mTcwMerge);

            // Make sure connections are updated
            pKFi->UpdateMap(pMergeMap);
            pKFi->mnMergeCorrectedForKF = mpCurrentKF->mnId;
            pMergeMap->AddKeyFrame(pKFi);
            pCurrentMap->EraseKeyFrame(pKFi);

            if(pCurrentMap->isImuInitialized())
            {
                pKFi->SetVelocity(pKFi->mVwbMerge);
            }
        }

        for(MapPoint* pMPi : spLocalWindowMPs)
        {
            if(!pMPi || pMPi->isBad())
                continue;

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->SetNormalVector(pMPi->mNormalVectorMerge);
            pMPi->UpdateMap(pMergeMap);
            pMergeMap->AddMapPoint(pMPi);
            pCurrentMap->EraseMapPoint(pMPi);
        }

        mpAtlas->ChangeMap(pMergeMap);
        mpAtlas->SetMapBad(pCurrentMap);
        pMergeMap->IncreaseChangeIndex();
        //TODO for debug
        pMergeMap->ChangeId(pCurrentMap->GetId());

        //std::cout << "[Merge]: merging maps finished" << std::endl;
    }

    //Rebuild the essential graph in the local window
    pCurrentMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpCurrentKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpCurrentKF->ChangeParent(mpMergeMatchedKF);
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();

        pNewChild->ChangeParent(pNewParent);

        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }

    //Update the connections between the local window
    mpMergeMatchedKF->UpdateConnections();

    vpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    //vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    //std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //std::cout << "[Merge]: start fuse points" << std::endl;
    SearchAndFuse(vCorrectedSim3, vpCheckFuseMapPoint);
    //std::cout << "[Merge]: fuse points finished" << std::endl;

    // Update connectivity
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }

    //std::cout << "[Merge]: Start welding bundle adjustment" << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartWeldingBA = std::chrono::steady_clock::now();

    double timeMergeMaps = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartWeldingBA - time_StartMerge).count();
    vdMergeMaps_ms.push_back(timeMergeMaps);
#endif

    bool bStop = false;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    if (mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
    {
        Optimizer::MergeInertialBA(mpCurrentKF,mpMergeMatchedKF,&bStop, pCurrentMap,vCorrectedSim3);
    }
    else
    {
        Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs, vpMergeConnectedKFs,&bStop);
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndWeldingBA = std::chrono::steady_clock::now();

    double timeWeldingBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndWeldingBA - time_StartWeldingBA).count();
    vdWeldingBA_ms.push_back(timeWeldingBA);
#endif
    //std::cout << "[Merge]: Welding bundle adjustment finished" << std::endl;

    //Update the non critical area from the current map to the merged map
    vector<KeyFrame*> vpCurrentMapKFs = pCurrentMap->GetAllKeyFrames();
    vector<MapPoint*> vpCurrentMapMPs = pCurrentMap->GetAllMapPoints();

    if(vpCurrentMapKFs.size() == 0){}
    else {
        if(mpTracker->mSensor == System::MONOCULAR)
        {
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information

            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }

                g2o::Sim3 g2oCorrectedSiw;

                Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
                //Pose without correction
                vNonCorrectedSim3[pKFi]=g2oSiw;

                Sophus::SE3d Tic = Tiw*Twc;
                g2o::Sim3 g2oSim(Tic.unit_quaternion(),Tic.translation(),1.0);
                g2oCorrectedSiw = g2oSim*mg2oMergeScw;
                vCorrectedSim3[pKFi]=g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                double s = g2oCorrectedSiw.scale();

                pKFi->mfScale = s;

                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / s);

                pKFi->mTcwBefMerge = pKFi->GetPose();
                pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

                pKFi->SetPose(correctedTiw.cast<float>());

                if(pCurrentMap->isImuInitialized())
                {
                    Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
                    pKFi->SetVelocity(Rcor * pKFi->GetVelocity()); // TODO: should add here scale s
                }

            }
            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad()|| pMPi->GetMap() != pCurrentMap)
                    continue;

                KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
                g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
                g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());

                pMPi->UpdateNormalAndDepth();
            }
        }

        // Optimize graph (and update the loop position for each element form the begining to the end)
        if(mpTracker->mSensor != System::MONOCULAR)
        {
            Optimizer::OptimizeEssentialGraph(mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }


        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

            //std::cout << "Merge outside KFs: " << vpCurrentMapKFs.size() << std::endl;
            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }
                //std::cout << "KF id: " << pKFi->mnId << std::endl;

                // Make sure connections are updated
                pKFi->UpdateMap(pMergeMap);
                pMergeMap->AddKeyFrame(pKFi);
                pCurrentMap->EraseKeyFrame(pKFi);
            }

            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad())
                    continue;

                pMPi->UpdateMap(pMergeMap);
                pMergeMap->AddMapPoint(pMPi);
                pCurrentMap->EraseMapPoint(pMPi);
            }
        }
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOptEss = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOptEss - time_EndWeldingBA).count();
    vdMergeOptEss_ms.push_back(timeOptEss);
#endif


    if(bRelaunchBA && (!pCurrentMap->isImuInitialized() || mpAtlas->CountMaps()==1))
    {
        std::cout << "[KF " << mpCurrentKF->mnId << "] GBA begin" << std::endl;
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        RunGlobalBundleAdjustment(pMergeMap, mpCurrentKF->mnId);
        std::cout << "[KF " << mpCurrentKF->mnId << "] GBA map update finished" << std::endl;
    }

    mpMergeMatchedKF->AddMergeEdge(mpCurrentKF);
    mpCurrentKF->AddMergeEdge(mpMergeMatchedKF);

    pCurrentMap->IncreaseChangeIndex();
    pMergeMap->IncreaseChangeIndex();

    mpAtlas->RemoveBadMaps();

}


void LoopClosing::MergeLocal2()
{
    //cout << "Merge detected!!!!" << endl;

    int numTemporalKFs = 11; //TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    // NonCorrectedSim3[mpCurrentKF]=mg2oLoopScw;

    assert(!mpLocalMapper->HasPendingKeyFrames());

    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    {
        float s_on = mSold_new.scale();
        Sophus::SE3f T_on(mSold_new.rotation().cast<float>(), mSold_new.translation().cast<float>());

        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        //cout << "updating active map to merge reference" << endl;
        //cout << "curr merge KF id: " << mpCurrentKF->mnId << endl;
        //cout << "curr tracking KF id: " << mpTracker->GetLastKeyFrame()->mnId << endl;
        bool bScaleVel=false;
        if(s_on!=1)
            bScaleVel=true;
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(T_on,s_on,bScaleVel);
        mpTracker->UpdateFrameIMU(s_on,mpCurrentKF->GetImuBias(),mpTracker->GetLastKeyFrame());

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    }

    const int numKFnew=pCurrentMap->KeyFramesInMap();

    if((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
       && !pCurrentMap->GetIniertialBA2()){
        // Map is not completly initialized
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.;
        ba << 0., 0., 0.;
        Optimizer::InertialOptimization(pCurrentMap,bg,ba);
        IMU::Bias b (ba[0],ba[1],ba[2],bg[0],bg[1],bg[2]);
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        mpTracker->UpdateFrameIMU(1.0f,b,mpTracker->GetLastKeyFrame());

        // Set map initialized
        pCurrentMap->SetIniertialBA2();
        pCurrentMap->SetIniertialBA1();
        pCurrentMap->SetImuInitialized();

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    // Load KFs and MPs from merge map
    //cout << "updating current map" << endl;
    {
        // Get Merge Map Mutex (This section stops tracking!!)
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map


        vector<KeyFrame*> vpMergeMapKFs = pMergeMap->GetAllKeyFrames();
        vector<MapPoint*> vpMergeMapMPs = pMergeMap->GetAllMapPoints();


        for(KeyFrame* pKFi : vpMergeMapKFs)
        {
            if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergeMap)
            {
                continue;
            }

            // Make sure connections are updated
            pKFi->UpdateMap(pCurrentMap);
            pCurrentMap->AddKeyFrame(pKFi);
            pMergeMap->EraseKeyFrame(pKFi);
        }

        for(MapPoint* pMPi : vpMergeMapMPs)
        {
            if(!pMPi || pMPi->isBad() || pMPi->GetMap() != pMergeMap)
                continue;

            pMPi->UpdateMap(pCurrentMap);
            pCurrentMap->AddMapPoint(pMPi);
            pMergeMap->EraseMapPoint(pMPi);
        }

        // Save non corrected poses (already merged maps)
        vector<KeyFrame*> vpKFs = pCurrentMap->GetAllKeyFrames();
        for(KeyFrame* pKFi : vpKFs)
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            NonCorrectedSim3[pKFi]=g2oSiw;
        }
    }

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end updating current map" << endl;

    // Critical zone
    //bool good = pCurrentMap->CheckEssentialGraph();
    /*if(!good)
        cout << "BAD ESSENTIAL GRAPH!!" << endl;*/

    //cout << "Update essential graph" << endl;
    // mpCurrentKF->UpdateConnections(); // to put at false mbFirstConnection
    pMergeMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpMergeMatchedKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpMergeMatchedKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpMergeMatchedKF->ChangeParent(mpCurrentKF);
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();
        pNewChild->ChangeParent(pNewParent);
        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end update essential graph" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 1!!" << endl;*/

    //cout << "Update relationship between KFs" << endl;
    vector<MapPoint*> vpCheckFuseMapPoint; // MapPoint vector from current map to allow to fuse duplicated points with the old map (merge)
    vector<KeyFrame*> vpCurrentConnectedKFs;

    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vector<KeyFrame*> aux = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.insert(mvpMergeConnectedKFs.end(), aux.begin(), aux.end());
    if (mvpMergeConnectedKFs.size()>6)
        mvpMergeConnectedKFs.erase(mvpMergeConnectedKFs.begin()+6,mvpMergeConnectedKFs.end());
    /*mvpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);*/

    mpCurrentKF->UpdateConnections();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    /*vpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);*/
    aux = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end());
    if (vpCurrentConnectedKFs.size()>6)
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin()+6,vpCurrentConnectedKFs.end());

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
        if(spMapPointMerge.size()>1000)
            break;
    }

    /*cout << "vpCurrentConnectedKFs.size() " << vpCurrentConnectedKFs.size() << endl;
    cout << "mvpMergeConnectedKFs.size() " << mvpMergeConnectedKFs.size() << endl;
    cout << "spMapPointMerge.size() " << spMapPointMerge.size() << endl;*/


    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));
    //cout << "Finished to update relationship between KFs" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 2!!" << endl;*/

    //cout << "start SearchAndFuse" << endl;
    SearchAndFuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint);
    //cout << "end SearchAndFuse" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 3!!" << endl;

    cout << "Init to update connections" << endl;*/


    for(KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    //cout << "end update connections" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 4!!" << endl;*/

    // TODO Check: If new map is too small, we suppose that not informaiton can be propagated from new to old map
    if (numKFnew<10){
        return;
    }

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 5!!" << endl;*/

    // Perform BA
    bool bStopFlag=false;
    KeyFrame* pCurrKF = mpTracker->GetLastKeyFrame();
    //cout << "start MergeInertialBA" << endl;
    Optimizer::MergeInertialBA(pCurrKF, mpMergeMatchedKF, &bStopFlag, pCurrentMap,CorrectedSim3);
    //cout << "end MergeInertialBA" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 6!!" << endl;*/

    return;
}

void LoopClosing::CheckObservations(set<KeyFrame*> &spKFsMap1, set<KeyFrame*> &spKFsMap2)
{
    cout << "----------------------" << endl;
    for(KeyFrame* pKFi1 : spKFsMap1)
    {
        map<KeyFrame*, int> mMatchedMP;
        set<MapPoint*> spMPs = pKFi1->GetMapPoints();

        for(MapPoint* pMPij : spMPs)
        {
            if(!pMPij || pMPij->isBad())
            {
                continue;
            }

            map<KeyFrame*, tuple<int,int>> mMPijObs = pMPij->GetObservations();
            for(KeyFrame* pKFi2 : spKFsMap2)
            {
                if(mMPijObs.find(pKFi2) != mMPijObs.end())
                {
                    if(mMatchedMP.find(pKFi2) != mMatchedMP.end())
                    {
                        mMatchedMP[pKFi2] = mMatchedMP[pKFi2] + 1;
                    }
                    else
                    {
                        mMatchedMP[pKFi2] = 1;
                    }
                }
            }

        }

        if(mMatchedMP.size() == 0)
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has not any matched MP with the other map" << endl;
        }
        else
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has matched MP with " << mMatchedMP.size() << " KF from the other map" << endl;
            for(pair<KeyFrame*, int> matchedKF : mMatchedMP)
            {
                cout << "   -KF: " << matchedKF.first->mnId << ", Number of matches: " << matchedKF.second << endl;
            }
        }
    }
    cout << "----------------------" << endl;
}


void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    //cout << "[FUSE]: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE: Intially there are " << CorrectedPosesMap.size() << " KFs" << endl;
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKFi = mit->first;
        Map* pMap = pKFi->GetMap();

        g2o::Sim3 g2oScw = mit->second;
        Sophus::Sim3f Scw = Converter::toSophus(g2oScw);

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        int numFused = matcher.Fuse(pKFi,Scw,vpMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {


                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);

            }
        }

        total_replaces += num_replaces;
    }
    //cout << "[FUSE]: " << total_replaces << " MPs had been fused" << endl;
}


void LoopClosing::SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    //cout << "FUSE-POSE: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE-POSE: Intially there are " << vConectedKFs.size() << " KFs" << endl;
    for(auto mit=vConectedKFs.begin(), mend=vConectedKFs.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKF = (*mit);
        Map* pMap = pKF->GetMap();
        Sophus::SE3f Tcw = pKF->GetPose();
        Sophus::Sim3f Scw(Tcw.unit_quaternion(),Tcw.translation());
        Scw.setScale(1.f);
        /*std::cout << "These should be zeros: " <<
            Scw.rotationMatrix() - Tcw.rotationMatrix() << std::endl <<
            Scw.translation() - Tcw.translation() << std::endl <<
            Scw.scale() - 1.f << std::endl;*/
        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,Scw,vpMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);
            }
        }
        /*cout << "FUSE-POSE: KF " << pKF->mnId << " ->" << num_replaces << " MPs fused" << endl;
        total_replaces += num_replaces;*/
    }
    //cout << "FUSE-POSE: " << total_replaces << " MPs had been fused" << endl;
}



void LoopClosing::ResetSynchronously()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }
    ResetIfRequested();
}

void LoopClosing::ResetActiveMapSynchronously(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }
    ResetIfRequested();
}

void LoopClosing::RequestReset()
{
    ResetSynchronously();
}

void LoopClosing::RequestResetActiveMap(Map *pMap)
{
    ResetActiveMapSynchronously(pMap);
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;  //TODO old variable, it is not use in the new algorithm
        mbResetRequested=false;
        mbResetActiveMapRequested = false;
        ResetTagLoopState();
        ClearTagLoopReferences("reset");
    }
    else if(mbResetActiveMapRequested)
    {

        for (list<KeyFrame*>::const_iterator it=mlpLoopKeyFrameQueue.begin(); it != mlpLoopKeyFrameQueue.end();)
        {
            KeyFrame* pKFi = *it;
            if(pKFi->GetMap() == mpMapToReset)
            {
                it = mlpLoopKeyFrameQueue.erase(it);
            }
            else
                ++it;
        }

        mLastLoopKFid=mpAtlas->GetLastInitKFid(); //TODO old variable, it is not use in the new algorithm
        mbResetActiveMapRequested=false;
        ResetTagLoopState();
        ClearTagLoopReferences("reset_active_map");

    }
}

void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)
{  
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartFGBA = std::chrono::steady_clock::now();

    nFGBA_exec += 1;

    vnGBAKFs.push_back(pActiveMap->GetAllKeyFrames().size());
    vnGBAMPs.push_back(pActiveMap->GetAllMapPoints().size());
#endif

    const bool bImuInit = pActiveMap->isImuInitialized();

    if(!bImuInit)
        Optimizer::GlobalBundleAdjustemnt(pActiveMap,10,&mbStopGBA,nLoopKF,false);
    else
    {
        const tag::TagLocalBAParams gbaTagParams =
            (mbTagLoopDetected && mTagLocalBAParams.enable) ? mTagLocalBAParams
                                                            : tag::TagLocalBAParams();
        const int fixedLoopTagId = mbTagLoopDetected ? mnLoopTagId : -1;
        Optimizer::FullInertialBA(pActiveMap,7,false,nLoopKF,&mbStopGBA,false,1e2,1e6,NULL,NULL,
                                  gbaTagParams, fixedLoopTagId);
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndGBA = std::chrono::steady_clock::now();

    double timeGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndGBA - time_StartFGBA).count();
    vdGBA_ms.push_back(timeGBA);

    if(mbStopGBA)
    {
        nFGBA_abort += 1;
    }
#endif

    // Update all MapPoints and KeyFrames
    {
        unique_lock<mutex> lock(mMutexGBA);

        if(!bImuInit && pActiveMap->isImuInitialized())
            return;

        if(!mbStopGBA)
        {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);
            ClearTagLoopReferences("gba");

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);
            // cout << "LC: Update Map Mutex adquired" << endl;

            //pActiveMap->PrintEssentialGraph();
            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(),pActiveMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                //cout << "---Updating KF " << pKF->mnId << " with " << sChilds.size() << " childs" << endl;
                //cout << " KF mnBAGlobalForKF: " << pKF->mnBAGlobalForKF << endl;
                Sophus::SE3f Twc = pKF->GetPoseInverse();
                //cout << "Twc: " << Twc << endl;
                //cout << "GBA: Correct KeyFrames" << endl;
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(!pChild || pChild->isBad())
                        continue;

                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        //cout << "++++New child with flag " << pChild->mnBAGlobalForKF << "; LoopKF: " << nLoopKF << endl;
                        //cout << " child id: " << pChild->mnId << endl;
                        Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                        //cout << "Child pose: " << Tchildc << endl;
                        //cout << "pKF->mTcwGBA: " << pKF->mTcwGBA << endl;
                        pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;

                        Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                        if(pChild->isVelocitySet()){
                            pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                        }
                        else
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);


                        //cout << "Child bias: " << pChild->GetImuBias() << endl;
                        pChild->mBiasGBA = pChild->GetImuBias();


                        pChild->mnBAGlobalForKF = nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                //cout << "-------Update pose" << endl;
                pKF->mTcwBefGBA = pKF->GetPose();
                //cout << "pKF->mTcwBefGBA: " << pKF->mTcwBefGBA << endl;
                pKF->SetPose(pKF->mTcwGBA);
                /*cv::Mat Tco_cn = pKF->mTcwBefGBA * pKF->mTcwGBA.inv();
                cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
                double dist = cv::norm(trasl);
                cout << "GBA: KF " << pKF->mnId << " had been moved " << dist << " meters" << endl;
                double desvX = 0;
                double desvY = 0;
                double desvZ = 0;
                if(pKF->mbHasHessian)
                {
                    cv::Mat hessianInv = pKF->mHessianPose.inv();

                    double covX = hessianInv.at<double>(3,3);
                    desvX = std::sqrt(covX);
                    double covY = hessianInv.at<double>(4,4);
                    desvY = std::sqrt(covY);
                    double covZ = hessianInv.at<double>(5,5);
                    desvZ = std::sqrt(covZ);
                    pKF->mbHasHessian = false;
                }
                if(dist > 1)
                {
                    cout << "--To much distance correction: It has " << pKF->GetConnectedKeyFrames().size() << " connected KFs" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(80).size() << " connected KF with 80 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(50).size() << " connected KF with 50 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(20).size() << " connected KF with 20 common matches or more" << endl;

                    cout << "--STD in meters(x, y, z): " << desvX << ", " << desvY << ", " << desvZ << endl;


                    string strNameFile = pKF->mNameFile;
                    cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

                    cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

                    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
                    int num_MPs = 0;
                    for(int i=0; i<vpMapPointsKF.size(); ++i)
                    {
                        if(!vpMapPointsKF[i] || vpMapPointsKF[i]->isBad())
                        {
                            continue;
                        }
                        num_MPs += 1;
                        string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                        cv::circle(imLeft, pKF->mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
                        cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    }
                    cout << "--It has " << num_MPs << " MPs matched in the map" << endl;

                    string namefile = "./test_GBA/GBA_" + to_string(nLoopKF) + "_KF" + to_string(pKF->mnId) +"_D" + to_string(dist) +".png";
                    cv::imwrite(namefile, imLeft);
                }*/


                if(pKF->bImu)
                {
                    //cout << "-------Update inertial values" << endl;
                    pKF->mVwbBefGBA = pKF->GetVelocity();
                    //if (pKF->mVwbGBA.empty())
                    //    Verbose::PrintMess("pKF->mVwbGBA is empty", Verbose::VERBOSITY_NORMAL);

                    //assert(!pKF->mVwbGBA.empty());
                    pKF->SetVelocity(pKF->mVwbGBA);
                    pKF->SetNewBias(pKF->mBiasGBA);                    
                }

                lpKFtoCheck.pop_front();
            }

            //cout << "GBA: Correct MapPoints" << endl;
            // Correct MapPoints
            const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    /*if(pRefKF->mTcwBefGBA.empty())
                        continue;*/

                    // Map to non-corrected camera
                    // cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    // cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

                    // Backproject using corrected camera
                    pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
                }
            }

            // 选择参考帧，回环global BA后更新Tag位姿
            const std::vector<Map::MapTagPtr> vpTags = pActiveMap->GetAllMapTags();
            for(const Map::MapTagPtr &pTagPtr : vpTags)
            {
                tag::MapTagData *pTag = pTagPtr.get();
                if(!pTag || pTag->IsBad() || !pTag->HasPose())
                    continue;
                if(mbTagLoopDetected && (pTag == mpLoopTag || pTag->Id() == mnLoopTagId))
                    continue;
                if(pTag->IsFixed())
                    continue;

                if(pTag->mnBAGlobalForKF == nLoopKF)
                {
                    pTag->SetPose(pTag->mTwtGBA);
                    continue;
                }

                KeyFrame *pRefKF = pTag->SelectReferenceKeyFrame();
                if(!pRefKF || pRefKF->isBad() || pRefKF->mnBAGlobalForKF != nLoopKF)
                    continue;
                const Sophus::SE3f TwtOld = pTag->GetPose();
                pTag->SetPose(pRefKF->GetPoseInverse() * pRefKF->mTcwBefGBA * TwtOld);
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndUpdateMap = std::chrono::steady_clock::now();

            double timeUpdateMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_EndGBA).count();
            vdUpdateMap_ms.push_back(timeUpdateMap);

            double timeFGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_StartFGBA).count();
            vdFGBATotal_ms.push_back(timeFGBA);
#endif
            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    // cout << "LC: Finish requested" << endl;
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
