#include "ua_tag/MapTagData.h"

#include "KeyFrame.h"

#include <algorithm>
#include <vector>

namespace ORB_SLAM3 {
namespace tag {

namespace {

std::array<Eigen::Vector3f, 4> CornersFromPose(const Sophus::SE3f &T_wt,
                                               float tag_size)
{
    std::array<Eigen::Vector3f, 4> corners{};
    if(tag_size <= 0.0f)
        return corners;

    // 与 OpenCV IPPE_SQUARE / BuildSquareObjectPoints 约定一致
    const float h = tag_size * 0.5f;
    const Eigen::Vector3f pts_t[4] = {
        Eigen::Vector3f(-h,  h, 0.f),
        Eigen::Vector3f( h,  h, 0.f),
        Eigen::Vector3f( h, -h, 0.f),
        Eigen::Vector3f(-h, -h, 0.f)};

    for(int i = 0; i < 4; ++i)
        corners[i] = T_wt * pts_t[i];
    return corners;
}

}  // namespace

int MapTagData::Id() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mnId;
}

void MapTagData::SetId(int id)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mnId = id;
}

float MapTagData::GetTagSize() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTagSize;
}

void MapTagData::SetTagSize(float tag_size)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTagSize = tag_size;
}

void MapTagData::SetPose(const Sophus::SE3f &T_wt)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTwt = T_wt;
    mbHasPose = true;
}

Sophus::SE3f MapTagData::GetPose() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTwt;
}

bool MapTagData::HasPose() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mbHasPose;
}

void MapTagData::SetMirrorPose(const Sophus::SE3f &T_wt)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTwtMirror = T_wt;
    mbHasMirrorPose = true;
}

Sophus::SE3f MapTagData::GetMirrorPose() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTwtMirror;
}

bool MapTagData::HasMirrorPose() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mbHasMirrorPose;
}

std::array<Eigen::Vector3f, 4> MapTagData::GetWorldCorners() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    if(!mbHasPose)
        return {};
    return CornersFromPose(mTwt, mTagSize);
}

bool MapTagData::HasWorldCorners() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mbHasPose && mTagSize > 0.0f;
}

std::array<Eigen::Vector3f, 4> MapTagData::GetMirrorWorldCorners() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    if(!mbHasMirrorPose)
        return {};
    return CornersFromPose(mTwtMirror, mTagSize);
}

bool MapTagData::HasMirrorWorldCorners() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mbHasMirrorPose && mTagSize > 0.0f;
}

void MapTagData::SetState(MapTagState state)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mState = state;
}

MapTagState MapTagData::GetState() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mState;
}

bool MapTagData::IsFixed() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mState == MapTagState::FIXED_ANCHOR;
}

bool MapTagData::IsBad() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mState == MapTagState::BAD;
}

Map *MapTagData::GetMap() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mpMap;
}

void MapTagData::SetMapInternal(Map *pMap)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mpMap = pMap;
}

void MapTagData::AddObservationInternal(KeyFrame *pKF, int leftIndex, int rightIndex)
{
    if(!pKF)
        return;
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    KeyFrameObservation obs;
    obs.leftIndex = leftIndex;
    obs.rightIndex = rightIndex;
    mObservations[pKF] = obs;
}

void MapTagData::EraseObservationInternal(KeyFrame *pKF)
{
    if(!pKF)
        return;
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mObservations.erase(pKF);
}

bool MapTagData::IsInKeyFrame(KeyFrame *pKF) const
{
    if(!pKF)
        return false;
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations.find(pKF) != mObservations.end();
}

MapTagData::KeyFrameObservations MapTagData::GetObservations() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

std::size_t MapTagData::Observations() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations.size();
}

void MapTagData::SetFactorVizWeight(const FactorVizWeight &w)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mFactorVizWeight = w;
}

MapTagData::FactorVizWeight MapTagData::GetFactorVizWeight() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mFactorVizWeight;
}

KeyFrame *MapTagData::SelectReferenceKeyFrame(int min_inlier_corners,
                                             bool allow_unresolved_fallback) const
{
    KeyFrameObservations observations;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        observations = mObservations;
    }

    if(min_inlier_corners < 1)
        min_inlier_corners = 1;
    if(min_inlier_corners > 4)
        min_inlier_corners = 4;

    struct Candidate
    {
        KeyFrame *pKF = nullptr;
        bool resolved = false;
        bool opt_quality = false;
        int n_tracked = 0;
    };
    std::vector<Candidate> candidates;

    auto observationAt = [](KeyFrame *pKF, int index, bool right) -> const TagObservation * {
        if(!pKF || index < 0)
            return nullptr;
        const TagFrameData::Observations &vec =
            right ? pKF->mTagFrameData.right : pKF->mTagFrameData.left;
        if(index >= static_cast<int>(vec.size()))
            return nullptr;
        return &vec[static_cast<size_t>(index)];
    };

    auto countCornerInliers = [](const TagObservation &obs) -> int {
        int nIn = 0;
        for(bool outlier : obs.corner_outliers)
        {
            if(!outlier)
                ++nIn;
        }
        return nIn;
    };

    for(const auto &kv : observations)
    {
        KeyFrame *pKF = kv.first;
        if(!pKF || pKF->isBad())
            continue;

        bool detect_valid = false;
        bool resolved = false;
        bool opt_quality = false;

        const TagObservation *cams[2] = {
            observationAt(pKF, kv.second.leftIndex, false),
            observationAt(pKF, kv.second.rightIndex, true)};
        for(const TagObservation *obs : cams)
        {
            if(!obs || !obs->IsDetectValid())
                continue;
            detect_valid = true;

            // 第二层：已消歧
            if(obs->IsAmbiguityResolved() && obs->pose_estimate.has_value() &&
               obs->pose_estimate->Selected() != nullptr)
                resolved = true;

            // 第三层：优化内点质量
            if(!obs->is_opt_outlier && countCornerInliers(*obs) >= min_inlier_corners)
                opt_quality = true;
        }

        // 第一层：至少一目检测有效，否则该 KF 不能作为参考
        if(!detect_valid)
            continue;

        Candidate c;
        c.pKF = pKF;
        c.resolved = resolved;
        c.opt_quality = opt_quality;
        c.n_tracked = pKF->TrackedMapPoints(1);
        candidates.push_back(c);
    }

    if(candidates.empty())
        return nullptr;

    // 第二层：存在已消歧参考 KF 时，不再从未消歧 KF 中选择
    bool any_resolved = false;
    for(const Candidate &c : candidates)
    {
        if(c.resolved)
        {
            any_resolved = true;
            break;
        }
    }
    if(any_resolved)
    {
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [](const Candidate &c) { return !c.resolved; }),
                         candidates.end());
    }
    else if(!allow_unresolved_fallback)
    {
        // 回环锚点：全部历史观测未消歧时不回退
        return nullptr;
    }

    // 第三层：若池中存在优化内点合格的 KF，则只在其中选择
    bool any_opt_quality = false;
    for(const Candidate &c : candidates)
    {
        if(c.opt_quality)
        {
            any_opt_quality = true;
            break;
        }
    }
    if(any_opt_quality)
    {
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [](const Candidate &c) { return !c.opt_quality; }),
                         candidates.end());
    }

    // TODO 第四层：几何质量（IPPE 重投影误差、Tag 面积、左右目一致、
    // decision_margin、hamming）。当前空出，不参与排序。

    // 第五层：同等 Tag 质量下用 ORB 地图稳定性排序
    KeyFrame *best = nullptr;
    int best_tracked = -1;
    for(const Candidate &c : candidates)
    {
        if(c.n_tracked > best_tracked)
        {
            best_tracked = c.n_tracked;
            best = c.pKF;
        }
    }
    return best;
}

}  // namespace tag
}  // namespace ORB_SLAM3
