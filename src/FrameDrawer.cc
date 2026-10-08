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

#include "FrameDrawer.h"
#include "Tracking.h"

#ifdef HAS_APRILTAG
#include "ua_tag/AprilTagVisualizer.h"
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <mutex>
#include <sstream>

namespace ORB_SLAM3
{

FrameDrawer::FrameDrawer(Atlas* pAtlas):both(false),mpAtlas(pAtlas),mbDrawTags(false),mbFisheyeStereo(false)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mImRight = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    vector<pair<cv::Point2f, cv::Point2f> > vTracks;
    int state; // Tracking state
    vector<float> vCurrentDepth;
    float thDepth;

    Frame currentFrame;
    vector<MapPoint*> vpLocalMap;
    vector<cv::KeyPoint> vMatchesKeys;
    vector<MapPoint*> vpMatchedMPs;
    vector<cv::KeyPoint> vOutlierKeys;
    vector<MapPoint*> vpOutlierMPs;
    map<long unsigned int, cv::Point2f> mProjectPoints;
    map<long unsigned int, cv::Point2f> mMatchedInImage;
    bool bDrawTags = false;
    tag::TagFrameData tagFrameData;

    cv::Scalar standardColor(0,255,0);
    cv::Scalar odometryColor(255,0,0);

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);
        bDrawTags = mbDrawTags;
        if(bDrawTags)
            tagFrameData = mTagFrameData;

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
            vTracks = mvTracks;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;

            currentFrame = mCurrentFrame;
            vpLocalMap = mvpLocalMap;
            vMatchesKeys = mvMatchedKeys;
            vpMatchedMPs = mvpMatchedMPs;
            vOutlierKeys = mvOutlierKeys;
            vpOutlierMPs = mvpOutlierMPs;
            mProjectPoints = mmProjectPoints;
            mMatchedInImage = mmMatchedInImage;

            vCurrentDepth = mvCurrentDepth;
            thDepth = mThDepth;

        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    }

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED)
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }
                cv::line(im,pt1,pt2,standardColor);
            }
        }
        for(vector<pair<cv::Point2f, cv::Point2f> >::iterator it=vTracks.begin(); it!=vTracks.end(); it++)
        {
            cv::Point2f pt1,pt2;
            if(imageScale != 1.f)
            {
                pt1 = (*it).first / imageScale;
                pt2 = (*it).second / imageScale;
            }
            else
            {
                pt1 = (*it).first;
                pt2 = (*it).second;
            }
            cv::line(im,pt1,pt2, standardColor,5);
        }

    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                cv::Point2f point;
                if(imageScale != 1.f)
                {
                    point = vCurrentKeys[i].pt / imageScale;
                    float px = vCurrentKeys[i].pt.x / imageScale;
                    float py = vCurrentKeys[i].pt.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;
                }
                else
                {
                    point = vCurrentKeys[i].pt;
                    pt1.x=vCurrentKeys[i].pt.x-r;
                    pt1.y=vCurrentKeys[i].pt.y-r;
                    pt2.x=vCurrentKeys[i].pt.x+r;
                    pt2.y=vCurrentKeys[i].pt.y+r;
                }

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,standardColor);
                    cv::circle(im,point,2,standardColor,-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,odometryColor);
                    cv::circle(im,point,2,odometryColor,-1);
                    mnTrackedVO++;
                }
            }
        }
    }

#ifdef HAS_APRILTAG
    if(bDrawTags)
        ua_tag::OverlayTags(im, tagFrameData, tag::CameraId::LEFT_OR_MONO, imageScale);
#endif

    if(both)
        cv::putText(im, "L", cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawRightFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state
    bool bDrawTags = false;
    bool bFisheyeStereo = false;
    tag::TagFrameData tagFrameData;
    vector<cv::KeyPoint> vLeftKeys;
    vector<float> vuRight;

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mImRight.copyTo(im);
        bDrawTags = mbDrawTags;
        bFisheyeStereo = mbFisheyeStereo;
        if(bDrawTags)
            tagFrameData = mTagFrameData;

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vLeftKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
            vuRight = mvuRight;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeysRight;
        }
    } // destroy scoped mutex -> release mutex

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }

                cv::line(im,pt1,pt2,cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {
        const float r = 5;
        const cv::Scalar mapColor(0,255,0);
        const cv::Scalar voColor(255,0,0);

        if(bFisheyeStereo && !vbVO.empty() && !vbMap.empty())
        {
            // 鱼眼双目：右目特征有独立 MapPoint 槽位
            mnTracked=0;
            mnTrackedVO=0;
            const int n = static_cast<int>(vCurrentKeys.size());
            const int Nleft = static_cast<int>(vLeftKeys.size());

            for(int i=0;i<n;i++)
            {
                const int idx = i + Nleft;
                if(idx >= static_cast<int>(vbVO.size()) || idx >= static_cast<int>(vbMap.size()))
                    break;
                if(vbVO[idx] || vbMap[idx])
                {
                    cv::Point2f pt1,pt2;
                    cv::Point2f point;
                    if(imageScale != 1.f)
                    {
                        point = vCurrentKeys[i].pt / imageScale;
                        float px = vCurrentKeys[i].pt.x / imageScale;
                        float py = vCurrentKeys[i].pt.y / imageScale;
                        pt1.x=px-r;
                        pt1.y=py-r;
                        pt2.x=px+r;
                        pt2.y=py+r;
                    }
                    else
                    {
                        point = vCurrentKeys[i].pt;
                        pt1.x=vCurrentKeys[i].pt.x-r;
                        pt1.y=vCurrentKeys[i].pt.y-r;
                        pt2.x=vCurrentKeys[i].pt.x+r;
                        pt2.y=vCurrentKeys[i].pt.y+r;
                    }

                    if(vbMap[idx])
                    {
                        cv::rectangle(im,pt1,pt2,mapColor);
                        cv::circle(im,point,2,mapColor,-1);
                        mnTracked++;
                    }
                    else
                    {
                        cv::rectangle(im,pt1,pt2,voColor);
                        cv::circle(im,point,2,voColor,-1);
                        mnTrackedVO++;
                    }
                }
            }
        }
        else if(!bFisheyeStereo)
        {
            // PinHole 经典双目：右目无独立 MP 槽；用左目跟踪点 + mvuRight 画立体匹配
            // 先淡画全部右目 ORB，再高亮与左目 Map/VO 对应的立体点
            for(size_t i = 0; i < vCurrentKeys.size(); ++i)
            {
                cv::Point2f point = vCurrentKeys[i].pt;
                if(imageScale != 1.f)
                    point *= (1.f / imageScale);
                cv::circle(im, point, 1, cv::Scalar(180, 180, 180), -1, cv::LINE_AA);
            }

            const int nLeft = static_cast<int>(vLeftKeys.size());
            const int nFlags = std::min(static_cast<int>(vbMap.size()),
                                        std::min(static_cast<int>(vbVO.size()), nLeft));
            const int nU = std::min(static_cast<int>(vuRight.size()), nFlags);
            for(int i = 0; i < nU; ++i)
            {
                if(vuRight[i] < 0.f)
                    continue;
                if(!vbMap[i] && !vbVO[i])
                    continue;

                cv::Point2f point(vuRight[i], vLeftKeys[i].pt.y);
                if(imageScale != 1.f)
                    point *= (1.f / imageScale);

                cv::Point2f pt1(point.x - r, point.y - r);
                cv::Point2f pt2(point.x + r, point.y + r);
                const cv::Scalar &color = vbMap[i] ? mapColor : voColor;
                cv::rectangle(im, pt1, pt2, color);
                cv::circle(im, point, 2, color, -1);
            }
        }
    }

#ifdef HAS_APRILTAG
    if(bDrawTags)
        ua_tag::OverlayTags(im, tagFrameData, tag::CameraId::RIGHT, imageScale);
#endif

    cv::putText(im, "R", cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawTagDetectFrame(float imageScale)
{
    cv::Mat imLeft, imRight;
    tag::TagFrameData tagFrameData;
    bool bDrawTags = false;
    int state = Tracking::NO_IMAGES_YET;

    {
        unique_lock<mutex> lock(mMutex);
        state = mState;
        bDrawTags = mbDrawTags;
        mImTagDetect.copyTo(imLeft);
        mImTagDetectRight.copyTo(imRight);
        if(bDrawTags)
            tagFrameData = mTagFrameData;
    }

    if(imLeft.empty() && imRight.empty())
        return cv::Mat();

    auto preparePanel = [&](cv::Mat &im, tag::CameraId cam, const char *label)
    {
        if(im.empty())
            return;
        if(imageScale != 1.f && imageScale > 0.f)
        {
            const int w = std::max(1, static_cast<int>(im.cols / imageScale));
            const int h = std::max(1, static_cast<int>(im.rows / imageScale));
            cv::resize(im, im, cv::Size(w, h));
        }
        if(im.channels() < 3)
            cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);
#ifdef HAS_APRILTAG
        if(bDrawTags)
            ua_tag::OverlayTags(im, tagFrameData, cam, imageScale);
#else
        (void)cam;
        (void)bDrawTags;
#endif
        cv::putText(im, label, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    };

    preparePanel(imLeft, tag::CameraId::LEFT_OR_MONO, "Tag L");
    preparePanel(imRight, tag::CameraId::RIGHT, "Tag R");

    cv::Mat vis;
    if(!imLeft.empty() && !imRight.empty())
    {
        if(imLeft.rows != imRight.rows)
        {
            const int h = std::max(imLeft.rows, imRight.rows);
            if(imLeft.rows != h)
                cv::copyMakeBorder(imLeft, imLeft, 0, h - imLeft.rows, 0, 0,
                                   cv::BORDER_CONSTANT);
            if(imRight.rows != h)
                cv::copyMakeBorder(imRight, imRight, 0, h - imRight.rows, 0, 0,
                                   cv::BORDER_CONSTANT);
        }
        cv::hconcat(imLeft, imRight, vis);
    }
    else
        vis = imLeft.empty() ? imRight : imLeft;

    std::stringstream s;
    s << " TAG DETECT (detector input";
    s << ", gamma if enabled) | L=" << tagFrameData.left.size()
      << " R=" << tagFrameData.right.size();
    if(state==Tracking::NOT_INITIALIZED)
        s << " | TRYING TO INITIALIZE";
    else if(state==Tracking::OK)
        s << " | SLAM";
    else if(state==Tracking::LOST)
        s << " | LOST";

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);
    cv::Mat imText(vis.rows+textSize.height+10, vis.cols, vis.type());
    vis.copyTo(imText.rowRange(0,vis.rows).colRange(0,vis.cols));
    imText.rowRange(vis.rows,imText.rows) = cv::Mat::zeros(textSize.height+10, vis.cols, vis.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,
                cv::Scalar(255,255,255),1,8);
    return imText;
}

void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nMaps = mpAtlas->CountMaps();
        int nKFs = mpAtlas->KeyFramesInMap();
        int nMPs = mpAtlas->MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    mThDepth = pTracker->mCurrentFrame.mThDepth;
    mvCurrentDepth = pTracker->mCurrentFrame.mvDepth;

    mbDrawTags = pTracker->mbTagShowInOrbViewer;
    mImTagDetect.release();
    mImTagDetectRight.release();
    if(mbDrawTags)
    {
        mTagFrameData = pTracker->mCurrentFrame.mTagFrameData;
        pTracker->mCurrentFrame.mImTagDetect.copyTo(mImTagDetect);
        pTracker->mCurrentFrame.mImTagDetectRight.copyTo(mImTagDetectRight);
    }
    else
        mTagFrameData.Clear();

    // PinHole 经典双目 Nleft==-1：mvpMapPoints 仅覆盖左目；鱼眼双目才是 left+right
    mbFisheyeStereo = (pTracker->mCurrentFrame.Nleft != -1);

    if(both){
        mvCurrentKeysRight = pTracker->mCurrentFrame.mvKeysRight;
        pTracker->mImRight.copyTo(mImRight);
        mvuRight = pTracker->mCurrentFrame.mvuRight;
        if(mbFisheyeStereo)
            N = mvCurrentKeys.size() + mvCurrentKeysRight.size();
        else
            N = mvCurrentKeys.size();
    }
    else{
        N = mvCurrentKeys.size();
        mvuRight.clear();
    }

    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    //Variables for the new visualization
    mCurrentFrame = pTracker->mCurrentFrame;
    mmProjectPoints = mCurrentFrame.mmProjectPoints;
    mmMatchedInImage.clear();

    mvpLocalMap = pTracker->GetLocalMapMPS();
    mvMatchedKeys.clear();
    mvMatchedKeys.reserve(N);
    mvpMatchedMPs.clear();
    mvpMatchedMPs.reserve(N);
    mvOutlierKeys.clear();
    mvOutlierKeys.reserve(N);
    mvpOutlierMPs.clear();
    mvpOutlierMPs.reserve(N);

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        const int nMapPts = static_cast<int>(pTracker->mCurrentFrame.mvpMapPoints.size());
        const int nLoop = std::min(N, nMapPts);
        const int nLeftKeys = static_cast<int>(mvCurrentKeys.size());
        for(int i=0;i<nLoop;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;

                    if(i < nLeftKeys)
                        mmMatchedInImage[pMP->mnId] = mvCurrentKeys[i].pt;
                    else if(mbFisheyeStereo)
                        mmMatchedInImage[pMP->mnId] = mvCurrentKeysRight[i - nLeftKeys].pt;
                }
                else
                {
                    mvpOutlierMPs.push_back(pMP);
                    if(i < nLeftKeys)
                        mvOutlierKeys.push_back(mvCurrentKeys[i]);
                    else if(mbFisheyeStereo)
                        mvOutlierKeys.push_back(mvCurrentKeysRight[i - nLeftKeys]);
                }
            }
        }

    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM
