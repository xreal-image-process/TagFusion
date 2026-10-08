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

#include "OptimizableTypes.h"
#include "G2oTypes.h"

#include "CameraModels/Pinhole.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"

#include <cmath>
#include <iostream>
#include <memory>

namespace ORB_SLAM3 {

namespace {

inline Eigen::Matrix<double, 3, 6> SE3LeftPerturbDeriv(const Eigen::Vector3d &xyz)
{
    const double x = xyz(0);
    const double y = xyz(1);
    const double z = xyz(2);
    Eigen::Matrix<double, 3, 6> SE3deriv;
    SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f,
                -z, 0.f, x, 0.f, 1.f, 0.f,
                y, -x, 0.f, 0.f, 0.f, 1.f;
    return SE3deriv;
}

template <typename EdgeT>
bool CheckBinarySE3Jacobians(EdgeT *edge, double delta, double tol, double *max_abs_err)
{
    if(!edge || !edge->pCamera)
        return false;

    g2o::VertexSE3Expmap *v0 =
        static_cast<g2o::VertexSE3Expmap *>(edge->vertices()[0]);
    g2o::VertexSE3Expmap *v1 =
        static_cast<g2o::VertexSE3Expmap *>(edge->vertices()[1]);
    if(!v0 || !v1)
        return false;

    const g2o::SE3Quat T0 = v0->estimate();
    const g2o::SE3Quat T1 = v1->estimate();

    // 解析 Jacobian 需经 JacobianWorkspace 映射存储后再写 _jacobianOplus*
    // 派生类 override 了无参 linearizeOplus，需显式走基类带 workspace 的入口
    using BinaryBase =
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap,
                            g2o::VertexSE3Expmap>;
    g2o::JacobianWorkspace jw;
    jw.updateSize(edge);
    jw.allocate();
    static_cast<BinaryBase *>(edge)->linearizeOplus(jw);
    const Eigen::Matrix<double, 2, 6> Ja = edge->jacobianOplusXi();
    const Eigen::Matrix<double, 2, 6> Jb = edge->jacobianOplusXj();

    Eigen::Matrix<double, 2, 6> Ja_num = Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Matrix<double, 2, 6> Jb_num = Eigen::Matrix<double, 2, 6>::Zero();

    for(int i = 0; i < 6; ++i)
    {
        Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
        xi(i) = delta;
        v0->setEstimate(g2o::SE3Quat::exp(xi) * T0);
        edge->computeError();
        const Eigen::Vector2d e_plus = edge->error();
        xi(i) = -delta;
        v0->setEstimate(g2o::SE3Quat::exp(xi) * T0);
        edge->computeError();
        const Eigen::Vector2d e_minus = edge->error();
        Ja_num.col(i) = (e_plus - e_minus) / (2.0 * delta);
        v0->setEstimate(T0);
    }

    for(int i = 0; i < 6; ++i)
    {
        Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
        xi(i) = delta;
        v1->setEstimate(g2o::SE3Quat::exp(xi) * T1);
        edge->computeError();
        const Eigen::Vector2d e_plus = edge->error();
        xi(i) = -delta;
        v1->setEstimate(g2o::SE3Quat::exp(xi) * T1);
        edge->computeError();
        const Eigen::Vector2d e_minus = edge->error();
        Jb_num.col(i) = (e_plus - e_minus) / (2.0 * delta);
        v1->setEstimate(T1);
    }

    const double err_a = (Ja - Ja_num).cwiseAbs().maxCoeff();
    const double err_b = (Jb - Jb_num).cwiseAbs().maxCoeff();
    const double err = std::max(err_a, err_b);
    if(max_abs_err)
        *max_abs_err = std::max(*max_abs_err, err);
    return err < tol;
}

}  // namespace
    bool EdgeSE3ProjectXYZOnlyPose::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
        g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z,   -y, 1.f, 0.f, 0.f,
                     -z , 0.f, x, 0.f, 1.f, 0.f,
                     y ,  -x , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(xyz_trans) * SE3deriv;
    }

    bool EdgeSE3ProjectXYZOnlyPoseToBody::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPoseToBody::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZOnlyPoseToBody::linearizeOplus() {
        g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T_lw(vi->estimate());
        Eigen::Vector3d X_l = T_lw.map(Xw);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(Xw));

        double x_w = X_l[0];
        double y_w = X_l[1];
        double z_w = X_l[2];

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z_w,   -y_w, 1.f, 0.f, 0.f,
                -z_w , 0.f, x_w, 0.f, 1.f, 0.f,
                y_w ,  -x_w , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }

    EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>() {
    }

    bool EdgeSE3ProjectXYZ::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZ::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZ::linearizeOplus() {
        g2o::VertexSE3Expmap * vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T(vj->estimate());
        g2o::VertexSBAPointXYZ* vi = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector3d xyz = vi->estimate();
        Eigen::Vector3d xyz_trans = T.map(xyz);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        auto projectJac = -pCamera->projectJac(xyz_trans);

        _jacobianOplusXi =  projectJac * T.rotation().toRotationMatrix();

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z,   -y, 1.f, 0.f, 0.f,
                -z , 0.f, x, 0.f, 1.f, 0.f,
                y ,  -x , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXj = projectJac * SE3deriv;
    }

    EdgeSE3ProjectXYZToBody::EdgeSE3ProjectXYZToBody() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>() {
    }

    bool EdgeSE3ProjectXYZToBody::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZToBody::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZToBody::linearizeOplus() {
        g2o::VertexSE3Expmap * vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T_lw(vj->estimate());
        g2o::SE3Quat T_rw = mTrl * T_lw;
        g2o::VertexSBAPointXYZ* vi = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector3d X_w = vi->estimate();
        Eigen::Vector3d X_l = T_lw.map(X_w);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(X_w));

        _jacobianOplusXi =  -pCamera->projectJac(X_r) * T_rw.rotation().toRotationMatrix();

        double x = X_l[0];
        double y = X_l[1];
        double z = X_l[2];

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z,   -y, 1.f, 0.f, 0.f,
                -z , 0.f, x, 0.f, 1.f, 0.f,
                y ,  -x , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXj = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }


    VertexSim3Expmap::VertexSim3Expmap() : BaseVertex<7, g2o::Sim3>()
    {
        _marginalized=false;
        _fix_scale = false;
    }

    bool VertexSim3Expmap::read(std::istream& is)
    {
        g2o::Vector7d cam2world;
        for (int i=0; i<6; i++){
            is >> cam2world[i];
        }
        is >> cam2world[6];

        float nextParam;
        for(size_t i = 0; i < pCamera1->size(); i++){
            is >> nextParam;
            pCamera1->setParameter(nextParam,i);
        }

        for(size_t i = 0; i < pCamera2->size(); i++){
            is >> nextParam;
            pCamera2->setParameter(nextParam,i);
        }

        setEstimate(g2o::Sim3(cam2world).inverse());
        return true;
    }

    bool VertexSim3Expmap::write(std::ostream& os) const
    {
        g2o::Sim3 cam2world(estimate().inverse());
        g2o::Vector7d lv=cam2world.log();
        for (int i=0; i<7; i++){
            os << lv[i] << " ";
        }

        for(size_t i = 0; i < pCamera1->size(); i++){
            os << pCamera1->getParameter(i) << " ";
        }

        for(size_t i = 0; i < pCamera2->size(); i++){
            os << pCamera2->getParameter(i) << " ";
        }

        return os.good();
    }

    EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ() :
            g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeSim3ProjectXYZ::read(std::istream& is)
    {
        for (int i=0; i<2; i++)
        {
            is >> _measurement[i];
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSim3ProjectXYZ::write(std::ostream& os) const
    {
        for (int i=0; i<2; i++){
            os  << _measurement[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }

    EdgeInverseSim3ProjectXYZ::EdgeInverseSim3ProjectXYZ() :
            g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeInverseSim3ProjectXYZ::read(std::istream& is)
    {
        for (int i=0; i<2; i++)
        {
            is >> _measurement[i];
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeInverseSim3ProjectXYZ::write(std::ostream& os) const
    {
        for (int i=0; i<2; i++){
            os  << _measurement[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }

    EdgeSE3ProjectTagCorner::EdgeSE3ProjectTagCorner()
        : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>()
    {
    }

    bool EdgeSE3ProjectTagCorner::read(std::istream &is)
    {
        for(int i = 0; i < 2; i++)
            is >> _measurement[i];
        for(int i = 0; i < 2; i++)
            for(int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if(i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectTagCorner::write(std::ostream &os) const
    {
        for(int i = 0; i < 2; i++)
            os << measurement()[i] << " ";
        for(int i = 0; i < 2; i++)
            for(int j = i; j < 2; j++)
                os << " " << information()(i, j);
        return os.good();
    }

    void EdgeSE3ProjectTagCorner::linearizeOplus()
    {
        const g2o::VertexSE3Expmap *vTag =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const g2o::VertexSE3Expmap *vCam =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);

        const g2o::SE3Quat T_wt = vTag->estimate();
        const g2o::SE3Quat T_cw = vCam->estimate();
        const Eigen::Vector3d Xw = T_wt.map(X_t);
        const Eigen::Vector3d Xc = T_cw.map(Xw);

        const Eigen::Matrix<double, 2, 3> projectJac = -pCamera->projectJac(Xc);
        const Eigen::Matrix3d Rcw = T_cw.rotation().toRotationMatrix();

        _jacobianOplusXi = projectJac * Rcw * SE3LeftPerturbDeriv(Xw);
        _jacobianOplusXj = projectJac * SE3LeftPerturbDeriv(Xc);
    }

    bool EdgeSE3ProjectTagCorner::checkJacobiansNumerical(double delta, double tol) const
    {
        auto *self = const_cast<EdgeSE3ProjectTagCorner *>(this);
        double max_err = 0.0;
        return CheckBinarySE3Jacobians(self, delta, tol, &max_err);
    }

    EdgeSE3ProjectTagCornerToBody::EdgeSE3ProjectTagCornerToBody()
        : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>()
    {
    }

    bool EdgeSE3ProjectTagCornerToBody::read(std::istream &is)
    {
        for(int i = 0; i < 2; i++)
            is >> _measurement[i];
        for(int i = 0; i < 2; i++)
            for(int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if(i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectTagCornerToBody::write(std::ostream &os) const
    {
        for(int i = 0; i < 2; i++)
            os << measurement()[i] << " ";
        for(int i = 0; i < 2; i++)
            for(int j = i; j < 2; j++)
                os << " " << information()(i, j);
        return os.good();
    }

    void EdgeSE3ProjectTagCornerToBody::linearizeOplus()
    {
        const g2o::VertexSE3Expmap *vTag =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const g2o::VertexSE3Expmap *vCam =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);

        const g2o::SE3Quat T_wt = vTag->estimate();
        const g2o::SE3Quat T_lw = vCam->estimate();
        const g2o::SE3Quat T_rw = mTrl * T_lw;
        const Eigen::Vector3d Xw = T_wt.map(X_t);
        const Eigen::Vector3d Xl = T_lw.map(Xw);
        const Eigen::Vector3d Xr = mTrl.map(Xl);

        const Eigen::Matrix<double, 2, 3> projectJac = -pCamera->projectJac(Xr);
        const Eigen::Matrix3d Rrl = mTrl.rotation().toRotationMatrix();
        const Eigen::Matrix3d Rrw = T_rw.rotation().toRotationMatrix();

        _jacobianOplusXi = projectJac * Rrw * SE3LeftPerturbDeriv(Xw);
        _jacobianOplusXj = projectJac * Rrl * SE3LeftPerturbDeriv(Xl);
    }

    bool EdgeSE3ProjectTagCornerToBody::checkJacobiansNumerical(double delta,
                                                                double tol) const
    {
        auto *self = const_cast<EdgeSE3ProjectTagCornerToBody *>(this);
        double max_err = 0.0;
        return CheckBinarySE3Jacobians(self, delta, tol, &max_err);
    }

    bool TestTagCornerEdgeJacobians(double *max_abs_err)
    {
        if(max_abs_err)
            *max_abs_err = 0.0;

        g2o::SparseOptimizer optimizer;
        auto *linearSolver =
            new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        auto *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(solver_ptr));

        std::vector<float> cam_params = {458.654f, 457.296f, 367.215f, 248.375f};
        std::unique_ptr<Pinhole> cam(new Pinhole(cam_params));

        auto *vTag = new g2o::VertexSE3Expmap();
        auto *vCam = new g2o::VertexSE3Expmap();
        vTag->setId(0);
        vCam->setId(1);
        {
            Eigen::Quaterniond q =
                Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX());
            vTag->setEstimate(g2o::SE3Quat(q, Eigen::Vector3d(0.3, -0.1, 1.5)));
        }
        {
            Eigen::Quaterniond q =
                Eigen::AngleAxisd(-0.05, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY());
            vCam->setEstimate(g2o::SE3Quat(q, Eigen::Vector3d(0.05, 0.02, 0.1)));
        }
        optimizer.addVertex(vTag);
        optimizer.addVertex(vCam);

        const Eigen::Vector3d Xt(-0.08, 0.08, 0.0);
        Eigen::Vector2d z =
            cam->project(vCam->estimate().map(vTag->estimate().map(Xt)));
        z += Eigen::Vector2d(0.7, -0.4);

        auto *e = new EdgeSE3ProjectTagCorner();
        e->setVertex(0, vTag);
        e->setVertex(1, vCam);
        e->setMeasurement(z);
        e->setInformation(Eigen::Matrix2d::Identity());
        e->X_t = Xt;
        e->pCamera = cam.get();
        optimizer.addEdge(e);

        double err = 0.0;
        if(!CheckBinarySE3Jacobians(e, 1e-8, 1e-5, &err))
        {
            std::cerr << "[TagCornerJac] left edge failed, max_abs_err=" << err
                      << std::endl;
            if(max_abs_err)
                *max_abs_err = err;
            return false;
        }

        g2o::SE3Quat Trl(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(-0.12, 0.0, 0.0));
        Eigen::Vector2d zr =
            cam->project((Trl * vCam->estimate()).map(vTag->estimate().map(Xt)));
        zr += Eigen::Vector2d(-0.5, 0.3);

        auto *eBody = new EdgeSE3ProjectTagCornerToBody();
        eBody->setVertex(0, vTag);
        eBody->setVertex(1, vCam);
        eBody->setMeasurement(zr);
        eBody->setInformation(Eigen::Matrix2d::Identity());
        eBody->X_t = Xt;
        eBody->pCamera = cam.get();
        eBody->mTrl = Trl;
        optimizer.addEdge(eBody);

        double err_body = 0.0;
        if(!CheckBinarySE3Jacobians(eBody, 1e-8, 1e-5, &err_body))
        {
            std::cerr << "[TagCornerJac] body edge failed, max_abs_err="
                      << err_body << std::endl;
            if(max_abs_err)
                *max_abs_err = std::max(err, err_body);
            return false;
        }

        if(max_abs_err)
            *max_abs_err = std::max(err, err_body);
        return true;
    }

    bool TestTagCornerInertialEdgeJacobians(double *max_abs_err)
    {
        if(max_abs_err)
            *max_abs_err = 0.0;

        std::vector<float> cam_params = {458.654f, 457.296f, 367.215f, 248.375f};
        std::unique_ptr<Pinhole> cam0(new Pinhole(cam_params));
        std::unique_ptr<Pinhole> cam1(new Pinhole(cam_params));

        auto fillImuCam = [&](int n_cams) -> ImuCamPose {
            ImuCamPose imu;
            imu.its = 0;
            Eigen::Quaterniond qwb =
                Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitY());
            imu.Rwb = qwb.toRotationMatrix();
            imu.twb = Eigen::Vector3d(0.04, -0.03, 0.02);

            imu.Rcb.resize(n_cams);
            imu.tcb.resize(n_cams);
            imu.Rbc.resize(n_cams);
            imu.tbc.resize(n_cams);
            imu.Rcw.resize(n_cams);
            imu.tcw.resize(n_cams);
            imu.pCamera.resize(n_cams);
            imu.bf = 0.0;

            imu.Rcb[0] = Eigen::Matrix3d::Identity();
            imu.tcb[0] = Eigen::Vector3d(0.05, 0.02, 0.01);
            imu.pCamera[0] = cam0.get();
            if(n_cams > 1)
            {
                imu.Rcb[1] = Eigen::Matrix3d::Identity();
                imu.tcb[1] = Eigen::Vector3d(0.05 - 0.12, 0.02, 0.01);
                imu.pCamera[1] = cam1.get();
            }
            for(int i = 0; i < n_cams; ++i)
            {
                imu.Rbc[i] = imu.Rcb[i].transpose();
                imu.tbc[i] = -imu.Rbc[i] * imu.tcb[i];
            }
            const Eigen::Matrix3d Rbw = imu.Rwb.transpose();
            const Eigen::Vector3d tbw = -Rbw * imu.twb;
            for(int i = 0; i < n_cams; ++i)
            {
                imu.Rcw[i] = imu.Rcb[i] * Rbw;
                imu.tcw[i] = imu.Rcb[i] * tbw + imu.tcb[i];
            }
            imu.Rwb0 = imu.Rwb;
            imu.DR.setIdentity();
            return imu;
        };

        auto *vTag = new g2o::VertexSE3Expmap();
        vTag->setId(0);
        {
            Eigen::Quaterniond q =
                Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX());
            vTag->setEstimate(g2o::SE3Quat(q, Eigen::Vector3d(0.3, -0.1, 1.5)));
        }

        const Eigen::Vector3d Xt(-0.08, 0.08, 0.0);
        const double delta = 1e-8;
        const double tol = 1e-5;

        auto checkCam = [&](int cam_idx, Pinhole *cam) -> bool {
            auto *vPose = new VertexPose();
            vPose->setId(1);
            vPose->setEstimate(fillImuCam(2));

            const Eigen::Vector3d Xw = vTag->estimate().map(Xt);
            Eigen::Vector2d z = vPose->estimate().Project(Xw, cam_idx);
            z += Eigen::Vector2d(0.6, -0.35);

            auto *e = new EdgeTagCornerInertial(cam_idx);
            e->setVertex(0, vTag);
            e->setVertex(1, vPose);
            e->setMeasurement(z);
            e->setInformation(Eigen::Matrix2d::Identity());
            e->X_t = Xt;
            e->pCamera = cam;

            using BinaryBase =
                g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap, VertexPose>;
            g2o::JacobianWorkspace jw;
            jw.updateSize(e);
            jw.allocate();
            static_cast<BinaryBase *>(e)->linearizeOplus(jw);
            const Eigen::Matrix<double, 2, 6> Ja = e->jacobianOplusXi();
            const Eigen::Matrix<double, 2, 6> Jb = e->jacobianOplusXj();

            const g2o::SE3Quat Ttag = vTag->estimate();
            const ImuCamPose Tpose = vPose->estimate();

            Eigen::Matrix<double, 2, 6> Ja_num = Eigen::Matrix<double, 2, 6>::Zero();
            Eigen::Matrix<double, 2, 6> Jb_num = Eigen::Matrix<double, 2, 6>::Zero();

            for(int i = 0; i < 6; ++i)
            {
                Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
                xi(i) = delta;
                vTag->setEstimate(g2o::SE3Quat::exp(xi) * Ttag);
                e->computeError();
                const Eigen::Vector2d e_plus = e->error();
                xi(i) = -delta;
                vTag->setEstimate(g2o::SE3Quat::exp(xi) * Ttag);
                e->computeError();
                const Eigen::Vector2d e_minus = e->error();
                Ja_num.col(i) = (e_plus - e_minus) / (2.0 * delta);
                vTag->setEstimate(Ttag);
            }

            for(int i = 0; i < 6; ++i)
            {
                Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
                xi(i) = delta;
                vPose->setEstimate(Tpose);
                vPose->oplusImpl(xi.data());
                e->computeError();
                const Eigen::Vector2d e_plus = e->error();
                xi(i) = -delta;
                vPose->setEstimate(Tpose);
                vPose->oplusImpl(xi.data());
                e->computeError();
                const Eigen::Vector2d e_minus = e->error();
                Jb_num.col(i) = (e_plus - e_minus) / (2.0 * delta);
                vPose->setEstimate(Tpose);
            }

            const double err_a = (Ja - Ja_num).cwiseAbs().maxCoeff();
            const double err_b = (Jb - Jb_num).cwiseAbs().maxCoeff();
            const double err = std::max(err_a, err_b);
            if(max_abs_err)
                *max_abs_err = std::max(*max_abs_err, err);
            if(err >= tol)
            {
                std::cerr << "[TagCornerInertialJac] cam_idx=" << cam_idx
                          << " failed, tag_err=" << err_a
                          << " pose_err=" << err_b << std::endl;
                return false;
            }
            return true;
        };

        if(!checkCam(0, cam0.get()) || !checkCam(1, cam1.get()))
            return false;
        return true;
    }

}
