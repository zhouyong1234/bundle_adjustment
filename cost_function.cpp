#include "cost_function.h"

CostFunction::CostFunction(MapPoint* map_point, Camera* camera, double fx, double fy, double cx, double cy, const Eigen::Vector2d& ob_z) : map_point_(map_point), camera_(camera), fx_(fx), fy_(fy), cx_(cx), cy_(cy), ob_z_(ob_z), huber_b_(1.0), info_matrix_(Eigen::Matrix2d::Identity()) {}


void CostFunction::setHuberParameter(double b)
{
    huber_b_ = b;
}

void CostFunction::setCovariance(const Eigen::Matrix2d& cov)
{
    info_matrix_ = cov.inverse();
}

void CostFunction::computeInterVars(Eigen::Vector2d& e, Eigen::Matrix2d& weighted_info, double& weighted_e2)
{
    const Eigen::Vector3d ptc = camera_->getPose() * map_point_->getPosition();
    const Eigen::Vector2d u(fx_ * ptc[0]/ptc[2]+cx_, fy_ * ptc[1]/ptc[2]+cy_);

    e = ob_z_ - u;

    double et_info_e = e.transpose() * info_matrix_ * e;

    double weight = 1.0;
    double sqrt_ete = sqrt(et_info_e);
    if(sqrt_ete > huber_b_)
    {
        weight = 2*huber_b_*sqrt_ete - huber_b_ * huber_b_;
        weight = weight / et_info_e;
    }

    weighted_info = weight * info_matrix_;
    weighted_e2 = weight * et_info_e;
}

void CostFunction::computeJT(Eigen::Matrix<double, 2, 6>& JT)
{
    const Eigen::Vector3d pt = camera_->getPose() * map_point_->getPosition();
    const double& x = pt(0);
    const double& y = pt(1);
    const double& z = pt(2);
    JT.setZero();
    const double z2 = z*z;

    JT(0,0) = fx_ / z;
    JT(0,1) = 0;
    JT(0,2) = -fx_*x/z2;
    JT(0,3) = -fx_*x*y/z2;
    JT(0,4) = fx_+fx_*x*x/z2;
    JT(0,5) = -fx_*y/z;

    JT(1,0) = 0;
    JT(1,1) = fy_/z;
    JT(1,2) = -fy_*y/z2;
    JT(1,3) = -fy_-fy_*y*y/z2;
    JT(1,4) = fy_*x*y/z2;
    JT(1,5) = fy_*x/z;

    JT = -JT;
}

void CostFunction::computeJX(Eigen::Matrix<double, 2, 3>& JX)
{
    const Eigen::Vector3d pt = camera_->getPose() * map_point_->getPosition();

    const double& x = pt(0);
    const double& y = pt(1);
    const double& z = pt(2);

    JX.setZero();

    Eigen::Matrix<double, 2, 3> Jtmp;
    Jtmp.setZero();
    Jtmp(0,0) = fx_ / z;
    Jtmp(0,1) = 0;
    Jtmp(0,2) = -fx_*x/(z*z);
    Jtmp(1,0) = 0;
    Jtmp(1,1) = fy_/z;
    Jtmp(1,2) = -fy_*y/(z*z);

    JX = -Jtmp * camera_->getPose().rotationMatrix();
}
