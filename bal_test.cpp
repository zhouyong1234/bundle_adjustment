#include <iostream>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "bundle_adjustment.h"

using namespace Sophus;
using namespace Eigen;
using namespace std;


class Observation
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Observation(int mpt_id, int cam_id, const Eigen::Vector2d& ob) :mpt_id_(mpt_id), cam_id_(cam_id), ob_(ob) {}
    
    int mpt_id_;
    int cam_id_;
    Eigen::Vector2d ob_;
};


void createData(int n_mappoints, int n_cameras, double fx, double fy, double cx, double cy, double height, double width, std::vector<Eigen::Vector3d>& mappoints, std::vector<Sophus::SE3d>& cameras, std::vector<Observation>& observations);

void addNoise(std::vector<Eigen::Vector3d>& mappoints, std::vector<Sophus::SE3d>& cameras, std::vector<Observation>& observations, double mpt_noise, double cam_trans_noise, double cam_rot_noise, double ob_noise);



int main(int argc, char **argv)
{

    const int n_mappoints = 1000;
    const int n_cameras = 6;

    const double fx = 525.0;
    const double fy = 525.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const double height = 640;
    const double width = 480;

    std::cout << "Start create data..." << std::endl;
    std::vector<Eigen::Vector3d> mappoints;
    std::vector<Sophus::SE3d> cameras;
    std::vector<Observation> observations;
    createData(n_mappoints, n_cameras, fx, fy, cx, cy, height, width, mappoints, cameras, observations);

    std::cout << "Total mappoints: " << mappoints.size() << " cameras: " << cameras.size() << " observations: " << observations.size() << std::endl;

    std::cout << "\n**** Start motion only BA test ****\n";

    double mpt_noise = 0.01;
    double cam_trans_noise = 0.1;
    double cam_rot_noise = 0.1;
    double ob_noise = 1.0;

    std::vector<Eigen::Vector3d> noise_mappoints;
    noise_mappoints = mappoints;
    std::vector<Sophus::SE3d> noise_cameras;
    noise_cameras = cameras;
    std::vector<Observation> noise_observations;
    noise_observations = observations;

    addNoise(noise_mappoints, noise_cameras, noise_observations, mpt_noise, cam_trans_noise, cam_rot_noise, ob_noise);

    // std::cout << "Total mappoints: " << noise_mappoints.size() << " cameras: " << noise_cameras.size() << " observations: " << noise_observations.size() << std::endl;

    BundleAdjustment ba;
    ba.setConvergenceCondition(100, 1e-5, 1e-10);
    ba.setVerbose(true);

    for(size_t i = 0; i < noise_mappoints.size(); i++)
    {
        const Eigen::Vector3d& npt = noise_mappoints.at(i);
        MapPoint* mpt = new MapPoint(npt, i);
        mpt->setFixed();
        ba.addMapPoint(mpt);
    }

    for(size_t i = 0; i < noise_cameras.size(); i++)
    {
        const Sophus::SE3d& ncam = noise_cameras.at(i);
        Camera* cam = new Camera(ncam, i);
        ba.addCamera(cam);
    }

    for(size_t i = 0; i < noise_observations.size(); i++)
    {
        const Observation& ob = noise_observations.at(i);
        MapPoint* mpt = ba.getMapPoint(ob.mpt_id_);
        Camera* cam = ba.getCamera(ob.cam_id_);
        CostFunction* cost_func = new CostFunction(mpt, cam, fx, fy, cx, cy, ob.ob_);
        ba.addCostFunction(cost_func);
    }

    ba.optimize();

    double sum_rot_error = 0.0;
    double sum_trans_error = 0.0;
    for(size_t i = 0; i < cameras.size(); i++)
    {
        Camera* cam = ba.getCamera(i);
        const Sophus::SE3d& opt_pose = cam->getPose();
        const Sophus::SE3d& org_pose = cameras.at(i);
        Sophus::SE3d pose_err = opt_pose * org_pose.inverse();
        sum_rot_error += pose_err.so3().log().norm();
        sum_trans_error += pose_err.translation().norm();

    }

    std::cout << "Mean rot error: " << sum_rot_error / (double)(cameras.size()) << "\tMean trans error: " << sum_trans_error / (double)(cameras.size()) << std::endl;

    std::cout << "\n**** Start struct only BA test ****\n";
    mpt_noise = 0.1;
    cam_trans_noise = 0.0;
    cam_rot_noise = 0.0;
    ob_noise = 1.0;

    noise_mappoints = mappoints;
    noise_cameras = cameras;
    noise_observations = observations;
    addNoise(noise_mappoints, noise_cameras, noise_observations, mpt_noise, cam_trans_noise, cam_rot_noise, ob_noise);

    BundleAdjustment ba_sba;
    ba_sba.setConvergenceCondition(20, 1e-5, 1e-10);
    ba_sba.setVerbose(true);

    for(size_t i = 0; i < noise_mappoints.size(); i++)
    {
        const Eigen::Vector3d& npt = noise_mappoints.at(i);
        MapPoint* mpt = new MapPoint(npt, i);
        ba_sba.addMapPoint(mpt);
    }

    for(size_t i = 0; i < noise_cameras.size(); i++)
    {
        const Sophus::SE3d& ncam = noise_cameras.at(i);
        Camera* cam = new Camera(ncam, i);
        cam->setFixed();
        ba_sba.addCamera(cam);
    }

    for(size_t i = 0; i < noise_observations.size(); i++)
    {
        const Observation& ob = noise_observations.at(i);
        MapPoint* mpt = ba_sba.getMapPoint(ob.mpt_id_);
        Camera* cam = ba_sba.getCamera(ob.cam_id_);
        CostFunction* cost_func = new CostFunction(mpt, cam, fx, fy, cx, cy, ob.ob_);
        ba_sba.addCostFunction(cost_func);
    }

    ba_sba.optimize();

    // compute point error
    double sum_point_error = 0.0;
    for(size_t i = 0; i < mappoints.size(); i++)
    {
        MapPoint* mpt = ba_sba.getMapPoint(i);
        const Eigen::Vector3d& opt_mpt = mpt->getPosition();
        const Eigen::Vector3d& org_mpt = mappoints.at(i);
        sum_point_error += (opt_mpt - org_mpt).norm();
    }

    std::cout << "Mean point error: " << sum_point_error / (double)(mappoints.size()) << std::endl;

    std::cout << "\n**** Start full BA test ****\n";
    mpt_noise = 0.05;
	cam_trans_noise = 0.1;
	cam_rot_noise = 0.1; 
	ob_noise = 1.0;

    noise_mappoints = mappoints;
	noise_cameras = cameras;
	noise_observations = observations;
	addNoise(noise_mappoints, noise_cameras, noise_observations, mpt_noise, cam_trans_noise, cam_rot_noise, ob_noise );

    BundleAdjustment full_ba;
    full_ba.setConvergenceCondition(20, 1e-5, 1e-10);
    full_ba.setVerbose(true);

    for(size_t i = 0; i < noise_mappoints.size(); i++)
    {
        const Eigen::Vector3d& npt = noise_mappoints.at(i);
        MapPoint* mpt = new MapPoint(npt, i);
        full_ba.addMapPoint(mpt);
    }

    for(size_t i = 0; i < noise_cameras.size(); i++)
    {
        const Sophus::SE3d& ncam = noise_cameras.at(i);
        Camera* cam = new Camera(ncam, i, i==0);
        full_ba.addCamera(cam);
    }

    for(size_t i = 0; i < noise_observations.size(); i ++)
	{
		const Observation& ob = noise_observations.at(i);
		MapPoint* mpt = full_ba.getMapPoint(ob.mpt_id_);
		Camera* cam = full_ba.getCamera(ob.cam_id_);
		CostFunction* cost_func = new CostFunction(mpt, cam, fx, fy, cx, cy, ob.ob_);
		full_ba.addCostFunction(cost_func);
	}

    full_ba.optimize();

    // Compute pose Error
	sum_rot_error = 0.0;
	sum_trans_error = 0.0;
	for(size_t i = 0; i < cameras.size(); i ++)
	{
		Camera* cam = full_ba.getCamera(i);
		const Sophus::SE3d& opt_pose = cam->getPose();
		const Sophus::SE3d& org_pose = cameras.at(i);
		Sophus::SE3d pose_err = opt_pose * org_pose.inverse();
		sum_rot_error += pose_err.so3().log().norm();
		sum_trans_error += pose_err.translation().norm();
	}
	std::cout << "Mean rot error: " << sum_rot_error / (double)(cameras.size())
	<< "\tMean trans error: " <<  sum_trans_error / (double)(cameras.size()) << std::endl;
	
	// Compute point Error
	sum_point_error = 0.0;
	for(size_t i = 0; i < mappoints.size(); i ++)
	{
		MapPoint* mpt = full_ba.getMapPoint(i);
		const Eigen::Vector3d& opt_mpt = mpt->getPosition();
		const Eigen::Vector3d& org_mpt = mappoints.at(i);
		sum_point_error += (opt_mpt - org_mpt).norm();
	}
	std::cout << "Mean point error: " << sum_point_error / (double)(mappoints.size())<< std::endl;

    return 0;
}


void createData(int n_mappoints, int n_cameras, double fx, double fy, double cx, double cy, double height, double width, std::vector<Eigen::Vector3d>& mappoints, std::vector<Sophus::SE3d>& cameras, std::vector<Observation>& observations)
{
    const double angle_range = 0.1;
    const double x_range = 1.0;
    const double y_range = 1.0;
    const double z_range = 0.5;

    const double x_min = -5.0;
    const double x_max = 5.0;
    const double y_min = -5.0;
    const double y_max = 5.0;
    const double z_min = 0.6;
    const double z_max = 8.0;

    cv::RNG rng(cv::getTickCount());

    // create cameras
    Eigen::Matrix3d Rx, Ry, Rz;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    for(int i = 0; i < n_cameras; i++)
    {
        // Rotation
        double tz = rng.uniform(-angle_range, angle_range);
        double ty = rng.uniform(-angle_range, angle_range);
        double tx = rng.uniform(-angle_range, angle_range);

        Rz << cos(tz), -sin(tz), 0,
              sin(tz),  cos(tz), 0,
              0,        0,       1;
        
        Ry << cos(ty), 0, sin(ty),
              0,       1,       0,
             -sin(ty), 0, cos(ty);

        Rx << 1,       0,        0,
              0, cos(tx), -sin(tx),
              0, sin(tx),  cos(tx);

        R = Rz * Ry * Rx;

        double x = rng.uniform(-x_range, x_range);
        double y = rng.uniform(-y_range, y_range);
        double z = rng.uniform(-z_range, z_range);

        t << x, y, z;

        // Eigen::Quaterniond q(R);
        Sophus::SE3d cam( R, t);
        // Sophus::SE3d cam(q, t);
        cameras.push_back(cam);
    }

    // create mappoints
    std::vector<Eigen::Vector3d> tmp_mappoints;
    for(int i = 0; i< n_mappoints; i++)
    {
        double x = rng.uniform(x_min, x_max);
        double y = rng.uniform(y_min, y_max);
        double z = rng.uniform(z_min, z_max);
        tmp_mappoints.push_back(Eigen::Vector3d(x,y,z));
    }

    // select good mappoints
    for(int i = 0; i< n_mappoints; i++)
    {
        const Eigen::Vector3d& ptw = tmp_mappoints.at(i);
        int n_obs = 0;
        for(int nc = 0; nc < n_cameras; nc++)
        {
            const Sophus::SE3d& cam_pose = cameras.at(nc);
            // project ptw to image
            const Eigen::Vector3d ptc = cam_pose * ptw;
            Eigen::Vector2d uv(fx * ptc[0]/ptc[2] + cx, fy * ptc[1]/ptc[2] + cy);

            if(uv[0] < 0 || uv[1] < 0 || uv[0] >= width || uv[1] >=height || ptc[2] < 0.1)
            {
                continue;
            }
            n_obs++;
        }

        if(n_obs < 2)
        {
            continue;
        }

        mappoints.push_back(ptw);
    }

    // create observations
    for(size_t i = 0; i < mappoints.size(); i++)
    {
        const Eigen::Vector3d& ptw = mappoints.at(i);
        for(int nc = 0; nc < n_cameras; nc++)
        {
            const Sophus::SE3d& cam_pose = cameras.at(nc);
            const Eigen::Vector3d ptc = cam_pose * ptw;
            Eigen::Vector2d uv(fx*ptc[0]/ptc[2]+cx, fy*ptc[1]/ptc[2]+cy);

            Observation ob(i, nc, uv);
            observations.push_back(ob);
        }
    }

    mappoints.shrink_to_fit();
    cameras.shrink_to_fit();
    observations.shrink_to_fit();
}


void addNoise(std::vector<Eigen::Vector3d>& mappoints, std::vector<Sophus::SE3d>& cameras, std::vector<Observation>& observations, double mpt_noise, double cam_trans_noise, double cam_rot_noise, double ob_noise)
{
    cv::RNG rng(cv::getTickCount());

    for(size_t i = 0; i < mappoints.size(); i++)
    {
        double nx = rng.gaussian(mpt_noise);
        double ny = rng.gaussian(mpt_noise);
        double nz = rng.gaussian(mpt_noise);
        mappoints.at(i) += Eigen::Vector3d(nx, ny, nz);
    }

    Eigen::Matrix3d Rx, Ry, Rz;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    for(size_t i = 0; i < cameras.size(); i++)
    {
        if(i == 0)
        {
            continue;
        }

        double tz = rng.gaussian(cam_rot_noise);
        double ty = rng.gaussian(cam_rot_noise);
        double tx = rng.gaussian(cam_rot_noise);

        Rz << cos(tz), -sin(tz), 0,
              sin(tz),  cos(tz), 0,
              0,        0,       1;
        
        Ry << cos(ty), 0, sin(ty),
              0,       1,       0,
             -sin(ty), 0, cos(ty);

        Rx << 1,       0,        0,
              0, cos(tx), -sin(tx),
              0, sin(tx),  cos(tx);

        R = Rz * Ry * Rx;

        double x = rng.gaussian(cam_trans_noise);
        double y = rng.gaussian(cam_trans_noise);
        double z = rng.gaussian(cam_trans_noise);

        t << x, y, z;

        Sophus::SE3d cam_noise(R, t);
        cameras.at(i) *= cam_noise;
    }

    for(size_t i = 0; i < observations.size(); i++)
    {
        double x = rng.gaussian(ob_noise);
        double y = rng.gaussian(ob_noise);
        observations.at(i).ob_ += Eigen::Vector2d(x, y);
    }
}




