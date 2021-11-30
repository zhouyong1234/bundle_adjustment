#include <iostream>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "common.h"

using namespace Sophus;
using namespace Eigen;
using namespace std;


typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double,  2, 12> Matrix212d;
typedef Eigen::Matrix<double,  12, 1> Vector12d;

struct Pose
{
    Pose() {}
    
    explicit Pose(double *data)
    {
        rotation = SO3d::exp(Vector3d(data[0], data[1], data[2]));
        translation = Vector3d(data[3], data[4], data[5]);
        focus = data[6];
        k1 = data[7];
        k2 = data[8];
    }

    void set_to(double *data)
    {
        auto r = rotation.log();
        for(int i = 0; i < 3; ++i) data[i] = r[i];
        for(int i = 0; i < 3; ++i) data[i+3] = translation[i];
        data[6] = focus;
        data[7] = k1;
        data[8] = k2;
    }


    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focus = 0;
    double k1 = 0, k2 = 0;
};



void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        std::cout << "Error" << std::endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}


void SolveBA(BALProblem &bal_problem)
{
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    const double *observations = bal_problem.observations();

    Matrix12d H = Matrix12d::Zero();
    Matrix212d J = Matrix212d::Zero();
    Vector12d b = Vector12d::Zero();
    Vector2d error;
    Vector2d predictions;
    double cost = 0, lastCost = 0;
    int count = 0;
    int iterations = 40;
    for(int iter = 0; iter < iterations; iter ++)
    {
        for(int i = 0; i < bal_problem.num_observations(); ++i)
        {
            double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
            double *point = points + point_block_size * bal_problem.point_index()[i];
            Vector3d pc = Pose(camera).rotation * Vector3d(point[0], point[1], point[2]) + Pose(camera).translation;

            double X = pc[0];
            double Y = pc[1];
            double Z = pc[2];

            pc = -pc / pc[2];
            double r2 = pc.squaredNorm();
            double distortion = 1.0 + r2 * (Pose(camera).k1 + Pose(camera).k2 * r2);
            predictions[0] = Pose(camera).focus * distortion * pc[0];
            predictions[1] = Pose(camera).focus * distortion * pc[1];
            error[0] = predictions[0] - observations[2*i+0];
            error[1] = predictions[1] - observations[2*i+1];
            

            double f = Pose(camera).focus;
            double k1 = Pose(camera).k1;
            double k2 = Pose(camera).k2;

            double dis_f = f * (1.0 + k1 * r2 + k2 * r2 * r2);


            J(0,0) = dis_f / Z;
            J(0,1) = 0;
            J(0,2) = -dis_f * X / (Z * Z);
            J(0,3) = -dis_f * X * Y / (Z * Z);
            J(0,4) = dis_f + dis_f * X * X / (Z * Z);
            J(0,5) = -dis_f * Y / Z;
            J(0,6) = distortion * pc[0];
            J(0,7) = f * r2 * pc[0];
            J(0,8) = f * r2 * r2 * pc[0];
            J(0,9) = -dis_f / Z;
            J(0,10) = 0;
            J(0,11) = dis_f * X / (Z * Z);
            J(1,0) = 0;
            J(1,1) = dis_f / Z;
            J(1,2) = -dis_f * Y / (Z * Z);
            J(1,3) = -dis_f - dis_f * Y * Y / (Z * Z);
            J(1,4) = dis_f * X * Y / (Z * Z);
            J(1,5) = dis_f * X / Z;
            J(1,6) = distortion * pc[1];
            J(1,7) = f * r2 * pc[1];
            J(1,8) = f * r2 * r2 * pc[1];
            J(1,9) = 0;
            J(1,10) = -dis_f / Z;
            J(1,11) = dis_f * Y / (Z * Z);

            H += J.transpose() * J;
            b += -J.transpose() * error;
            cost += error.transpose() * error;
            count++;
        }

        Vector12d update = H.ldlt().solve(b);
        // std::cout << update.transpose() << std::endl;
        
        
        cost /= count;

        std::cout << cost << std::endl;

        if(isnan(update[0]))
        {
            cout << "update is nan" << endl;
            break;
        }

        if(iter > 0 && cost > lastCost)
        {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;

    }
}





