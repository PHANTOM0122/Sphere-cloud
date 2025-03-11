// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "relative_pose.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "PoseLib/solvers/gen_relpose_6pt.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_7pt.h"
#include "PoseLib/solvers/p6lp.h"

#include <iostream>

namespace poselib {

double findMedian(Eigen::VectorXd vec) {
    // Eigen::VectorXd의 요소를 std::vector로 복사합니다.
    std::vector<double> values(vec.data(), vec.data() + vec.size());

    // 값을 정렬합니다.
    std::sort(values.begin(), values.end());

    // 중앙값을 계산합니다.
    size_t n = values.size();
    if (n % 2 == 0) {
        // 짝수 개의 요소: 중앙에 위치한 두 값의 평균을 반환합니다.
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    } else {
        // 홀수 개의 요소: 중앙에 위치한 값을 반환합니다.
        return values[n / 2];
    }
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> findClosestPointsBetweenLines (const Eigen::Vector3d& P1, const Eigen::Vector3d& d1,
    const Eigen::Vector3d& P2, const Eigen::Vector3d& d2) {
    
    Eigen::Vector3d n = d1.cross(d2);
    Eigen::Matrix3d M;
    M.col(0) = d1;
    M.col(1) = -d2;
    M.col(2) = n;

    if (std::abs(n.norm()) < 1e-6) {
        std::cerr << "Lines are parallel or nearly parallel" << std::endl;
        return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    }

    Eigen::Vector3d b = P2 - P1;
    Eigen::Vector3d x = M.colPivHouseholderQr().solve(b.cross(n));

    Eigen::Vector3d Q1 = P1 + x(0) * d1;
    Eigen::Vector3d Q2 = P2 + x(1) * d2;

    return {Q1, Q2};
}

void RelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_5pt(x1s, x2s, models);
}

double RelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void RelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 5) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }
    refine_relpose(x1_inlier, x2_inlier, pose, bundle_opt);
}

void GeneralizedRelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    // 5+1 solver
    bool done = false;
    int pair0 = 0, pair1 = 1;
    while (!done) {
        pair0 = random_int(rng) % matches.size();
        if (matches[pair0].x1.size() < 5)
            continue;

        pair1 = random_int(rng) % matches.size();
        if (pair0 == pair1 || matches[pair1].x1.size() == 0)
            continue;

        done = true;
    }

    // Sample 5 points from the first camera pair
    CameraPose pose1 = rig1_poses[matches[pair0].cam_id1];
    CameraPose pose2 = rig2_poses[matches[pair0].cam_id2];
    Eigen::Vector3d p1 = pose1.center();
    Eigen::Vector3d p2 = pose2.center();
    draw_sample(5, matches[pair0].x1.size(), &sample, rng);
    for (size_t k = 0; k < 5; ++k) {
        x1s[k] = pose1.derotate(matches[pair0].x1[sample[k]].homogeneous().normalized());
        p1s[k] = p1;
        x2s[k] = pose2.derotate(matches[pair0].x2[sample[k]].homogeneous().normalized());
        p2s[k] = p2;
    }

    // Sample one point from the second camera pair
    pose1 = rig1_poses[matches[pair1].cam_id1];
    pose2 = rig2_poses[matches[pair1].cam_id2];
    p1 = pose1.center();
    p2 = pose2.center();
    size_t ind = random_int(rng) % matches[pair1].x1.size();
    x1s[5] = pose1.derotate(matches[pair1].x1[ind].homogeneous().normalized());
    p1s[5] = p1;
    x2s[5] = pose2.derotate(matches[pair1].x2[ind].homogeneous().normalized());
    p2s[5] = p2;

    gen_relpose_5p1pt(p1s, x1s, p2s, x2s, models);
}

double GeneralizedRelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {

    *inlier_count = 0;
    double cost = 0;
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose.t);
        pose2.q = quat_multiply(pose2.q, pose.q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        size_t local_inlier_count = 0;
        cost += compute_epipolar_msac_score(relpose, m.x1, m.x2, opt.max_epipolar_error * opt.max_epipolar_error, &local_inlier_count); // Using Epipolar distance
        *inlier_count += local_inlier_count;
    }

    return cost;
}

void GeneralizedRelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    std::vector<PairwiseMatches> inlier_matches;
    inlier_matches.resize(matches.size());

    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose->t);
        pose2.q = quat_multiply(pose2.q, pose->q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        // Compute inliers
        std::vector<char> inliers;
        int num_inl = get_inliers_epipolar(relpose, m.x1, m.x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers); // Using Epipolar distance

        inlier_matches[match_k].cam_id1 = m.cam_id1;
        inlier_matches[match_k].cam_id2 = m.cam_id2;
        inlier_matches[match_k].x1.reserve(num_inl);
        inlier_matches[match_k].x2.reserve(num_inl);

        for (size_t k = 0; k < m.x1.size(); ++k) {
            if (inliers[k]) {
                inlier_matches[match_k].x1.push_back(m.x1[k]);
                inlier_matches[match_k].x2.push_back(m.x2[k]);
            }
        }
    }
    refine_generalized_relpose(inlier_matches, rig1_poses, rig2_poses, pose, bundle_opt);
}


void SpherePoseEstimator::generate_models(std::vector<CameraPose> *models) {

    draw_sample(3, points2D.size(), &sample, rng); // Select 3 points

    for (size_t k = 0; k < 3; ++k) {

        x1s[k] = lines3D[sample[k]];       // 3D normalized rays (center -> point direction)
        X2s[k] = points3D_rgbd[sample[k]]; // RGBD-points
    }

    p3p(x1s, X2s, models); // P3P solver with cheirality condition
    
    // Inverse pose
    for (size_t k = 0; k < models->size(); ++k) {
        CameraPose &pose = (*models)[k];

        CameraPose pose_inv;
        pose_inv.q = quat_conj(pose.q);
        pose_inv.t = -pose_inv.rotate(pose.t);
        
        // Re-assign to save pose as world->cam
        pose.q = pose_inv.q;
        pose.t = pose_inv.t;
    }
}

double SpherePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    
    *inlier_count = 0;
    double c_e_squared; // epipolar residual
    double c_d, c_d_squared; // Depth residual
    double cost_sum = 0.0; // sum of residual
    double cost = 0.0; // Total cost

    Eigen::Vector3d line3D_cam; // 3D line direction vectors defined in camera coordinate
    Eigen::Vector3d line3D_cam_point; // 3D line point offset
    Eigen::Vector3d Zero (0, 0, 0);
    
    // Essential matrix from pose
    Eigen::Matrix3d E; 
    essential_from_motion(pose, &E);

    // Inlier threshold 
    double r_sq_th = opt.max_epipolar_error * opt.max_epipolar_error; // threshold for line reprojection

    double z_sq_th = (0.1) * (0.1); // threshold for depth (beta)

    double sq_th = lambda1*r_sq_th + lambda2*z_sq_th; // total threshold

    for (size_t match_k = 0; match_k < points2D.size(); ++match_k) {

        /////////////////// Epipolar residual part /////////////////////
        Eigen::Vector3d l = lines3D[match_k] / abs(lines3D[match_k](2));            
       
        double C = points2D[match_k].homogeneous().dot(E * (l));
        double nJc_sq = (E.block<2, 3>(0, 0) * (l)).squaredNorm();
        c_e_squared = (C * C) / nJc_sq;

        // Conver 3D lines (world coordinate) into camera coordinate
        line3D_cam = pose.rotate(lines3D[match_k]);

        // Solve and find intersection
        Eigen::Matrix2d A;
        A << -line3D_cam.x(), points3D_rgbd[match_k].x(),
            -line3D_cam.z(), points3D_rgbd[match_k].z();
        Eigen::Vector2d b(pose.t(0), pose.t(2));

        // Solve At = b
        Eigen::Vector2d t = A.fullPivLu().solve(b);
        double beta_i = t(1);

        // Set depth residual
        c_d = (beta_i-1);
        c_d_squared = c_d * c_d;
    
        ////////////////////////////////////////////////////////////////////////////////////
        // Sum of reproejection cost, depth residual cost
        ////////////////////////////////////////////////////////////////////////////////////
        if (beta_i < 0){ 
            cost += sq_th;
        } else {
            cost_sum = lambda1*(c_e_squared) + lambda2*(c_d_squared);
            
            if (c_e_squared < r_sq_th && c_d_squared < z_sq_th)
            {
                // Check cheirality
                bool cheirality = check_cheirality_new(pose, lines3D[match_k], points2D[match_k].homogeneous().normalized(), 0.0);
                if (cheirality){
                    (*inlier_count)++;
                    cost += cost_sum;
                } else{ // Out of cheirality
                    cost += sq_th;
                }
            } else { // Out of inliers
                    cost += sq_th;
            }
        }
    }
    return cost;
}

void SpherePoseEstimator::refine_model(CameraPose *pose) const {
    
    // Bundle Adjustment options
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED; // Default is Cauchyloss
    bundle_opt.loss_scale = opt.max_epipolar_error; // This is considering reprojection
    bundle_opt.max_iterations = 25;

    // Inlier threshold 
    double c_e_squared; // epipolar residual
    double c_d, c_d_squared; // Depth residual
    double r_sq_th = 5 * (opt.max_epipolar_error * opt.max_epipolar_error); // generous threshold
    double z_sq_th = (0.1) * (0.1); // threshold for depth(beta): beta < 0.9 | beta > 1.1

    // World -> cam
    CameraPose relpose; 
    relpose.q = pose->q;
    relpose.t = pose->t;

    // Get Essential matrix from pose
    Eigen::Matrix3d E; 
    essential_from_motion(relpose, &E);    

    Eigen::Vector3d line3D_cam; // 3D line direction vectors defined in camera coordinate
    Eigen::Vector3d line3D_cam_point;  // 3D line point offset

    // Inlier sets
    std::vector<Eigen::Vector3d> points3D_inliers, lines3D_inliers, points3D_rgbd_inliers;
    std::vector<Eigen::Vector2d> points2D_inliers;
    
    for (size_t pt_k = 0; pt_k < points2D.size(); ++pt_k) {
    
        /////////////////// Epipolar residual part /////////////////////            
        Eigen::Vector3d l = lines3D[pt_k] / abs(lines3D[pt_k](2));      

        double C = points2D[pt_k].homogeneous().dot(E * (l));
        double nJc_sq = (E.block<2, 3>(0, 0) * (l)).squaredNorm();
        c_e_squared = (C * C) / nJc_sq;

        /////////////////// Depth residual part /////////////////////
        // Conver 3D lines (world coordinate) into camera coordinate
        line3D_cam = relpose.rotate(lines3D[pt_k]);

        // Solve and find intersection
        Eigen::Matrix2d A;
        A << -line3D_cam.x(), points3D_rgbd[pt_k].x(),
            -line3D_cam.z(), points3D_rgbd[pt_k].z();
        Eigen::Vector2d b(relpose.t(0), relpose.t(2));

        // Solve At = b
        Eigen::Vector2d t = A.fullPivLu().solve(b);
        double beta_i = t(1);

        // If beta is negative, skip
        if (beta_i < 0){
            continue;
        }

        // // Set residual
        c_d = (beta_i-1);
        c_d_squared = c_d * c_d;

        // Get inliers
        if( c_e_squared < r_sq_th && c_d_squared < z_sq_th)
        {            
            // Check cheirality
            bool cheirality = check_cheirality_new(relpose, lines3D[pt_k], points2D[pt_k].homogeneous().normalized(), 0.0);
            if (cheirality){
                // Append to inlier set!
                points3D_inliers.push_back(points3D[pt_k]);
                lines3D_inliers.push_back(lines3D[pt_k]);
                points2D_inliers.push_back(points2D[pt_k]);
                points3D_rgbd_inliers.push_back(points3D_rgbd[pt_k]);
            } else{ // Out of cheirality
                continue;
            }
        } else { // Out of inliers
            continue;
        }
    }
    refine_sphere_pose(points3D_inliers, lines3D_inliers, points2D_inliers, points3D_rgbd_inliers, pose, opt, bundle_opt, lambda1, lambda2);
}


void FundamentalEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_7pt(x1s, x2s, models);
}

double FundamentalEstimator::score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const {
    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void FundamentalEstimator::refine_model(Eigen::Matrix3d *F) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    refine_fundamental(x1, x2, F, bundle_opt);
}

} // namespace poselib
