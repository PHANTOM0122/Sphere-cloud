// Copyright (c) 2020, Viktor Larsson
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
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "essential.h"

#include <array>
namespace poselib {

void essential_from_motion(const CameraPose &pose, Eigen::Matrix3d *E) {
    *E << 0.0, -pose.t(2), pose.t(1), pose.t(2), 0.0, -pose.t(0), -pose.t(1), pose.t(0), 0.0;
    *E = (*E) * pose.R();
}

bool check_cheirality_origin(const CameraPose &pose, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2, double min_depth) {
    
    // This code assumes that x1 and x2 are unit vectors
    const Eigen::Vector3d Rx1 = pose.rotate(x1);

    // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
    // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)

    const double a = -Rx1.dot(x2);
    const double b1 = -Rx1.dot(pose.t);
    const double b2 = x2.dot(pose.t);

    // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
    const double lambda1 = b1 - a * b2;
    const double lambda2 = -a * b1 + b2;

    min_depth = min_depth * (1 - a * a);
    return lambda1 > min_depth && lambda2 > min_depth;
}

// Modified for p5+1R minimal solver for ray cloud. 
// Enforce cheirality constraint only for second(query) camera
bool check_cheirality(const CameraPose &pose, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2, double min_depth) {
    // This code assumes that x1 and x2 are unit vectors
    const Eigen::Vector3d Rx1 = pose.rotate(x1);

    // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
    // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)

    const double a = -Rx1.dot(x2);
    const double b1 = -Rx1.dot(pose.t);
    const double b2 = x2.dot(pose.t);

    // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
    // const double lambda1 = b1 - a * b2;
    const double lambda2 = -a * b1 + b2;

    min_depth = min_depth * (1 - a * a);
    // return lambda1 > min_depth && lambda2 > min_depth;
    return lambda2 > min_depth;
}

// P6L cheirality check
bool check_cheirality(const CameraPose &pose, const Eigen::Vector3d &p1, const Eigen::Vector3d &x1,
                      const Eigen::Vector3d &p2, const Eigen::Vector3d &x2, double min_depth) {

    // This code assumes that x1 and x2 are unit vectors
    const Eigen::Vector3d Rx1 = pose.rotate(x1);

    // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
    // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
    const Eigen::Vector3d rhs = pose.t + pose.rotate(p1) - p2;
    const double a = -Rx1.dot(x2);
    const double b1 = -Rx1.dot(rhs);
    const double b2 = x2.dot(rhs);

    // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
    const double lambda2 = -a * b1 + b2;

    min_depth = min_depth * (1 - a * a);
    return lambda2 > min_depth;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Modified cheirality for sphere line cloud
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool check_cheirality_new(const CameraPose &pose, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2, double min_depth) {
    
    // This code assumes that x1(3D lines) and x2(Rays from keypoints) are unit vectors
    Eigen::Vector3d v1 = pose.rotate(x1); // 3D Line in cam coordinate
    Eigen::Vector3d point1 = pose.apply(Eigen::Vector3d(0.0, 0.0, 0.0)); // Center of sphere cloud in camera coordinate
    Eigen::Vector3d v2 = x2; 
    Eigen::Vector3d point2 (0.0, 0.0, 0.0);

    Eigen::Matrix2d A;
    A << v1.dot(v1), -v1.dot(v2), 
         v1.dot(v2), -v2.dot(v2);
    
    Eigen::Vector2d b;
    b << (point2 - point1).dot(v1), (point2 - point1).dot(v2);

    // Solve 
    Eigen::Vector2d t = A.fullPivLu().solve(b);
    
    // The scale direction of line should be postiive (same as in front of camera)
    return t(0) > 0 && t(1) > 0;
}

bool check_cheirality_new(const CameraPose &pose, const Eigen::Vector3d &p1, const Eigen::Vector3d &x1,
                      const Eigen::Vector3d &p2, const Eigen::Vector3d &x2, double min_depth) {
    
    // This code assumes that x1 and x2 are unit vectors

    // Rotate
    const Eigen::Vector3d Rx1 = pose.rotate(x1);
    const Eigen::Vector3d Rp1 = pose.apply(p1);

    // Found the closest points between two vectors
    ////////////////////////////////////////////////////////////////////////////////////
    // Check cheirality 
    Eigen::Vector3d v1 = Rx1;
    Eigen::Vector3d point1 = Rp1;
    Eigen::Vector3d v2 = x2;
    Eigen::Vector3d point2 = p2;

    Eigen::Matrix2d A;
    A << v1.dot(v1), -v1.dot(v2), 
         v1.dot(v2), -v2.dot(v2);
    
    Eigen::Vector2d b;
    b << (point2 - point1).dot(v1), (point2 - point1).dot(v2);

    // Solve 
    Eigen::Vector2d t = A.fullPivLu().solve(b);
    
    // Eigen::Vector3d closestPointOnLine1 = point1 + t(0) * v1;
    // Eigen::Vector3d closestPointOnLine2 = point2 + t(1) * v2;

    return t(0) > 0 && t(1) > 0;
    ////////////////////////////////////////////////////////////////////////////////////
}


void motion_from_essential(const Eigen::Matrix3d &E, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2,
                           CameraPoseVector *relative_poses) {

    // Compute the necessary cross products
    Eigen::Vector3d u12 = E.col(0).cross(E.col(1));
    Eigen::Vector3d u13 = E.col(0).cross(E.col(2));
    Eigen::Vector3d u23 = E.col(1).cross(E.col(2));
    const double n12 = u12.squaredNorm();
    const double n13 = u13.squaredNorm();
    const double n23 = u23.squaredNorm();
    Eigen::Matrix3d UW;
    Eigen::Matrix3d Vt;

    // Compute the U*W factor
    if (n12 > n13) {
        if (n12 > n23) {
            UW.col(1) = E.col(0).normalized();
            UW.col(2) = u12 / std::sqrt(n12);
        } else {
            UW.col(1) = E.col(1).normalized();
            UW.col(2) = u23 / std::sqrt(n23);
        }
    } else {
        if (n13 > n23) {
            UW.col(1) = E.col(0).normalized();
            UW.col(2) = u13 / std::sqrt(n13);
        } else {
            UW.col(1) = E.col(1).normalized();
            UW.col(2) = u23 / std::sqrt(n23);
        }
    }
    UW.col(0) = -UW.col(2).cross(UW.col(1));

    // Compute the V factor
    Vt.row(0) = UW.col(1).transpose() * E;
    Vt.row(1) = -UW.col(0).transpose() * E;
    Vt.row(0).normalize();

    // Here v1 and v2 should be orthogonal. However, if E is not exactly an essential matrix they might not be
    // To ensure we end up with a rotation matrix we orthogonalize them again here, this should be a nop for good data
    Vt.row(1) -= Vt.row(0).dot(Vt.row(1)) * Vt.row(0);

    Vt.row(1).normalize();
    Vt.row(2) = Vt.row(0).cross(Vt.row(1));

    // Original cheirality check (both!)
    // poselib::CameraPose pose;
    // pose.q = rotmat_to_quat(UW * Vt);
    // pose.t = UW.col(2);
    // if (check_cheirality_origin(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }
    // pose.t = -pose.t;
    // if (check_cheirality_origin(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }
    // // U * W.transpose()
    // UW.block<3, 2>(0, 0) = -UW.block<3, 2>(0, 0);
    // pose.q = rotmat_to_quat(UW * Vt);
    // if (check_cheirality_origin(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }
    // pose.t = -pose.t;
    // if (check_cheirality_origin(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }

    ///////////////////////////////////////////////////////////
    // // For ray cloud
    poselib::CameraPose pose;
    pose.q = rotmat_to_quat(UW * Vt);
    pose.t = UW.col(2);
    if (check_cheirality(pose, x1, x2)) {
        relative_poses->emplace_back(pose);
    }
    pose.t = -pose.t;
    if (check_cheirality(pose, x1, x2)) {
        relative_poses->emplace_back(pose);
    }
    // U * W.transpose()
    UW.block<3, 2>(0, 0) = -UW.block<3, 2>(0, 0);
    pose.q = rotmat_to_quat(UW * Vt);
    if (check_cheirality(pose, x1, x2)) {
        relative_poses->emplace_back(pose);
    }
    pose.t = -pose.t;
    if (check_cheirality(pose, x1, x2)) {
        relative_poses->emplace_back(pose);
    }
    ///////////////////////////////////////////////////////////

    // if (check_cheirality_new(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }
    // pose.t = -pose.t;
    // if (check_cheirality_new(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }
    // // U * W.transpose()
    // UW.block<3, 2>(0, 0) = -UW.block<3, 2>(0, 0);
    // pose.q = rotmat_to_quat(UW * Vt);
    // if (check_cheirality_new(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }
    // pose.t = -pose.t;
    // if (check_cheirality_new(pose, x1, x2)) {
    //     relative_poses->emplace_back(pose);
    // }
}

void motion_from_essential_planar(double e01, double e21, double e10, double e12, const Eigen::Vector3d &x1,
                                  const Eigen::Vector3d &x2, poselib::CameraPoseVector *relative_poses) {

    Eigen::Vector2d z;
    z << -e01 * e10 - e21 * e12, -e21 * e10 + e01 * e12;
    z.normalize();

    CameraPose pose;
    Eigen::Matrix3d R;
    R << z(0), 0.0, -z(1), 0.0, 1.0, 0.0, z(1), 0.0, z(0);
    pose.q = rotmat_to_quat(R);
    pose.t << e21, 0.0, -e01;
    pose.t.normalize();

    if (check_cheirality(pose, x1, x2)) {
        relative_poses->push_back(pose);
    }
    pose.t = -pose.t;
    if (check_cheirality(pose, x1, x2)) {
        relative_poses->push_back(pose);
    }

    // There are two more flipped solutions where
    //    R = [a 0 b; 0 -1 0; b 0 -a]
    // These are probably not interesting in the planar case

    /*
            z << e01 * e10 - e21 * e12, e21* e10 + e01 * e12;
            z.normalize();
            pose.R << z(0), 0.0, z(1), 0.0, -1.0, 0.0, z(1), 0.0, -z(0);
            relative_poses->push_back(pose);
            pose.t = -pose.t;
            relative_poses->push_back(pose);
    */
}

void motion_from_essential_svd(const Eigen::Matrix3d &E, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2,
                               poselib::CameraPoseVector *relative_poses) {
    Eigen::JacobiSVD<Eigen::Matrix3d> USV(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = USV.matrixU();
    Eigen::Matrix3d Vt = USV.matrixV().transpose();

    // Last column of U is undetermined since d = (a a 0).
    if (U.determinant() < 0) {
        U.col(2) *= -1;
    }
    // Last row of Vt is undetermined since d = (a a 0).
    if (Vt.determinant() < 0) {
        Vt.row(2) *= -1;
    }

    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    const Eigen::Matrix3d U_W_Vt = U * W * Vt;
    const Eigen::Matrix3d U_Wt_Vt = U * W.transpose() * Vt;

    const std::array<Eigen::Matrix3d, 2> R{{U_W_Vt, U_Wt_Vt}};
    const std::array<Eigen::Vector3d, 2> t{{U.col(2), -U.col(2)}};
    if (relative_poses) {
        poselib::CameraPose pose;
        pose.q = rotmat_to_quat(R[0]);
        pose.t = t[0];
        if (check_cheirality(pose, x1, x2)) {
            relative_poses->emplace_back(pose);
        }

        pose.t = t[1];
        if (check_cheirality(pose, x1, x2)) {
            relative_poses->emplace_back(pose);
        }

        pose.q = rotmat_to_quat(R[1]);
        pose.t = t[0];
        if (check_cheirality(pose, x1, x2)) {
            relative_poses->emplace_back(pose);
        }

        pose.t = t[1];
        if (check_cheirality(pose, x1, x2)) {
            relative_poses->emplace_back(pose);
        }
    }
}

} // namespace poselib