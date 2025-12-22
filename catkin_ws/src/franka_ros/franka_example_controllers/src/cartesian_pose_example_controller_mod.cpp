// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_pose_example_controller_mod.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <controller_interface/controller_base.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Dense>

namespace franka_example_controllers {

bool CartesianPoseExampleController_Mod::init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) {

  pose_sub_ = node_handle.subscribe("desired_poses", 1, &CartesianPoseExampleController_Mod::topicCallback, this);
  cartesian_pose_interface_ = robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianPoseExampleController_Mod: Could not get Cartesian Pose "
        "interface from hardware");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianPoseExampleController_Mod: Could not get parameter arm_id");
    return false;
  }

  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseExampleController_Mod: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianPoseExampleController_Mod: Could not get state interface from hardware");
    return false;
  }

  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");

    std::array<double, 7> q_start = state_handle.getRobotState().q;
    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle.getRobotState().q[i] - q_start[i]) > 0.1) {
        ROS_ERROR_STREAM(
            "CartesianPoseExampleController_Mod: Robot is not in the expected starting position for "
            "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
            "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
        return false;
      }
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseExampleController_Mod: Exception getting state handle: " << e.what());
    return false;
  }

  return true;
}

void CartesianPoseExampleController_Mod::starting(const ros::Time& /* time */) {
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  target_position_[0] = initial_pose_[12]+0.2;
  target_position_[1] = initial_pose_[13];
  target_position_[2] = initial_pose_[14];
  ROS_INFO_STREAM("target position set to x: " << target_position_[0] << " y: " << target_position_[1] << " z: " << target_position_[2]);
  motion_duration_= 5.0;
  elapsed_time_ = ros::Duration(0.0);
}

std::array<double, 16> CartesianPoseExampleController_Mod::poseToArray(const geometry_msgs::Pose& pose) {
 
  Eigen::Quaterniond q(
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z);
 
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = q.toRotationMatrix();
  T(0,3) = pose.position.x;
  T(1,3) = pose.position.y;
  T(2,3) = pose.position.z;
 
  std::array<double,16> out;
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::ColMajor>>(out.data()) = T;
  return out;
}


void CartesianPoseExampleController_Mod::update(const ros::Time& /* time */,
                                     const ros::Duration& period) {
  elapsed_time_ += period;
  std::array<double, 16> new_pose = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  double t;
  std::array<double, 3> distance = {
      std::abs(target_position_[0] - new_pose[12]),
      std::abs(target_position_[1] - new_pose[13]),
      std::abs(target_position_[2] - new_pose[14])
  };

  //ROS_INFO_STREAM("Distance to target x: " << distance[0] << " y: " << distance[1] << " z: " << distance[2]);
  
  if (distance[0] <= 0.002) {
    //pose_received_ = false; // target raggiunto
    ROS_INFO_STREAM("Target reached");
    ROS_INFO_STREAM("Final position x: " << new_pose[12] << " y: " << new_pose[13] << " z: " << new_pose[14]);
    initial_pose_ = new_pose;
    t = 0.0;
    return;
  }
  
  if (pose_received_) {
    ROS_INFO_STREAM("pose received: " << pose_received_);
    double norm_distance = std::sqrt(distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2]);
    double v_max = 0.55;         // m/s
    sigma = norm_distance / (v_max * std::sqrt(M_PI));
    T = 6.0 * sigma;
    ROS_INFO_STREAM(" T: " << T);
    //ROS_INFO_STREAM("s: " << s);
    //s = std::min(s, 1.0);

    //double ds = s - s_prev_;
    //s_prev_ = s;
    pose_received_ = false;

  }
  t = std::min(elapsed_time_.toSec(), T);
  double tau = (t - T / 2.0) / sigma;
  double s = 0.5 * (1.0 + std::erf(tau));

  double target_delta_x = 0.001;
  double target_delta_z = 0.10;

    for (int i = 0; i < 1; i++) {
      //double delta = ds * (target_position_[i] - initial_pose_[12 + i]);
      double delta = s * target_delta_x;
      new_pose[12 + i] += delta;
    } 
  cartesian_pose_handle_->setCommand(new_pose);
}

//void CartesianPoseExampleController_Mod::update(const ros::Time& /* time */,
//                                     const ros::Duration& period) {
//  elapsed_time_ += period;
//  std::array<double, 16> new_pose = cartesian_pose_handle_->getRobotState().O_T_EE_d;
//  double t = elapsed_time_.toSec();
//  double T = motion_duration_;
//  std::array<double, 3> distance = {
//      std::abs(target_position_[0] - new_pose[12]),
//      std::abs(target_position_[1] - new_pose[13]),
//      std::abs(target_position_[2] - new_pose[14])
//  };
//
//  ROS_INFO_STREAM("Distance to target x: " << distance[0] << " y: " << distance[1] << " z: " << distance[2]);
//  
//  if (distance[0] <= 0.002 && distance[2] <= 0.002) {
//    pose_received_ = false; // target raggiunto
//    ROS_INFO_STREAM("Target reached");
//    ROS_INFO_STREAM("Final position x: " << new_pose[12] << " y: " << new_pose[13] << " z: " << new_pose[14]);
//    initial_pose_ = new_pose;
//    t = 0.0;
//    return;
//  }
//  
//  //if (pose_received_) {
//
//    double s = 0.5 * (1.0 - std::cos(M_PI * t / T));
//    //ROS_INFO_STREAM("s: " << s);
//    //s = std::min(s, 1.0);
//
//    //double ds = s - s_prev_;
//    //s_prev_ = s;
//
//    double target_delta_x = 0.001;
//    double target_delta_z = 0.10;
//
//    for (int i = 0; i < 3; i++) {
//      //double delta = ds * (target_position_[i] - initial_pose_[12 + i]);
//      double delta = s * target_delta_x;
//      new_pose[12 + i] += delta;
//    }
//
//  //}
//  cartesian_pose_handle_->setCommand(new_pose);
//}

// Callback per ricevere i comandi da ROS
void CartesianPoseExampleController_Mod::topicCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  // Aggiorniamo il target. La logica di "update" si occuperà di raggiungerlo dolcemente.
  target_position_[0] = msg->pose.position.x;
  target_position_[1] = msg->pose.position.y;
  target_position_[2] = msg->pose.position.z;
  s_prev_ = 0.0;
  elapsed_time_ = ros::Duration(0.0);
  pose_received_ = true;
 
  // Nota: Questo codice mantiene l'orientamento fisso a quello iniziale.
  // Se vuoi cambiare orientamento, devi passare un geometry_msgs::Pose.
}

}  // namespace franka_controllers
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianPoseExampleController_Mod,
controller_interface::ControllerBase)
//void CartesianPoseExampleController_Mod::update(const ros::Time& /* time */,
//                                            const ros::Duration& period) {
//  
//  elapsed_time_ += period;
//
//  std::array<double, 16> command_pose;
//  command_pose = initial_pose_;
//
//  {
//    std::lock_guard<std::mutex> lock(pose_mutex_);
//    if (pose_received_) {
//      command_pose = poseToArray(target_pose_.pose);
//      pose_received_ = false;
//      interpolation_start_time_ = elapsed_time_;
//      interpolating_ = true;
//    }
//  }
//  if (interpolating_) {
//    std::array<double, 16> current_pose = cartesian_pose_handle_->getRobotState().O_T_EE_d;
//    double step_size = 0.00001;
//    double dx = (command_pose[12] - current_pose[12])*step_size;
//    double dy = (command_pose[13] - current_pose[13])*step_size;
//    double dz = (command_pose[14] - current_pose[14])*step_size;
//
//    command_pose = current_pose;
//    command_pose[12] += dx;
//    command_pose[13] += dy;
//    command_pose[14] += dz;
//  }
//  cartesian_pose_handle_->setCommand(command_pose);
//}
//
//
//void CartesianPoseExampleController_Mod::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
//  std::lock_guard<std::mutex> lock(pose_mutex_);
//  target_pose_ = *msg;
//  pose_received_ = true;
//}
//
//}  // namespace franka_example_controllers
//

