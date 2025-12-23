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
#include <cmath>

namespace franka_example_controllers {

bool CartesianPoseExampleController_Mod::init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) {

  //pose_sub_ = node_handle.subscribe("desired_poses", 1, &CartesianPoseExampleController_Mod::topicCallback, this);
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

    std::array<double, 7> q_start = state_handle.getRobotState().q_d;
    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle.getRobotState().q_d[i] - q_start[i]) > 0.1) {
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
  target_position_[0] = initial_pose_[12] + 0.1;
  target_position_[1] = initial_pose_[13];
  target_position_[2] = initial_pose_[14] + 0.1;
  ROS_INFO_STREAM("target position set to x: " << target_position_[0] << " y: " << target_position_[1] << " z: " << target_position_[2]);
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

  double radius = 0.15;
  double angle = M_PI / 4 * (1 - std::cos(M_PI / 5.0 * elapsed_time_.toSec()));
  double delta_x = radius * std::sin(angle);
  double delta_z = radius * (std::cos(angle) - 1);
  ROS_INFO_STREAM("Angle: " << angle);
  //ROS_INFO_STREAM("dx: " << delta_x << " dz: " << delta_z);
  std::array<double, 16> new_pose = initial_pose_;
  new_pose[12] -= delta_x;
  new_pose[14] -= delta_z;
  cartesian_pose_handle_->setCommand(new_pose);
}

}  // namespace franka_controllers
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianPoseExampleController_Mod,
controller_interface::ControllerBase)
