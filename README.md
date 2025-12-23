# HairHockey‑SmartRobotics

Brief project for autonomous puck tracking and Franka Panda control in simulation (Gazebo) + MoveIt. The repository integrates scene perception, particle‑filter based tracking and attack logic with Franka control stacks and example controllers.

## Main functionalities
- Real‑time puck detection & table homography
  - Tracker and homography: [`process_frame`](catkin_ws/src/scene/scripts/origin_detector.py) / [`compute_homography`](catkin_ws/src/scene/scripts/disk_tracker.py) — see [catkin_ws/src/scene/scripts/origin_detector.py](catkin_ws/src/scene/scripts/origin_detector.py) and [catkin_ws/src/scene/scripts/disk_tracker.py](catkin_ws/src/scene/scripts/disk_tracker.py).
  - Pixel→meter conversion: [`pixel_to_meter_fast`](catkin_ws/src/scene/scripts/disk_tracker.py).
- Monte‑Carlo filtering and decision making for puck prediction / robot attack
  - Filter class: [`MontecarloFilter`](catkin_ws/src/scene/scripts/montecarlo_filter.py) (and modified variant: [catkin_ws/src/scene/scripts/montecarlo_filter_modified.py](catkin_ws/src/scene/scripts/montecarlo_filter_modified.py)).
  - Attack strategies and robot reach checks implemented in the filter's `run` / `update` methods — see [catkin_ws/src/scene/scripts/montecarlo_filter.py](catkin_ws/src/scene/scripts/montecarlo_filter.py).
- Robot motion and visualization
  - Panda arm wrapper: [`PandaArm`](catkin_ws/src/scene/scripts/move_franka.py) — [catkin_ws/src/scene/scripts/move_franka.py](catkin_ws/src/scene/scripts/move_franka.py).
  - Example scripts to move to start / visualize sphere in RViz: [catkin_ws/src/franka_ros/franka_example_controllers/scripts/move_to_start.py](catkin_ws/src/franka_ros/franka_example_controllers/scripts/move_to_start.py).
- Simulation utilities
  - Gazebo puck bounce / physics helper: [catkin_ws/src/scene/scripts/bounce.py](catkin_ws/src/scene/scripts/bounce.py).
  - Scene manager to synchronize models and MoveIt: [catkin_ws/src/scene/scripts/scene_manager.py](catkin_ws/src/scene/scripts/scene_manager.py).

## Key services & controllers (Franka)
- Hardware + services container and helpers:
  - Service wrapper / container: [`franka_hw::ServiceContainer`](catkin_ws/src/franka_ros/franka_hw/include/franka_hw/services.h) and service helper [`franka_hw::advertiseService`](catkin_ws/src/franka_ros/franka_hw/include/franka_hw/services.h).
  - Service setup implementations: [`franka_hw::setupServices`](catkin_ws/src/franka_ros/franka_hw/src/services.cpp).
  - Combinable HW service/action setup: [`FrankaCombinableHW::setupServicesAndActionServers`](catkin_ws/src/franka_ros/franka_hw/src/franka_combinable_hw.cpp).
- Control node and lifecycle:
  - Main control node: [catkin_ws/src/franka_ros/franka_control/src/franka_control_node.cpp](catkin_ws/src/franka_ros/franka_control/src/franka_control_node.cpp).
  - Gripper node & action servers: [catkin_ws/src/franka_ros/franka_gripper/src/franka_gripper_node.cpp](catkin_ws/src/franka_ros/franka_gripper/src/franka_gripper_node.cpp).
- Example controllers and teleoperation:
  - Teleop / PD follower example: [`franka_example_controllers::TeleopJointPDExampleController`](catkin_ws/src/franka_ros/franka_example_controllers/include/franka_example_controllers/teleop_joint_pd_example_controller.h) and implementation [catkin_ws/src/franka_ros/franka_example_controllers/src/teleop_joint_pd_example_controller.cpp](catkin_ws/src/franka_ros/franka_example_controllers/src/teleop_joint_pd_example_controller.cpp).
  - Various example controllers and CMake targets: [catkin_ws/src/franka_ros/franka_example_controllers/CMakeLists.txt](catkin_ws/src/franka_ros/franka_example_controllers/CMakeLists.txt).

## Quickstart (simulated workflow)
1. Build and install libfranka (example in repo README).  
2. Build workspace:
   - source ROS (e.g. `source /opt/ros/noetic/setup.bash`) then run `catkin_make` in `catkin_ws`.
3. Launch stack (example terminals):
   - franka control: `roslaunch franka_control franka_control.launch robot_ip:=172.16.0.2 load_gripper:=false` — see [catkin_ws/src/franka_ros/franka_control/src/franka_control_node.cpp](catkin_ws/src/franka_ros/franka_control/src/franka_control_node.cpp).
   - Gazebo with Panda: `roslaunch scene all.launch` — scene launch files in [catkin_ws/src/scene/launch](catkin_ws/src/scene/).
   - MoveIt + RViz: `roslaunch scene my.launch`.
   - Start perception + tracking: run disk tracker script [catkin_ws/src/scene/scripts/disk_tracker.py](catkin_ws/src/scene/scripts/disk_tracker.py).
   - Optional: run `scene_manager.py` ([catkin_ws/src/scene/scripts/scene_manager.py](catkin_ws/src/scene/scripts/scene_manager.py)) to manage models.

## Build / CI / tooling
- CMake + catkin: packages include CMakeLists in respective folders (example: [catkin_ws/src/franka_ros/franka_hw/CMakeLists.txt](catkin_ws/src/franka_ros/franka_hw/CMakeLists.txt), [catkin_ws/src/franka_ros/franka_gazebo/CMakeLists.txt](catkin_ws/src/franka_ros/franka_gazebo/CMakeLists.txt)).
- Formatting and static checks: Clang/pep tooling included via [catkin_ws/src/franka_ros/cmake/ClangTools.cmake](catkin_ws/src/franka_ros/cmake/ClangTools.cmake) and [catkin_ws/src/franka_ros/cmake/PepTools.cmake](catkin_ws/src/franka_ros/cmake/PepTools.cmake).
- CI pipeline example: [catkin_ws/src/franka_ros/Jenkinsfile](catkin_ws/src/franka_ros/Jenkinsfile).

## Useful entry points (files)
- Scene / perception:
  - [catkin_ws/src/scene/scripts/disk_tracker.py](catkin_ws/src/scene/scripts/disk_tracker.py) — main tracker.
  - [catkin_ws/src/scene/scripts/origin_detector.py](catkin_ws/src/scene/scripts/origin_detector.py) — corner/line detection and `process_frame`.
  - [catkin_ws/src/scene/scripts/montecarlo_filter.py](catkin_ws/src/scene/scripts/montecarlo_filter.py) — particle filter (`MontecarloFilter`).
  - [catkin_ws/src/scene/scripts/move_franka.py](catkin_ws/src/scene/scripts/move_franka.py) — `PandaArm` + MoveIt helpers.
  - [catkin_ws/src/scene/scripts/bounce.py](catkin_ws/src/scene/scripts/bounce.py) — Gazebo puck motion helper.
- Franka stack:
  - [catkin_ws/src/franka_ros/franka_hw/include/franka_hw/services.h](catkin_ws/src/franka_ros/franka_hw/include/franka_hw/services.h)
  - [catkin_ws/src/franka_ros/franka_hw/src/services.cpp](catkin_ws/src/franka_ros/franka_hw/src/services.cpp)
  - [catkin_ws/src/franka_ros/franka_control/src/franka_control_node.cpp](catkin_ws/src/franka_ros/franka_control/src/franka_control_node.cpp)
  - [catkin_ws/src/franka_ros/franka_example_controllers/include/franka_example_controllers/teleop_joint_pd_example_controller.h](catkin_ws/src/franka_ros/franka_example_controllers/include/franka_example_controllers/teleop_joint_pd_example_controller.h)

## Notes & tips
- If `FrankaConfig.cmake` not found, build and install `libfranka` (see current README instructions in this repo).
- Before first `catkin_make`, remove `build` and `devel` folders and source ROS setup: `source /opt/ros/noetic/setup.bash`.

For details on any item above, open the referenced file links.