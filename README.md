<div align="center">
  <img src="resources/copertina.png" alt="Copertina" width="300" height="300"/>
</div>

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

## Environment Setup
1. Build and install libfranka in the home directory:
```bash
git clone --recursive https://github.com/frankaemika/libfranka --branch 0.8.0
cd libfranka
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
cpack -G DEB
sudo dpkg -i libfranka-0.8.0-amd64.deb
```
2. Install ros controllers:
```bash
sudo apt-get install ros-noetic-ros-control ros-noetic-ros-controllers
```
3. Install boost-sml:
```bash
sudo apt-get install ros-noetic-boost-sml
```
4. Install Moveit:
```bash
sudo apt install ros-noetic-moveit
```
5. Clone the repository in the home direcotory and open terminal 

6. Build workspace:
```bash
cd ./HairHockey-SmartRobotics/catkin_ws
catkin_make
```

## Start execution
1. Move inside repo directory:
```bash
cd ~/HairHockey-SmartRobotics/catkin_ws
source devel/setup.bash
```
2. Start control:
  - For just virtual simulation run
    ```bash
    roslaunch scene all.launch gazebo:=true
    ```
  - For physical robot run
    ``` bash
    roslaunch franka_control franka_control.launch robot_ip:=172.16.0.2 load_gripper:=false
    ```

    then in a new terminal run
    ```bash
    roslaunch scene all.launch
    ```
3. Start Rviz:
```bash
roslaunch scene my.launch
```
4. Run Disk Tracker with Montecarlo
```bash
rosrun scene disk_tracker.py
```

## Further project specifications

The movement of the robot is managed and controlled via MoveIt, which is not ideal for this task since it's not real-time compatible, and more suited for task where high complexity trajectory planing is required, such as those where movement in all 3 dimensions is needed, and where there are a lot of environment obstacles that the robot must avoid. Since here the only obstacle to avoid is the border of the table, and the movement can be semplified to planar, the use of the Cartesia Pose Controller would be better: the low level of the controller coupled with no planning and continuous stream reading, makes it real-time compliant. Unfortunately, due to time restrictions, and possible defects in the robot and its joints velocities initialization, it was impossible for us to use it effectively. The code implementation we made for the CartesianPoseExampleController can be seen however in the develop branch, with some specifications on how to run that and a possible solution to make it work.