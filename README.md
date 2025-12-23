![Tumbnail](resources/copertina.png)
<img src="resources/copertina.png" alt="Tumbnail" width="300" heigth="150/>
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
3. Installare ros controllers:
```bash
sudo apt-get install ros-noetic-ros-control ros-noetic-ros-controllers
```
5. Installare boost-sml:
```bash
sudo apt-get install ros-noetic-boost-sml
```
7. Clone the repository in the home direcotory and open terminal 

8. Build workspace:
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

For details on any item above, open the referenced file links.
