# AirHockey using Cartesian Pose Controller

This is a small README file that explains how to control the robot using CartesianPoseExampleController_Mod instead of Moveit. For the suboptimal but working project using MoveIt, move to the new_develop branch. 

Though this approach with a low level position controller should be theoretically better to make the robot move faster and have a real-time control, we didn't manage to make it work due to time restrictions and some complications. 

The main problem we had is that the robot enters in Reflex mode as soon as the controller starts moving towards the target position, so then we would press the E-Stop button to stop and reset the robot out of this mode. We noticed however that, if the controller is not stopped when the E-stop button is pressed and resetted, just as the button resets, the robot would perform the desired motion as expected. 

The error we get from the robot, when entering in Reflex mode, is sometimes a velocity limits or discontinuity violation, or an acceleration discontinuity violation, but most of the times it's an error saying that the motion command was rejected because the robot was still moving when it received it.

Given these observations what we supposed is that the matrices exploiting the velocities of the joints of the robot, are not set to a 0-vector at initialization of the controller: when the controller then receives the target position to reach, it's asked to move at a way slower speed than it thinks it's moving, and so producing the said errors that make it enter Reflex mode. 

What we think may solve the problem, even if we were not able to test it, is to modify, at startup and everytime a new position is sent to the robot, the joints velocities vectors, which should be **dq** and/or **dq_d** (the documentation is available at https://docs.ros.org/en/kinetic/api/libfranka/html/structfranka_1_1RobotState.html). They should be retrievable in cpp with the command
```bash
cartesian_pose_handle_->getRobotState()
```
and then modified by topic subscription and publication. They should also be visible by running on a first terminal
```bash
roslaunch franka_ros franka_control.launch robot_ip:=\<your_robot_ip\>
```
and then on a second terminal with the command 
```bash
rostopic echo /franka_state_controller/franka_states
```

## Environment Setup (same as in new_develop branch)
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
1. Start the controller
```bash
roslaunc franka_example_controllers franka_pose_example_controller.launch robot_ip:=\<your_robot_ip\> load_gripper:=false
```
2. From this point forward we could not solve the problems we had and so here is where to start trying solving the problem with the initial speed of the robot, making it not go in Reflex mode because it thinks it's already moving 

As the code is now it simply imposes as target pose one which is 10cm forward on the x and z axis. 

The next step after the Reflex problem is solved should be to uncomment pose_sub_ in order to create a topic where to publish the positions, uncomment also the topicCallback method to accept and read positionf from that topic, and modify the target position update logic consequently. 