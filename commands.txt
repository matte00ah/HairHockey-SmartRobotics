# HairHockey-SmartRobotics

# Terminale 0: franka control
roslaunch franka_control franka_control.launch robot_ip:=172.16.0.2 load_gripper:=false

# Terminale 1: Gazebo con Panda
roslaunch scene all.launch

# Terminale 2: MoveIt + RViz
roslaunch scene my.launch

# Terminale 3: SceneManager.py
rosrun scene scene_manager.py

#Prima di lanciare disk_tracker
jobs -l
kill -9 {pid}

# Framesc
disk_tracker: converte da pixel a metri
move_franka: sposta dall'angolo a sinistra del tavolo al WORLD frame


# In caso di errore su catkin_make fatto la prima volta che si clona il git: 

# Could not find a package configuration file provided by "Franka" (requested
# version 0.8.0) with any of the following names:
#
#   FrankaConfig.cmake
#   franka-config.cmake

# si deve fare la build di libfrakna come segue
git clone --recursive https://github.com/frankaemika/libfranka --branch 0.8.0
cd libfranka
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
cpack -G DEB
sudo dpkg -i libfranka-0.8.0-amd64.deb

# Per installare ros-controllers
sudo apt-get install ros-noetic-ros-control ros-noetic-ros-controllers
# Per installare boost-sml
sudo apt-get install ros-noetic-boost-sml


# Modificato joint_limit.yaml messo in joint_1 la velocity a 0

# Comando per leggere lo stato attuale del robot
rostopic echo /franka_state_controller/franka_states | grep robot_mode


# Prima del catkin_make
cancella le cartelle build e devel
source /opt/ros/noetic/setup.bash
catkin_make


# modifica fatta in franka_arm.xacro
<joint name="${arm_id}_joint6" type="revolute">
      <origin rpy="${pi/2} 0 0" xyz="0 0 0" />  # rpy="${pi/2} ${-pi/2} 0"
