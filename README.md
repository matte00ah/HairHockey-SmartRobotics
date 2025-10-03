# HairHockey-SmartRobotics

# Terminale 1: Gazebo con Panda
roslaunch scene all.launch

# Terminale 2: MoveIt + RViz
roslaunch scene my.launch

# Terminale 3: SceneManager.py
rosrun scene scene_manager.py


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