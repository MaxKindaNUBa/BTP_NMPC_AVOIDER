cmake_minimum_required(VERSION 3.8)
project(nmpc_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VesselState.msg"
  "msg/ControlCommand.msg"
  "msg/PredictionHorizon.msg"
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  "msg/ActiveReference.msg"
  "msg/SimStatus.msg"
  "msg/SolverStatus.msg"
  "srv/SolveNMPC.srv"
  "srv/GetScenario.srv"
  "srv/ResetSim.srv"
  DEPENDENCIES std_msgs
)

ament_package()
