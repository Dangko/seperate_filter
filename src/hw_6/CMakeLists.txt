cmake_minimum_required(VERSION 3.0.2)
project(hw_6)

#set(OpenCV_DIR /opt/usr/include)
set(OpenCV_DIR /usr/share/OpenCV)

find_package(catkin REQUIRED
        roscpp
        OpenCV
        )

catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES image_pkg
        #  CATKIN_DEPENDS other_catkin_pkg
        #  DEPENDS system_lib
)



include_directories(${catkin_INCLUDE_DIRS}${OpenCV_INCLUDE_DIRS})

add_executable(hw_6 src/hw_6.cpp)
target_link_libraries(hw_6 ${catkin_LIBRARIES} ${OpenCV_LIBS})