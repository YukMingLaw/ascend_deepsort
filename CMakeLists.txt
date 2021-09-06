cmake_minimum_required(VERSION 3.5)

project(ACL_DEEPSORT)

add_definitions(-std=c++11)
add_compile_options(-fPIC)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set(OpenCV_DIR /root/opencv4.4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

find_package(OpenMP)

set(ACL_PATH /usr/local/Ascend/ascend-toolkit/latest)
INCLUDE_DIRECTORIES(
    ${ACL_PATH}/acllib/include
    ${OpenCV_INCLUDE_DIRS}
    AclProcess
    KalmanFilter
    MunkresAssignment
    )

set(ACL_LIB_PATH ${ACL_PATH}/acllib/lib64/stub)
link_directories(${ACL_LIB_PATH})

file(GLOB_RECURSE native_srcs 
    main.cpp
    AclProcess/*.cpp
    KalmanFilter/*.cpp
    MunkresAssignment/*.cpp
    MunkresAssignment/munkres/*.cpp
)

add_executable(${PROJECT_NAME} ${native_srcs})

target_link_libraries(${PROJECT_NAME} ascendcl OpenMP::OpenMP_CXX OpenMP::OpenMP_C ${OpenCV_LIBS} pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -s stdc++)
