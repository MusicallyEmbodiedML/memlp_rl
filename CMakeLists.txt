cmake_minimum_required(VERSION 3.29)
project(memlp_rl)

set(CMAKE_CXX_STANDARD 14)


file(GLOB MODULE_SOURCES "memlp/*.cpp")

add_executable(memlp_rl main.cpp ${MODULE_SOURCES}
        RLEnvironment.h
        TestEnv.h
        OrnsteinUhleneckNoise.h)
