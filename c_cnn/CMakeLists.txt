cmake_minimum_required(VERSION 4.0)
project(c_cnn C)

set(CMAKE_C_STANDARD 11)

include_directories(rawnetwork/include)

add_executable(c_cnn
        rawnetwork/include/activators.h
        rawnetwork/include/components.h
        rawnetwork/include/computational.h
        rawnetwork/include/functional.h
        rawnetwork/include/helpers.h
        rawnetwork/include/types.h
        rawnetwork/src/activators.c
        rawnetwork/src/components.c
        rawnetwork/src/computational.c
        rawnetwork/src/functional.c
        rawnetwork/src/helpers.c
        rawnetwork/src/main.c)
