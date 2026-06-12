cmake_minimum_required(VERSION 3.15)
project(MLCodec_extensions_cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}/pybind11/share/cmake/pybind11")

pybind11_add_module(MLCodec_extensions_cpp
    py_rans/py_rans.cpp
    py_rans/rans.cpp
)

target_include_directories(MLCodec_extensions_cpp PRIVATE py_rans)

set_target_properties(MLCodec_extensions_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
