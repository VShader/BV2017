cmake_minimum_required(VERSION 3.7)
project(BV_Tutorial)


# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)


find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS widgets quickcontrols2 charts REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(sources
    Aufgabe_3.cpp
    )

set(headers
    )

set(qtFiles
    )

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DQT_QML_DEBUG ")

add_executable(${PROJECT_NAME} ${sources} ${headers} ${qtFiles})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Qt5::QuickControls2 Qt5::Widgets Qt5::Charts)
