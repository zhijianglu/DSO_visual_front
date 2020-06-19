
#TODO####################################### 去光度畸变矫正
add_executable(Undistorter
        demo_Undistorter.cpp
        )
target_link_libraries( Undistorter ${OpenCV_LIBS} ${PROJECT_NAME}_Undistorter
        )


#TODO####################################### 图像像素处理：计算图像梯度  选择像素点
add_executable(PixelProcess
        demo_PixelPixelProcess.cpp
        )
target_link_libraries( PixelProcess
        ${OpenCV_LIBS}
        ${PROJECT_NAME}_PixelGradient
        ${PROJECT_NAME}_PixelSelector
        ${PROJECT_NAME}_Utils
        )

add_executable(ComputeGradient
        demo_ComputeGrident.cpp
        )
target_link_libraries( ComputeGradient
        ${OpenCV_LIBS}
        ${PROJECT_NAME}_PixelGradient
        ${PROJECT_NAME}_PixelSelector
        ${PROJECT_NAME}_Utils
        )


