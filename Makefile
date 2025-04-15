# all:
# 	g++ Principal.cpp -I/usr/local/include/opencv4 -L/usr/local/lib/ -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_videoio -o Principal
# run:
# 	./Principal


# Nombre del ejecutable
all:
	nvcc main.cu -O3 -I/usr/local/include/opencv4 -L/usr/local/lib/ -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_videoio -o Principal
run:
	./Principal