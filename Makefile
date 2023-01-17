# set compiler
CC		= gcc
CPPC	= g++
NVCC	= nvcc	

# set dependent libraries' root path
EIGEN_ROOT_DIR	= /public/home/LeiChao/src/BLAS/eigen-3.4.0
CUDA_ROOT_DIR	= /usr/local/cuda-11.1

CUDA_LIB_DIR	= $(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR	= $(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS	= -lcudadevrt -lcudart 

# cuda copmute capability https://stackoverflow.com/questions/28451859/cuda-invalid-device-function-how-to-know-architecture-code
ARCH	= -arch=sm_80
CUDAFLAG= -expt-relaxed-constexpr -rdc=true

# Project File Structure
SRC_DIR = ./src
OBJ_DIR = ./bin
INC_DIR = ./include

INCPATH	=	-I ${CUDA_INC_DIR}	\
			-I ${EIGEN_ROOT_DIR}	\
			-I ${INC_DIR}

LIBPATH	=	-L ${CUDA_LIB_DIR}

CPPFLAG	= -std=c++17 -g
OPT		= -O2 -march=native

# final executabl file
EXE		= main
# Object files
OBJS	= ${OBJ_DIR}/main.o	${OBJ_DIR}/Solver.o 	${OBJ_DIR}/Helper.o	${OBJ_DIR}/mmio.o	${OBJ_DIR}/CUDAKernel.o

# LINK STAGE
${EXE}:	${OBJS}
	${NVCC}	${OBJS}	-o	${EXE}	${LIBPATH}	${CUDA_LINK_LIBS}	${ARCH}

# COMPILE STAGE
${OBJ_DIR}/main.o: 	main.cpp
	${CPPC}	${INCPATH}	-c	main.cpp	-o	${OBJ_DIR}/main.o	${CPPFLAG}

${OBJ_DIR}/Solver.o:	${SRC_DIR}/Solver.cpp
	${CPPC}	${INCPATH}	-c	${SRC_DIR}/Solver.cpp	-o	${OBJ_DIR}/Solver.o	${CPPFLAG}

${OBJ_DIR}/Helper.o:	${SRC_DIR}/Helper.cpp
	${CPPC}	${INCPATH}	-c	${SRC_DIR}/Helper.cpp	-o	${OBJ_DIR}/Helper.o	${CPPFLAG}

${OBJ_DIR}/mmio.o:		${SRC_DIR}/mmio.c
	${CC}	${INCPATH}	-c	${SRC_DIR}/mmio.c		-o	${OBJ_DIR}/mmio.o

${OBJ_DIR}/CUDAKernel.o:	${SRC_DIR}/CUDAKernel.cu
	${NVCC}	${INCPATH}	-c	${SRC_DIR}/CUDAKernel.cu	-o	${OBJ_DIR}/CUDAKernel.o	${ARCH}	${CPPFLAG} ${CUDAFLAG}

clean:
	rm main ./bin/*

# 关于 链接 顺序杂谈
# 每次链接时，只会链接需要的部分？ 这个可以周末做做实验
# 所以在 生成 main.cpp 的时候，要将 这三个都给引入进来
# https://zhuanlan.zhihu.com/p/81681440

# Makefile 中的符号，过一遍 tutorial
# https://makefiletutorial.com/

# dot product 原型
# https://bitbucket.org/jsandham/algorithms_in_cuda/src/master/dot_product/

# CUDA 结合 EIGEN, 这个其实用处不大，我不需要在 CUDA 中使用 EIGEN 提供的函数
# https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable/blob/master/Makefile