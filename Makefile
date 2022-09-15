build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c manager_handler.c -o manager_handler.o
	nvcc -I./inc -c cuda_function.cu -o cuda_function.o
	mpicxx -fopenmp -o mpiCudaOpenMP  main.o manager_handler.o cuda_function.o /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpenMP

run:
	mpiexec -np 4 ./mpiCudaOpenMP

runOn2:
	mpiexec -np 4 -machinefile  mf  -map-by  node  ./mpiCudaOpenMP