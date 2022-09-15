#include "cuda_function.h"

__device__  __host__ float f(int *pic, int *obj,int pic_dim,int obj_dim,int pic_start_row,int pic_start_col){
    float sum=0;
    for(int i=0;i<obj_dim;i++){
        for(int j=0;j<obj_dim;j++){
            float pic_val=pic[(pic_start_row+i)*pic_dim+pic_start_col+j];
            float obj_val=obj[i*obj_dim+j];
            sum+=fabs(pic_val-obj_val)/pic_val;
        }
    }
    return sum;
}

__global__ void kernel(int *pic, int *obj, float* result,int pic_dim,int obj_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pic_row=i/pic_dim;
    int pic_col=i%pic_dim;
    result[i]=f(pic,obj,pic_dim,obj_dim,pic_row,pic_col);
}

void transfer_2D_mat_to_1D(int *one_dim_mat,int **mat_h,int N){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            one_dim_mat[i*N+j]=mat_h[i][j];
}

void check_cuda_allocation(cudaError_t err,int type){
    if (err != cudaSuccess) {
        if(type==1)
            fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        if(type==2)
            fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        if(type==3)
            fprintf(stderr, "Failed to copy data from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
int computeOnGPU(manager *my_data, int object_index){
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int pic_dim=my_data->pictures[0].dim;
    int *pic;
    int temp_1D_pic[pic_dim*pic_dim];
    const size_t pic_size = sizeof(int) * size_t(pic_dim*pic_dim);
    err =cudaMalloc((void **)&pic,  pic_size);
    check_cuda_allocation(err,1);

    transfer_2D_mat_to_1D(temp_1D_pic,my_data->pictures[0].mat,pic_dim);
    err = cudaMemcpy(pic, temp_1D_pic,  pic_size, cudaMemcpyHostToDevice);
    check_cuda_allocation(err,2);
    

    int obj_dim=my_data->objects[object_index].dim;
    int *obj;
    int temp_1D_obj[obj_dim*obj_dim];
    const size_t obj_size = sizeof(int) * size_t(obj_dim*obj_dim);
    err =cudaMalloc((void **)&obj,  obj_size);
    check_cuda_allocation(err,1);

    transfer_2D_mat_to_1D(temp_1D_obj,my_data->objects[object_index].mat,obj_dim);
    err = cudaMemcpy(obj, temp_1D_obj,  obj_size, cudaMemcpyHostToDevice);
    check_cuda_allocation(err,1);


    float *result;
    const size_t result_size = sizeof(float) * size_t(pic_dim*pic_dim);
    err =cudaMalloc((void **)&result,  result_size);
    check_cuda_allocation(err,1);

    float *host_result;
    host_result=(float*) malloc(result_size);
    if (host_result == NULL){
        printf("Error allocating memory");
        exit(EXIT_FAILURE);
    }

    int bool_found_match=0;
    kernel<<<pic_dim-obj_dim+1, pic_dim-obj_dim+1>>>(pic, obj, result, pic_dim, obj_dim);
    
    err = cudaMemcpy(host_result, result,  result_size, cudaMemcpyDeviceToHost);//copy result arry to device
    check_cuda_allocation(err,3);
    
    for(int i=0;i<pic_dim*pic_dim;i++){
        if(host_result[i]<my_data->matching_value){
            bool_found_match=1;
            add_matching(my_data,my_data->pictures[0].ID,my_data->objects[object_index].ID,i/pic_dim,i%pic_dim);
            break;
        }
    }

    if (cudaFree(obj) != cudaSuccess || cudaFree(pic) != cudaSuccess|| cudaFree(result) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    free(host_result);

    return bool_found_match;
}


