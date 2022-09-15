#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "manager_handler.h"
#include "cuda_function.h"

#define MASTER 0
#define FILE_NAME_INPUT "input.txt"
#define FILE_NAME_OUTPUT "output.txt"
#define READY 0
#define GET_ID 1
#define GET_DIM 2
#define STOP -1
#define GET_MAT 4
#define GET_STR_SIZE 5
#define GET_STR 6

///CPU -openmp
int threads_calculation(int **pic, int **obj,int obj_dim, int pic_start_row,int pic_start_col,double matching_value){
    double sum = (-1)*matching_value;
    // Calculate difference between picture and object and check if it is less than matching value
    #pragma omp parallel for
    for (int i = 0; i < obj_dim; i++) {
        for (int j = 0; j < obj_dim ; j++) {
            double pic_val = pic[pic_start_row + i][pic_start_col + j];
            double obj_val = obj[i][j];
            sum += fabs(((pic_val - obj_val) / (pic_val)));
            if(sum>0) 
                break;
        }
    }
    return sum<0;
}

int computeOnCPU(manager *my_data, int object_index){
    int found=0;
    #pragma omp parallel for
    for (int row = 0; row < my_data->pictures[0].dim - my_data->objects[object_index].dim + 1; row++) {
        for (int col = 0; col < my_data->pictures[0].dim - my_data->objects[object_index].dim + 1 && found==0; col++) {
            // Check if matching is found
            if (threads_calculation(my_data->pictures[0].mat, my_data->objects[object_index].mat, my_data->objects[object_index].dim, row, col, my_data->matching_value)) {
                if(found==1)
                    break;
                found= 1;
                add_matching(my_data,my_data->pictures[0].ID,my_data->objects[object_index].ID,row,col);
                break;
            }
        }
    }
    return found;
}

void master_function(manager *my_data,int processes_count,MPI_Status status){
    int pictures_left=my_data->pictures_count-1;
    int next_ready_slave;
    int i=0;
    while (i<=pictures_left)
    {
        MPI_Recv(&next_ready_slave,1,MPI_INT,MPI_ANY_SOURCE,READY,MPI_COMM_WORLD,&status);//waiting for the next available slave
        //send pic to next_ready_slave
        MPI_Send(&my_data->pictures[i].ID,1,MPI_INT,next_ready_slave,GET_ID,MPI_COMM_WORLD);//sending picture i ID to process next_ready_slave
        MPI_Send(&my_data->pictures[i].dim,1,MPI_INT,next_ready_slave,GET_DIM,MPI_COMM_WORLD);//sending picture i dim to process next_ready_slave
        for(int j=0;j<my_data->pictures[i].dim;j++)
            for(int k=0;k<my_data->pictures[i].dim;k++)
                MPI_Send(&my_data->pictures[i].mat[j][k], 1, MPI_INT,next_ready_slave, GET_MAT+my_data->pictures[i].ID, MPI_COMM_WORLD);//sending picture i mat[j][k] to process next_ready_slave each value will have diff tag
            
        i++;
    }
    //tell all the slaves to stop
    int stop=STOP;
    for(i=1;i<processes_count;i++){
        MPI_Recv(&next_ready_slave,1,MPI_INT,i,READY,MPI_COMM_WORLD,&status);//clear master recv buffer
        MPI_Send(&stop,1,MPI_INT,i,GET_ID,MPI_COMM_WORLD);//sending process i ID -1 --> the process will stop
        }

    //collect all the matches

}

void slave_function(int my_rank,manager *my_data,MPI_Status status){
    int found;
    my_data->pictures_count=1;
    my_data->pictures=(mat_info*) malloc(sizeof(mat_info));//slave holds only one picture at any given time
    my_data->data_string=(char*) malloc(sizeof(char));
    while (1)
    {
        MPI_Send(&my_rank,1,MPI_INT,MASTER,READY,MPI_COMM_WORLD);//informing master that the process is ready
        MPI_Recv(&my_data->pictures[0].ID,1,MPI_INT,MASTER,GET_ID,MPI_COMM_WORLD,&status);//geting picture ID (if ID=-1 stop)
        if(my_data->pictures[0].ID==STOP)
            break;
        MPI_Recv(&my_data->pictures[0].dim,1,MPI_INT,MASTER,GET_DIM,MPI_COMM_WORLD,&status);//geting picture dim
        ////printf("rank=%d     pic_id=%d   pic_dim=%d\n",my_rank,my_data->pictures[0].ID,my_data->pictures[0].dim);
        my_data->pictures[0].mat=(int**) malloc(my_data->pictures[0].dim*sizeof(int*));
        for(int j=0;j<my_data->pictures[0].dim;j++){
            my_data->pictures[0].mat[j]=(int*) malloc(my_data->pictures[0].dim*sizeof(int));
            for(int k=0;k<my_data->pictures[0].dim;k++)
                MPI_Recv(&my_data->pictures[0].mat[j][k],1,MPI_INT,MASTER,GET_MAT+my_data->pictures[0].ID,MPI_COMM_WORLD,&status);//geting picture ID (if ID=-1 stop)
        }
        found=0;
        if(my_rank%2==0 && my_data->pictures[0].dim<1000){//cuda
            for(int i=0;i<my_data->objects_count;i++){
                if(computeOnGPU(my_data, i)==1){
                    found=1;
                    break;
                }
            }
        }else{//openmp
            for(int i=0;i<my_data->objects_count;i++)
                if(computeOnCPU(my_data, i)==1){
                    found=1;
                    break;
                }
        }

        if(found==0){
            add_no_matching_found(my_data,my_data->pictures[0].ID);
        }
    }
}

void master_get_strings(int processes_count,const char *filename,MPI_Status status){
    char** matching;
    int string_size;
    matching=(char**) calloc(processes_count-1,sizeof(char*));
    for(int i=1;i<processes_count;i++){
        MPI_Recv(&string_size,1,MPI_INT,i,GET_STR_SIZE,MPI_COMM_WORLD,&status);//get string size
        matching[i-1]=(char*) calloc(string_size,string_size);
        MPI_Recv(matching[i-1],string_size,MPI_CHAR,i,GET_STR,MPI_COMM_WORLD,&status);//get string
    }
    write_log(matching,processes_count,filename);
    for(int i=0;i<processes_count-1;i++)
        if(matching[i])
        free(matching[i]);
    free(matching);
}

void slave_send_strings(char* matchings){
    int size=strlen(matchings)+1; 
    MPI_Send(&size,1,MPI_INT,MASTER,GET_STR_SIZE,MPI_COMM_WORLD);
    MPI_Send(matchings,size,MPI_CHAR,MASTER,GET_STR,MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    //init mpi//
    int my_rank;
    int processes_count;
    MPI_Status status;
    MPI_Init(&argc,&argv); //start mpi
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank); //find process rank
    MPI_Comm_size(MPI_COMM_WORLD,&processes_count); //find number of processes
    if(processes_count==1){
        printf("Cant run application with 1 process");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //init params
    manager my_data;
    if(my_rank==MASTER)
        my_data=set_manager_from_file(FILE_NAME_INPUT);
    
    
    set_slaves_manager(my_rank,&my_data);

    //split images
    if(my_rank==MASTER)
        master_function(&my_data,processes_count, status);
    else
        slave_function(my_rank,&my_data, status);
    
    MPI_Barrier(MPI_COMM_WORLD);
    //collect matches
    if(my_rank==MASTER)
        master_get_strings(processes_count,FILE_NAME_OUTPUT,status);
    else{
        sum_data_to_string(&my_data);
        slave_send_strings(my_data.data_string);
    }
    if(my_rank==MASTER)
        free_data(&my_data);//free my_data
    else{
        if(my_data.data_string)
            free(my_data.data_string);
        if(my_data.matches_count>0)
            free(my_data.matchings);
    }
    

    MPI_Finalize();
    return 0;
}