#include "manager_handler.h"
#include <mpi.h>
#include <string.h>


manager set_manager_from_file(const char* file_name){
    //opening the file
    FILE *fp;
    fp = fopen(file_name, "r");
    CHECK_POINTER(fp, "Can't open file for reading \n");

    manager m;
    fscanf(fp, "%lf", &m.matching_value);

    //adding pictures
    fscanf(fp, "%d", &m.pictures_count);
    m.pictures=(mat_info *) malloc(m.pictures_count*sizeof(mat_info));
    CHECK_POINTER(m.pictures,ALLOCATION_ERROR);
    for(int i=0;i<m.pictures_count;i++){
        fscanf(fp, "%d", &m.pictures[i].ID);
        fscanf(fp, "%d", &m.pictures[i].dim);
        m.pictures[i].mat=(int **) malloc(m.pictures[i].dim*sizeof(int*));
        CHECK_POINTER(m.pictures[i].mat,ALLOCATION_ERROR);
        for(int j=0;j<m.pictures[i].dim;j++){
            m.pictures[i].mat[j]=(int *) malloc(m.pictures[i].dim*sizeof(int));
            CHECK_POINTER(m.pictures[i].mat[j],ALLOCATION_ERROR);
            for(int k=0;k<m.pictures[i].dim;k++)
                fscanf(fp, "%d", &m.pictures[i].mat[j][k]);
        }
    }

    //adding objects
    fscanf(fp, "%d", &m.objects_count);
    m.objects=(mat_info *) malloc(m.objects_count*sizeof(mat_info));
    CHECK_POINTER(m.objects,ALLOCATION_ERROR);
    for(int i=0;i<m.objects_count;i++){
        fscanf(fp, "%d", &m.objects[i].ID);
        fscanf(fp, "%d", &m.objects[i].dim);
        m.objects[i].mat=(int **) malloc(m.objects[i].dim*sizeof(int*));
        CHECK_POINTER(m.objects[i].mat,ALLOCATION_ERROR);
        for(int j=0;j<m.objects[i].dim;j++){
            m.objects[i].mat[j]=(int *) malloc(m.objects[i].dim*sizeof(int));
            CHECK_POINTER(m.objects[i].mat[j],ALLOCATION_ERROR);
            for(int k=0;k<m.objects[i].dim;k++)
                fscanf(fp, "%d", &m.objects[i].mat[j][k]);
        }
    }
    return m;
}

void set_slaves_manager(int my_rank,manager *my_data){
    //broadcast matching value and  objects count
    MPI_Bcast(&my_data->matching_value, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&my_data->objects_count, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    my_data->matches_count=0;
    my_data->no_matches_count=0;
    //add objects to slave
    if(my_rank!=MASTER){
        my_data->objects=(mat_info*) malloc(my_data->objects_count*sizeof(mat_info));
        CHECK_POINTER(my_data->objects,ALLOCATION_ERROR);
        }
    
    for(int j=0;j<my_data->objects_count;j++){
        MPI_Bcast(&my_data->objects[j].ID, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        MPI_Bcast(&my_data->objects[j].dim, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        if(my_rank!=MASTER){
            my_data->objects[j].mat=(int**) malloc(my_data->objects[j].dim*sizeof(int*));
            CHECK_POINTER( my_data->objects[j].mat,ALLOCATION_ERROR);
            }
        for(int k=0;k<my_data->objects[j].dim;k++){
            if(my_rank!=MASTER)
                my_data->objects[j].mat[k]=(int*) malloc(my_data->objects[j].dim*sizeof(int));
            for(int z=0;z<my_data->objects[j].dim;z++){
                MPI_Bcast(&my_data->objects[j].mat[k][z], 1, MPI_INT, MASTER, MPI_COMM_WORLD);
            }
        }
    }
}

void free_matrix(int **mat,int size){
    for(int i = 0; i < size; i++)
        free(mat[i]);
    free(mat);
}

void write_log(char** matching,int processes_count,const char* filename){
    FILE *fp = fopen(filename, "w");
    CHECK_POINTER(fp, "Can't open file for writing \n");
    for(int i=0;i<processes_count-1;i++){
        //printf("%s\n",matching[i]);
        fprintf(fp, "%s",matching[i]+1);
        }
    fclose(fp);
}


void add_matching(manager *m,int pID,int oID,int i, int j){
    if(m->matches_count!=0)
        m->matchings=(match*)realloc( m->matchings,(m->matches_count+1)*sizeof(match));
    else
        m->matchings=(match*)malloc(sizeof(match));
    CHECK_POINTER(m->matchings,ALLOCATION_ERROR);
    m->matchings[m->matches_count].p_ID=pID;
    m->matchings[m->matches_count].o_ID=oID;
    m->matchings[m->matches_count].i=i;
    m->matchings[m->matches_count].j=j;
    m->matches_count++;
}

void add_no_matching_found(manager *m,int pID){
    m->no_match=(int*)realloc( m->no_match,(m->no_matches_count+1)*sizeof(int));
    CHECK_POINTER(m->matchings,ALLOCATION_ERROR);
    m->no_match[m->no_matches_count]=pID;
    m->no_matches_count++;
}

void sum_data_to_string(manager *m){
    char tmp[100];
    m->data_string=(char*)malloc(1*sizeof(char));
    m->data_string[1]='\0';
    strcpy(m->data_string," ");
    for(int i=0;i<m->matches_count;i++){
        sprintf(tmp,"Picture - %d Found Object - %d In Position(%d,%d)\n",m->matchings[i].p_ID,m->matchings[i].o_ID,m->matchings[i].i,m->matchings[i].j);
        strcat(m->data_string,tmp);
    }
    for(int i=0;i<m->no_matches_count;i++){
        sprintf(tmp,"Picture - %d No object Were Found \n",m->no_match[i]);
        strcat(m->data_string,tmp);
    }
}


void free_data(manager* my_data){
    for(int i=0;i<my_data->pictures_count;i++)
        free_matrix(my_data->pictures[i].mat,my_data->pictures[i].dim);
    
    for(int i=0;i<my_data->objects_count;i++)
        free_matrix(my_data->objects[i].mat,my_data->objects[i].dim);
}

    
