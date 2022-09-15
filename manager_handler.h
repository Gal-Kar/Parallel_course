#pragma once
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#define MASTER 0

struct mat_info {
    int ID;
	int dim;
    int **mat;
};

struct match {
    int p_ID;
    int o_ID;
    int i;
    int j;
};

struct manager {
    double matching_value;
	int pictures_count;
    int objects_count;
    int matches_count;
    int no_matches_count;
    mat_info *pictures;
    mat_info *objects;
    match *matchings;
    int *no_match;
    char *data_string;
};

#define ALLOCATION_ERROR "Error allocating memory"
#define CHECK_POINTER(pointer, err) {\
        if (pointer == NULL) {\
            printf("%s",err);\
            fflush(stdout);\
            MPI_Abort(MPI_COMM_WORLD,1);\
        }\
}

manager set_manager_from_file(const char* filename);
void add_matching(manager *m,int pID,int oID,int x, int y);
void add_no_matching_found(manager *m,int pID);
void write_log(char** matching,int processes_count,const char* filename);
void set_slaves_manager(int my_rank,manager *my_data);
void sum_data_to_string(manager *m);
void free_matrix(int **mat,int size);
void free_data(manager* my_data);