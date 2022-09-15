//project submitted by Gal Karasnty 208934786
this program allocate pictures to processes dynamically when process is free,
than the evan process are calculated with calculated cuda and the odd with openmp (if the picture size is high it will also be calculated with openmp because cuda couldn't allocate enough memory)

Known Issue- 
Because the new input file I had to change the program what cause an allocation error I couldn't fixed ,
this make the program crush *after writing the output file at the MPI_FINALIZE this issue also can cause error for picture  ID 1.