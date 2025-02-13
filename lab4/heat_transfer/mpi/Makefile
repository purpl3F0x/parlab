.KEEP_STATE:
MPICC = mpicc
CFLAGS = -Wall -Wextra -Wno-unused -lm -O3 -march=native
 

all: jacobi_mpi jacobi_mpi_conv seidel_mpi seidel_mpi_conv redblack_mpi redblack_mpi_conv

##############
### JACOBI ###
##############

jacobi_mpi: jacobi_mpi.c utils.o
	$(MPICC) $(CFLAGS) jacobi_mpi.c utils.c -o jacobi_mpi

jacobi_mpi_conv: jacobi_mpi.c utils.o
	$(MPICC) $(CFLAGS) jacobi_mpi.c utils.c -o jacobi_mpi_conv -DTEST_CONV

##############
## G.SEIDEL ##
##############

seidel_mpi: gauss_seidel_SOR_mpi.c utils.o
	$(MPICC) $(CFLAGS) gauss_seidel_SOR_mpi.c utils.c -o seidel_mpi

seidel_mpi_conv: gauss_seidel_SOR_mpi.c utils.o
	$(MPICC) $(CFLAGS) gauss_seidel_SOR_mpi.c utils.c -o seidel_mpi_conv -DTEST_CONV

################
## RED-BLACK ##
################

redblack_mpi: red_black_SOR_mpi.c utils.o
	$(MPICC) $(CFLAGS) red_black_SOR_mpi.c utils.c -o redblack_mpi

redblack_mpi_conv: red_black_SOR_mpi.c utils.o
	$(MPICC) $(CFLAGS) red_black_SOR_mpi.c utils.c -o redblack_mpi_conv -DTEST_CONV



################
#### UTILS	####
################

util.o: util.c
	$(MPICC) $(CFLAGS) -c $< -o $@



clean:
	rm -f jacobi_mpi jacobi_mpi_conv seidel_mpi seidel_mpi_conv redblack_mpi redblack_mpi_conv redblacksor_mpi *.o