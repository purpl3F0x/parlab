#include "mpi.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void GaussSeidel(double** u_previous,
                 double** u_current,
                 int      X_min,
                 int      X_max,
                 int      Y_min,
                 int      Y_max,
                 double   omega) {
    int i, j;
    for (i = X_min; i < X_max; i++)
        for (j = Y_min; j < Y_max; j++)
            u_current[i][j] =
              u_previous[i][j] + (u_current[i - 1][j] + u_previous[i + 1][j] + u_current[i][j - 1] +
                                  u_previous[i][j + 1] - 4 * u_previous[i][j]) *
                                   omega / 4.0;
}

int main(int argc, char** argv) {
    int rank, size;
    int global[2],
      local[2]; //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded
      [2];       //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2]; //processor grid dimensions
    int i, j, t;
    int global_converged = 0, converged = 0; //flags for convergence, global and per process
    MPI_Datatype dummy; //dummy datatype used to align user-defined datatypes in memory
    double       omega; //relaxation factor - useless for Jacobi


    struct timeval tts, ttf, tcs, tcf, tcvs,
      tcvf; //Timers: total-> tts,ttf, computation -> tcs,tcf, convergence -> tcvs, tcvf

    double ttotal = 0, tcomp = 0, tconv = 0, total_time, comp_time, conv_time;

    double **
      U = NULL,
    **u_current, **u_previous,
    **swap; //Global matrix, local current and previous matrices, pointer to swap between current and previous

    MPI_Status status;

    MPI_Request prev_request[6];
    MPI_Status  prev_status[6];
    MPI_Request current_request[2];
    MPI_Status  current_status[2];
    int         prev_request_len    = 0;
    int         current_request_len = 0;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //----Read 2D-domain dimensions and process grid dimensions from stdin----//

    if (argc != 5) {
        fprintf(stderr, "Usage: mpirun .... ./exec X Y Px Py");
        exit(-1);
    } else {
        global[0] = atoi(argv[1]);
        global[1] = atoi(argv[2]);
        grid[0]   = atoi(argv[3]);
        grid[1]   = atoi(argv[4]);
    }

    //----Create 2D-cartesian communicator----//
    //----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;             //CART_COMM: the new 2D-cartesian communicator
    int      periods[2] = { 0, 0 }; //periods={0,0}: the 2D-grid is non-periodic
    int      rank_grid[2]; //rank_grid: the position of each process on the new communicator

    MPI_Cart_create(MPI_COMM_WORLD, 2, grid, periods, 0, &CART_COMM); //communicator creation
    MPI_Cart_coords(CART_COMM, rank, 2, rank_grid); //rank mapping on the new communicator

    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//

    for (i = 0; i < 2; i++) {
        if (global[i] % grid[i] == 0) {
            local[i]         = global[i] / grid[i];
            global_padded[i] = global[i];
        } else {
            local[i]         = (global[i] / grid[i]) + 1;
            global_padded[i] = local[i] * grid[i];
        }
    }

    //Initialization of omega
    omega = 2.0 / (1 + sin(3.14 / global[0]));


    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
    if (rank == 0) {
        U = allocate2d(global_padded[0], global_padded[1]);
        init2d(U, global[0], global[1]);
    }

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

    u_previous = allocate2d(local[0] + 2, local[1] + 2);
    u_current  = allocate2d(local[0] + 2, local[1] + 2);

    //----Distribute global 2D-domain from rank 0 to all processes----//

    //----Appropriate datatypes are defined here----//
    /*****The usage of datatypes is optional*****/

    //----Datatype definition for the 2D-subdomain on the global matrix----//

    MPI_Datatype global_block;
    MPI_Type_vector(local[0], local[1], global_padded[1], MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &global_block);
    MPI_Type_commit(&global_block);

    //----Datatype definition for the 2D-subdomain on the local matrix----//

    MPI_Datatype local_block;
    MPI_Type_vector(local[0], local[1], local[1] + 2, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &local_block);
    MPI_Type_commit(&local_block);

    //----Rank 0 defines positions and counts of local blocks (2D-subdomains) on global matrix----//
    int *scatteroffset = NULL, *scattercounts = NULL;
    if (rank == 0) {
        scatteroffset = (int*)malloc(size * sizeof(int));
        scattercounts = (int*)malloc(size * sizeof(int));
        for (i = 0; i < grid[0]; i++)
            for (j = 0; j < grid[1]; j++) {
                scattercounts[i * grid[1] + j] = 1;
                scatteroffset[i * grid[1] + j] = (local[0] * local[1] * grid[1] * i + local[1] * j);
            }
    }


    //----Rank 0 scatters the global matrix----//

    //*************TODO*******************//

    /*Fill your code here*/

    /*Make sure u_current and u_previous are both initialized*/
    zero2d(u_previous, local[0] + 2, local[1] + 2);
    zero2d(u_current, local[0] + 2, local[1] + 2);

    // clang-format off
    double* U0_ptr = (rank == 0) ? &(U[0][0]) : NULL;
    MPI_Scatterv(U0_ptr, scattercounts, scatteroffset, global_block, &(u_previous[1][1]), 1, local_block, 0, MPI_COMM_WORLD);
    MPI_Scatterv(U0_ptr, scattercounts, scatteroffset, global_block, &(u_current[1][1]), 1, local_block, 0, MPI_COMM_WORLD);
    // clang-format on


    //************************************//


    if (rank == 0)
        free2d(U);


    //----Define datatypes or allocate buffers for message passing----//

    //*************TODO*******************//

    /*Fill your code here*/
    MPI_Datatype column;
    MPI_Type_vector(local[0], 1, local[1] + 2, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &column);
    MPI_Type_commit(&column);


    //************************************//


    //----Find the 4 neighbors with which a process exchanges messages----//

    //*************TODO*******************//
    int north, south, east, west;

    /*Fill your code here*/
    MPI_Cart_shift(CART_COMM, 0, 1, &north, &south);
    MPI_Cart_shift(CART_COMM, 1, 1, &west, &east);


    /*Make sure you handle non-existing neighbors appropriately*/


    //************************************//


    //---Define the iteration ranges per process-----//
    //*************TODO*******************//

    int i_min, i_max, j_min, j_max;


    /*Fill your code here*/


    /*Three types of ranges:
                -internal processes
                -boundary processes
                -boundary processes and padded global array
        */

    /* internal process (ghost cell only) */
    i_min = 1;
    i_max = local[0] + 1;

    /* boundary process - no possible padding */
    if (north == MPI_PROC_NULL) {
        i_min = 2; // ghost cell + boundary
    }

    /* boundary process and padded global array */
    if (south == MPI_PROC_NULL) {
        i_max -= (global_padded[0] - global[0]) + 1;
    }

    /* internal process (ghost cell only) */
    j_min = 1;
    j_max = local[1] + 1;

    /* boundary process - no possible padding */
    if (west == MPI_PROC_NULL) {
        j_min = 2; //ghost cell + boundary
    }

    /* boundary process and padded global array */
    if (east == MPI_PROC_NULL) {
        j_max -= (global_padded[1] - global[1]) + 1;
    }


    //************************************//


    //----Computational core----//
    gettimeofday(&tts, NULL);
#ifdef TEST_CONV
    for (t = 0; t < T && !global_converged; t++) {
#else
    #undef T
    #define T 256
    for (t = 0; t < T; t++) {
#endif


        //*************TODO*******************//


        /*Fill your code here*/

        swap       = u_previous;
        u_previous = u_current;
        u_current  = swap;

        /*Compute and Communicate*/

        current_request_len = 0;
        prev_request_len    = 0;


        // Communicate with north
        if (north != MPI_PROC_NULL) {
            MPI_Isend(&u_previous[1][1],
                      local[1],
                      MPI_DOUBLE,
                      north,
                      0,
                      MPI_COMM_WORLD,
                      &prev_request[prev_request_len]);
            MPI_Irecv(&u_current[0][1],
                      local[1],
                      MPI_DOUBLE,
                      north,
                      0,
                      MPI_COMM_WORLD,
                      &prev_request[prev_request_len + 1]);

            prev_request_len += 2;
        }

        // Communicate with west
        if (west != MPI_PROC_NULL) {
            MPI_Isend(&u_previous[1][1],
                      1,
                      column,
                      west,
                      0,
                      MPI_COMM_WORLD,
                      &prev_request[prev_request_len]);
            MPI_Irecv(&u_current[1][0],
                      1,
                      column,
                      west,
                      0,
                      MPI_COMM_WORLD,
                      &prev_request[prev_request_len + 1]);

            prev_request_len += 2;
        }

        // Communicate with south
        if (south != MPI_PROC_NULL) {
            MPI_Irecv(&u_previous[local[0] + 1][1],
                      local[1],
                      MPI_DOUBLE,
                      south,
                      0,
                      MPI_COMM_WORLD,
                      &prev_request[prev_request_len]);

            prev_request_len += 1;
        }

        // Communicate with east
        if (east != MPI_PROC_NULL) {
            MPI_Irecv(&u_previous[1][local[1] + 1],
                      1,
                      column,
                      east,
                      0,
                      MPI_COMM_WORLD,
                      &prev_request[prev_request_len]);

            prev_request_len += 1;
        }

        MPI_Waitall(prev_request_len, prev_request, prev_status);

        /*Add appropriate timers for computation*/

        gettimeofday(&tcs, NULL);
        {
            GaussSeidel(u_previous, u_current, i_min, i_max, j_min, j_max, omega);
        }
        gettimeofday(&tcf, NULL);
        tcomp += (tcf.tv_sec - tcs.tv_sec) + (tcf.tv_usec - tcs.tv_usec) * 0.000001;

        if (south != MPI_PROC_NULL) {
                MPI_Isend(&u_current[local[0]][1], local[1], MPI_DOUBLE, south, 0, MPI_COMM_WORLD, &current_request[current_request_len]);
            current_request_len += 1;
        }

        if (east != MPI_PROC_NULL){
            MPI_Isend(&u_current[1][local[1]], 1, column, east, 0, MPI_COMM_WORLD, &current_request[current_request_len]);
            current_request_len += 1;
        }

        MPI_Waitall(current_request_len, current_request, current_status);

#ifdef TEST_CONV
        if (t % C == 0) {
            //*************TODO**************//
            /*Test convergence*/
            gettimeofday(&tcvs, NULL);
            converged = converge(u_previous, u_current, i_min, i_max, j_min, j_max);

            MPI_Allreduce(&converged, &global_converged, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            gettimeofday(&tcvf, NULL);

            tconv += (tcvf.tv_sec - tcvs.tv_sec) + (tcvf.tv_usec - tcvs.tv_usec) * 0.000001;
        }
#endif


        //************************************//
    }
    gettimeofday(&ttf, NULL);

    ttotal = (ttf.tv_sec - tts.tv_sec) + (ttf.tv_usec - tts.tv_usec) * 0.000001;

    MPI_Reduce(&ttotal, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tcomp, &comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tconv, &conv_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    //----Rank 0 gathers local matrices back to the global matrix----//

    if (rank == 0) {
        U      = allocate2d(global_padded[0], global_padded[1]);
        U0_ptr = &(U[0][0]);
    } else {
        U0_ptr = NULL;
    }


    //*************TODO*******************//


    /*Fill your code here*/

    MPI_Gatherv(&u_current[1][1],
                1,
                local_block,
                U0_ptr,
                scattercounts,
                scatteroffset,
                global_block,
                0,
                MPI_COMM_WORLD);

    //************************************//


    //----Printing results----//

    if (rank == 0) {
        printf("Gauss-Seidel X=%d, Y=%d, Workers=%d, Px=%d, Py=%d, Iter=%d, ComputationTime=%lf, "
               "TotalTime=%lf, "
               "midpoint=%lf\n",
               global[0],
               global[1],
               size,
               grid[0],
               grid[1],
               t,
               comp_time,
               total_time,
               U[global[0] / 2][global[1] / 2]);

#ifdef PRINT_RESULTS
        char* s = malloc(50 * sizeof(char));
        sprintf(s, "resSeidelMPI_%dx%d_%dx%d", global[0], global[1], grid[0], grid[1]);
        fprint2d(s, U, global[0], global[1]);
        free(s);
#endif
    }

    MPI_Finalize();
    return 0;
}