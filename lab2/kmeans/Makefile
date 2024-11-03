.KEEP_STATE:

CC = gcc
CFLAGS = -Wall -Wextra -O2 --fast-math -D_NO_LOG
OMPFLAGS = -fopenmp $(CFLAGS)
LDFLAGS = 
H_FILES = kmeans.h
COMM_SRC = file_io.c util.c

_NUMA_AWARE ?= 0
ifeq ($(_NUMA_AWARE), 1)
	OMPFLAGS += -D_NUMA_AWARE
endif

# all: kmeans_seq
all: kmeans_seq kmeans_omp_naive kmeans_omp_reduction

kmeans_seq: main.o file_io.o util.o seq_kmeans.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_naive: main.o file_io_omp.o util.o omp_naive_kmeans.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_reduction: main.o file_io_omp.o util.o omp_reduction_kmeans.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)


main.o: main.c $(H_FILES)
	$(CC) $(CFLAGS) -c $< -o $@

seq_kmeans.o: seq_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(CFLAGS) -c $< -o $@
omp_naive_kmeans.o: omp_naive_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(OMPFLAGS) -c $< -o $@
omp_reduction_kmeans.o: omp_reduction_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(OMPFLAGS) -c $< -o $@

file_io_omp.o: file_io.c
	$(CC) $(OMPFLAGS) -c $< -o $@

file_io.o: file_io.c
	$(CC) $(CFLAGS) -c $< -o $@

util.o: util.c
	$(CC) $(CFLAGS) -c $< -o $@

dep: $(_NUMA_AWARE)

clean:
	rm -rf *.o kmeans_seq kmeans_omp_naive kmeans_omp_reduction
