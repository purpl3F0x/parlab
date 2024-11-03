.phony: all clean

all: fw fw_sr fw_tiled 

CC=gcc
CFLAGS= -Wall -Wextra -O2 -march=native -ffast-math

HDEPS+=%.h

OBJS=util.o

fw: $(OBJS) fw.c 
	$(CC) $(OBJS) fw.c -o fw $(CFLAGS)
fw_sr: fw_sr.c 
	$(CC) $(OBJS) fw_sr.c -o fw_sr $(CFLAGS) -fopenmp
fw_tiled: fw_tiled.c 
	$(CC) $(OBJS) fw_tiled.c -o fw_tiled $(CFLAGS) -fopenmp

%.o: %.c $(HDEPS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o fw fw_sr fw_tiled 

