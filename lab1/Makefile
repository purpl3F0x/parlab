CFLAGS += -O3 --fast-math -march=native -Wall --openmp -std=c99

all: game_of_life

game_of_life: Game_Of_Life.c
	gcc $(CFLAGS) -o game_of_life Game_Of_Life.c

clean:
	rm game_of_life
