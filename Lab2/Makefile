CC = mpicc
NODES_NUM = 1 
CORES_NUM = 4 

SIZE = 100
PROC_X = 2 
PROC_Y = 2


FP_LIBS = -lm
GD_LIBS = -lm

FP_OBJS = MPI_Fempois.o
GD_OBJS = GridDist.o

all: MPI_Fempois GridDist

clean:
	rm -f *.o input*-*.dat output*.dat mapping*.dat timer.dat

MPI_Fempois: $(FP_OBJS)
	mpicc -o $@ $(FP_OBJS) $(FP_LIBS)

GridDist: $(GD_OBJS)
	gcc -o $@ $(GD_OBJS) $(GD_LIBS)

MPI_Fempois.o:
	mpicc -c MPI_Fempois.c

GridDist.o:
	gcc -c GridDist.c

run-grid-adapt:
	./GridDist $(PROC_X) $(PROC_Y) $(SIZE) $(SIZE) adapt 

run-grid:
	./GridDist $(PROC_X) $(PROC_Y) $(SIZE) $(SIZE)  
run:
#	mpirun -np 4 MPI_Fempois
	prun -v -np $(NODES_NUM) -$(CORES_NUM) -sge-script $$PRUN_ETC/prun-openmpi ./MPI_Fempois

run-10:
	for i in 1 2 3 4 5 6 7 8 9 10 ; \
			do make run ; \
	done

