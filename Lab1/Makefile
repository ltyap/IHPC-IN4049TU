
NODES_NUM = 1 
CORES_NUM = 4 

SIZE = 100
PROC_X = 2 
PROC_Y = 2

CC = mpicc

FP_LIBS = -lm

clean:
	rm -f *.o 

FP_OBJS = MPI_Poisson.o

all: MPI_Poisson

MPI_Poisson: $(FP_OBJS)
	mpicc -o $@ $(FP_OBJS) $(FP_LIBS)

MPI_Poisson.o:
	mpicc -c MPI_Poisson.c
	
run:
#	mpirun -np 4 MPI_Fempois
	prun -v -np $(NODES_NUM) -$(CORES_NUM) -sge-script $$PRUN_ETC/prun-openmpi ./MPI_Poisson $(PROC_X) $(PROC_Y) $(SIZE) $(SIZE)
