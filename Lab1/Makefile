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
	
