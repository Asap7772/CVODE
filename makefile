CVODE_ROOT= /u/asingh2/cvodeLib/instdir
CVODE_LIBS = $(CVODE_ROOT)/lib64
CVODE_INCS = $(CVODE_ROOT)/include
# PathCVODE = /u/asingh2/cvodeLib/instdir/include/cvode
# PathNvector = /u/asingh2/cvodeLib/instdir/include/nvector
# PathSundials = /u/asingh2/cvodeLib/instdir/include/sundials
# PathLinSolve = /u/asingh2/cvodeLib/instdir/include/sunlinsol
# PathMatrix = /u/asingh2/cvodeLib/instdir/include/sunmatrix
# PathNonLinSolve = /u/asingh2/cvodeLib/instdir/include/sunnonlinsol

INCLUDES = -I$(CVODE_INCS)/cvode \
	-I$(CVODE_INCS)/nvector \
	-I$(CVODE_INCS)/sundials \
	-I$(CVODE_INCS)/sunlinsol \
	-I$(CVODE_INCS)/sunmatrix \
	-I$(CVODE_INCS)/sunnonlinsol
LIBRARIES = -L$(CVODE_LIBS) -lsundials_cvode -lsundials_nveccuda -lsundials_nvecserial -lsundials_sunlinsolband -lsundials_sunlinsoldense \
	-lsundials_sunlinsolpcg -lsundials_sunlinsolspbcgs -lsundials_sunlinsolspfgmr -lsundials_sunlinsolspgmr \
	-lsundials_sunlinsolsptfqmr -lsundials_sunmatrixband -lsundials_sunmatrixdense -lsundials_sunmatrixsparse \
	 -lsundials_sunnonlinsolfixedpoint -lsundials_sunnonlinsolnewton

CFLAGS = -Wall -std=c99 -fpic

all :cvodeTest.o jacobian.o fblin.o spline.o cvodeTest.so

cvodeTest.so: cvodeTest.o jacobian.o fblin.o spline.o
	gcc -shared -o $@ $^ $(LIBRARIES)

cvodeTest.o : cvodeTest.c jacobian.h fblin.h
	gcc $(CFLAGS) -o $@ -c $< $(INCLUDES)

jacobian.o : jacobian.c spline.h
	gcc $(CFLAGS) -o $@ -c $< $(INCLUDES)

fblin.o : fblin.c spline.h
	gcc $(CFLAGS) -o $@ -c $< $(INCLUDES)

spline.o : spline.c
	gcc $(CFLAGS)  -o $@ -c $< $(INCLUDES)

clean:
	rm *.o

