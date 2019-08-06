all:
	OBJLIST = /Users/anikaitsingh/CLionProjects/CVODE_Functions/ObjectList

	$(LINK) $@ $(ObjectFiles) $(LIB_LINK) \

	jacobian.o: 	jacobian.c
		gcc -Wall -o $@ -c $^
	spline.o: 	spline.c
		gcc -Wall -o $@ -c $^
	fblin.o:	fblin.c
		gcc -Wall -o $@ -c $^
clean:
	- rm -f *.o *.mod