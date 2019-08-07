all :cvodeTest.o jacobian.o fblin.o spline.o

cvodeTest.o : cvodeTest.c jacobian.h fblin.h
	gcc -Wall -o $@ -c $<

jacobian.o : jacobian.c spline.h
	gcc -Wall -o $@ -c $<

fblin.o : fblin.c spline.h
	gcc -Wall -o $@ -c $<

spline.o : spline.c
	gcc -Wall -o $@ -c $<

clean:
	rm *.o
