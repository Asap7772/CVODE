#include <stdio.h>
#include <math.h>
#include "spline.h"
// Used to Calculate RHS
// Created by Anikait Singh on 2019-08-01.
//

/* double max(double one, double two) { */
/*     return one > two ? one : two; */
/* } */

/* double min(double one, double two) { */
/*     return one > two ? two : one; */
/* } */

/*
void rhside_lsode_kernel_(double *phi_pointer, double *q, double *qdot, double *rmin_pointer, double *rmax_pointer,
                          double *phimin_pointer, double *phimax_pointer, double *zmin_pointer, double *zmax_pointer,
                          int *nr_pointer, int *nphi_pointer, int *nz_pointer, double *eps1_pointer,
                          double *eps2_pointer, double *eps3_pointer, double *raxis, double *phiaxis,
                          double *zaxis, double *BR4D, double *BZ4D, double *delta_phi_pointer) {
    double phi = *phi_pointer;
    double rmin = *rmin_pointer;
    double rmax = *rmax_pointer;
    double phimin = *phimin_pointer;
    double phimax = *phimax_pointer;
    double zmin = *zmin_pointer;
    double zmax = *zmax_pointer;
    int nr = *nr_pointer;
    int nphi = *nphi_pointer;
    int nz = *nz_pointer;
    double eps1 = *eps1_pointer;
    double eps2 = *eps2_pointer;
    double eps3 = *eps3_pointer;
    double delta_phi = *delta_phi_pointer;


    const int ivec = 1; //loop of one implemented in case
    int ier;
    double r_temp, phi_temp, z_temp, br_temp, bz_temp, hy, hz, hyi, hzi, one = 1;
    double fval[1], xparam, yparam, zparam, hx, hxi;
    int i, j, k;
    int ict[8] = {1, 1, 1, 1, 0, 0, 0, 0};

    ier = 0;
    r_temp = q[0];
    z_temp = q[1];
    phi_temp = fmod(phi, delta_phi);

    if (phi_temp < 0) {
        phi_temp = delta_phi + phi_temp;
    }

    br_temp = 0.0;
    bz_temp = 0.0;
    if ((r_temp >= rmin - eps1) && (r_temp <= rmax + eps1) &&
        (phi_temp >= phimin - eps2) && (phi_temp <= phimax + eps2) &&
        (z_temp >= zmin - eps3) && (z_temp <= zmax + eps3)) {
        int count = 0;
        for (int i = 0; i < nr; i++) {
            if (raxis[i] < r_temp) {
                count++;
            }
        }
        i = fmin(fmax(count, 1), nr - 1);

        count = 0;
        for (int i = 0; i < nphi; i++) {
            if (phiaxis[i] < phi_temp) {
                count++;
            }
        }
        j = fmin(fmax(count, 1), nphi - 1);

        count = 0;
        for (int i = 0; i < nz; i++) {
            if (zaxis[i] < z_temp) {
                count++;
            }
        }
        k = fmin(fmax(count, 1), nz - 1);
        hx = raxis[i] - raxis[i - 1];
        hy = phiaxis[j] - phiaxis[j - 1];
        hz = zaxis[k] - zaxis[k - 1];
        hxi = one / hx;
        hyi = one / hy;
        hzi = one / hz;
        xparam = (r_temp - raxis[i - 1]) * hxi;
        yparam = (phi_temp - phiaxis[j - 1]) * hyi;
        zparam = (z_temp - zaxis[k - 1]) * hzi;
        r8herm3fcn(ict, ivec, 1, fval, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BR4D, nr, nphi, nz);
        br_temp = fval[0];
        r8herm3fcn(ict, ivec, 1, fval, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BZ4D, nr, nphi, nz);
        bz_temp = fval[0];
    }

    qdot[0] = br_temp;
    qdot[1] = bz_temp;
}
*/

void rhside_lsode_kernel(double phi, double *q, double *q_d, double *qdot, double *qdot_d, double rmin, double rmax,
                          double phimin, double phimax, double zmin, double zmax,
                          int nr, int nphi, int nz, double eps1,
                          double eps2, double eps3, double *raxis, double *phiaxis,
                          double *zaxis, double *BR4D, double *BZ4D,double *raxis_d,
			  double *phiaxis_d, double *zaxis_d, double *BR4D_d, double *BZ4D_d, double delta_phi) {

    const int ivec = 1; //loop of one implemented in case
    int ier;
    double r_temp, phi_temp, z_temp, br_temp, bz_temp, hy, hz, hyi, hzi, one = 1;
    double fval[1], xparam, yparam, zparam, hx, hxi;
    int i, j, k;
    //int ict[8] = {1, 1, 1, 1, 0, 0, 0, 0};

    ier = 0;
    r_temp = q[0];
    z_temp = q[1];
    phi_temp = fmod(phi, delta_phi);

    if (phi_temp < 0) {
        phi_temp = delta_phi + phi_temp;
    }

    br_temp = 0.0;
    bz_temp = 0.0;
    if ((r_temp >= rmin - eps1) && (r_temp <= rmax + eps1) &&
        (phi_temp >= phimin - eps2) && (phi_temp <= phimax + eps2) &&
        (z_temp >= zmin - eps3) && (z_temp <= zmax + eps3)) {
        int count = 0;
        for (int ii = 0; ii < nr; ii++) {
            if (raxis[ii] < r_temp) {
                count++;
            }
        }
        i = fmin(fmax(count, 1), nr - 1);

        count = 0;
        for (int ii = 0; ii < nphi; ii++) {
            if (phiaxis[ii] < phi_temp) {
                count++;
            }
        }
        j = fmin(fmax(count, 1), nphi - 1);

        count = 0;
        for (int ii = 0; ii < nz; ii++) {
            if (zaxis[ii] < z_temp) {
                count++;
            }
        }
        k = fmin(fmax(count, 1), nz - 1);

        hx = raxis[i] - raxis[i - 1];
        hy = phiaxis[j] - phiaxis[j - 1];
        hz = zaxis[k] - zaxis[k - 1];

        hxi = one / hx;
        hyi = one / hy;
        hzi = one / hz;

        xparam = (r_temp - raxis[i - 1]) * hxi;
        yparam = (phi_temp - phiaxis[j - 1]) * hyi;
        zparam = (z_temp - zaxis[k - 1]) * hzi;

	double *fval_d;
	cudaMalloc(&fval_d, sizeof(fval));
	cudaMemcpy(fval_d, fval, sizeof(fval), cudaMemcpyHostToDevice);

        r8herm3fcn<<<1,1>>>(ivec, 1, fval_d, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BR4D_d, nr, nphi, nz);
	cudaMemcpy(fval, fval_d, sizeof(fval), cudaMemcpyDeviceToHost);	
        br_temp = fval[0];

        r8herm3fcn<<<1,1>>>(ivec, 1, fval_d, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BZ4D_d, nr, nphi, nz);
	cudaMemcpy(fval, fval_d, sizeof(fval), cudaMemcpyDeviceToHost);		
        bz_temp = fval[0];
    }

    qdot[0] = br_temp;
    qdot[1] = bz_temp;
}
