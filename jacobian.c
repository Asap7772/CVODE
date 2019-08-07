#include <stdio.h>
#include <math.h>
#include "spline.h"
// Used to Calculate Jacobian
// Created by Anikait Singh on 2019-08-01.
//

/* double max(double one, double two) { */
/*     return one > two ? one : two; */
/* } */

/* double min(double one, double two) { */
/*     return one > two ? two : one; */
/* } */

void jacobian_lsode_kernelC_(int *neq_pointer, double *phi_pointer, double *q, int *ml_pointer, int *mp_pointer,
                             double *pd, int *nrpd_pointer, double *rmin_pointer, double *rmax_pointer,
                             double *phimin_pointer, double *phimax_pointer, double *zmin_pointer, double *zmax_pointer,
                             int *nr_pointer, int *nphi_pointer, int *nz_pointer, double *eps1_pointer,
                             double *eps2_pointer, double *eps3_pointer, double *raxis, double *phiaxis,
                             double *zaxis, double *BR4D, double *BZ4D, double *delta_phi_pointer) {
    int neq = *neq_pointer;
    double phi = *phi_pointer;
    int ml = *ml_pointer;
    int mp = *mp_pointer;
    int nrpd = *nrpd_pointer;
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


    int ier, i, j, k;
    double r_temp, phi_temp, z_temp, xparam,
            yparam, zparam, hx, hy, hz, hxi, hyi, hzi, one = 1;
    double fval[4];
    int ict[8] = {1, 1, 1, 1, 0, 0, 0, 0};

    ier = 0;
    r_temp = q[0];
    z_temp = q[1];
    phi_temp = fmod(phi, delta_phi);

    if (phi_temp < 0) {
        phi_temp = delta_phi + phi_temp;
    }

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
        r8herm3fcn(ict, 1, 1, fval, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BR4D, nr, nphi, nz);
//        //dBR/dR F had (1,1) transposed for C
        pd[0] = fval[1];
//        //dBR/dZ F had (1,2) transposed for C
        pd[2] = fval[3];
        r8herm3fcn(ict, 1, 1, fval, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BZ4D, nr, nphi, nz);
//        //dBZ/dR F had (2,1) transposed for C
        pd[1] = fval[1];
//        //dBZ/dZ F had (2,2) transposed for C
        pd[3] = fval[3];
    }
}

void jacobian_lsode_kernelC(int neq, double phi, double *q, double *pd, int nrpd, double rmin, double rmax, double phimin, double phimax,
                            double zmin, double zmax, int nr, int nphi, int nz, double eps1, double eps2, double eps3,
                            double *raxis, double *phiaxis, double *zaxis, double *BR4D, double *BZ4D,
                            double delta_phi) {

    int ier, i, j, k;
    double r_temp, phi_temp, z_temp, xparam,
            yparam, zparam, hx, hy, hz, hxi, hyi, hzi, one = 1;
    double fval[4];
    int ict[8] = {1, 1, 1, 1, 0, 0, 0, 0};

    ier = 0;
    r_temp = q[0];
    z_temp = q[1];
    phi_temp = fmod(phi, delta_phi);

    if (phi_temp < 0) {
        phi_temp = delta_phi + phi_temp;
    }

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
        r8herm3fcn(ict, 1, 1, fval, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BR4D, nr, nphi, nz);
//        //dBR/dR F had (1,1) transposed for C
        pd[0] = fval[1];
//        //dBR/dZ F had (1,2) transposed for C
        pd[2] = fval[3];
        r8herm3fcn(ict, 1, 1, fval, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BZ4D, nr, nphi, nz);
//        //dBZ/dR F had (2,1) transposed for C
        pd[1] = fval[1];
//        //dBZ/dZ F had (2,2) transposed for C
        pd[3] = fval[3];
    }
}
