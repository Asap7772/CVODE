#include <stdio.h>
#include <math.h>
#include "spline.h"
// Used to Calculate Jacobian
// Created by Anikait Singh on 2019-08-01.
//

void jacobian_lsode_kernelC(int neq, double phi, double *q,double *q_d, double *pd, int nrpd, double rmin, double rmax, double phimin,
                            double phimax, double zmin, double zmax, int nr, int nphi, int nz, double eps1, double eps2, double eps3,
                            double *raxis, double *phiaxis,double *zaxis, double *BR4D, double *BZ4D,double *raxis_d,
                            double *phiaxis_d, double *zaxis_d, double *BR4D_d, double *BZ4D_d, double delta_phi) {

    int ier, i, j, k;
    double r_temp, phi_temp, z_temp, xparam,
            yparam, zparam, hx, hy, hz, hxi, hyi, hzi, one = 1;
    double fval[4];
    //int ict[8] = {1, 1, 1, 1, 0, 0, 0, 0};

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

        r8herm3fcn<<<1,1>>>(1, 1, fval_d, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BR4D_d, nr, nphi, nz);
//        //dBR/dR F had (1,1) transposed for C
        cudaMemcpy(fval, fval_d, sizeof(fval), cudaMemcpyDeviceToHost);
        pd[0] = fval[1];
//        //dBR/dZ F had (1,2) transposed for C
        pd[2] = fval[3];


        r8herm3fcn<<<1,1>>>(1, 1, fval_d, i, j, k, xparam, yparam, zparam, hx, hxi, hy, hyi, hz, hzi, BZ4D_d, nr, nphi, nz);
        cudaMemcpy(fval, fval_d, sizeof(fval), cudaMemcpyDeviceToHost);
//        //dBZ/dR F had (2,1) transposed for C
        pd[1] = fval[1];
//        //dBZ/dZ F had (2,2) transposed for C
        pd[3] = fval[3];
        cudaFree(fval_d);
    }
}
