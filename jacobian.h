//
// Created by Anikait Singh on 2019-08-06.
//

#ifndef JACOBIAN_H
#define JACOBIAN_H

void jacobian_lsode_kernelC(int neq, double phi, double *q, double *pd, int nrpd, double rmin, double rmax, double phimin, double phimax,
                            double zmin, double zmax, int nr, int nphi, int nz, double eps1, double eps2, double eps3,
                            double *raxis, double *phiaxis, double *zaxis, double *BR4D, double *BZ4D,
                            double delta_phi);

#endif //JACOBIAN_H
