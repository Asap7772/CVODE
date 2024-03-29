//
// Created by Anikait Singh on 2019-08-06.
//

#ifndef FBLIN_H
#define FBLIN_H

void rhside_lsode_kernel(double phi, double *q,double* q_d, double *qdot, double* qdot_d, double rmin, double rmax,
                         double phimin, double phimax, double zmin, double zmax,
                         int nr, int nphi, int nz, double eps1, double eps2, double eps3,
			 double *raxis, double *phiaxis,double *zaxis, double *BR4D, double *BZ4D,
			 double *raxis_d, double *phiaxis_d,double *zaxis_d, double *BR4D_d, double *BZ4D_d,
			 double delta_phi);

#endif //FBLIN_H
