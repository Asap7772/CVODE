//
// Created by Anikait Singh on 2019-08-05.
//

#ifndef SPLINE_H
#define SPLINE_H
__global__
void r8herm3fcn(int ivec, int ivecd, double *fval, int i, int j, int k, double xp,
                double yp, double zp, double hx, double hxi, double hy, double hyi,
                double hz, double hzi, double *fin, int inf2, int inf3,
                int nz);

#endif //SPLINE_H
