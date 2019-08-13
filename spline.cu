//
// Created by Anikait Singh on 2019-08-01.
// for interpolation
//


//arr[i][j] to arr[j][i]
//r is rows
//c is columns
//static double transposeLookup2D(double *arr, int i, int j, int r, int c) {
//    return arr[j * r + i];
//}

//arr[i][j][k][l] to arr[l][k][j][i]
//s1,s2,s3,s4 is size of i, j, k, l respetively
#ifdef __NVCC__
__device__
#endif
double transposeLookup4D(double *arr, int i, int j, int k, int l, int s1, int s2, int s3, int s4) {
    //j, k, l are one based
    j = j - 1;
    k = k - 1;
    l = l - 1;
    return arr[l * s3 * s2 * s1 + k * s2 * s1 + j * s1 + i];
}

//ivec = 1
#ifdef __NVCC__
__global__
#endif
void r8herm3fcn(int ivec, int ivecd, double *fval, int i, int j, int k, double xp,
                double yp, double zp, double hx, double hxi, double hy, double hyi,
                double hz, double hzi, double *fin, int inf2, int inf3,
                int nz) {

    int iadr;
    double xpi, xp2, xpi2, ax, axbar, bx, bxbar, ypi, yp2, ypi2, ay;
    double aybar, by, bybar, zpi, zp2, zpi2, az, azbar, bz, bzbar, axp;
    double axbarp, bxp, bxbarp, ayp, aybarp, byp, bybarp, azp, azbarp, bzp;
    double bzbarp;

    double sum = 0;
    int ict[8] = {1,1,1,1,0,0,0,0};

    //x
    xpi = 1.0 - xp;
    xp2 = xp * xp;
    xpi2 = xpi * xpi;
    ax = xp2 * (3.0 - 2.0 * xp);
    axbar = 1.0 - ax;
    bx = -xp2 * xpi;
    bxbar = xpi2 * xp;

    //y
    ypi = 1.0 - yp;
    yp2 = yp * yp;
    ypi2 = ypi * ypi;
    ay = yp2 * (3.0 - 2.0 * yp);
    aybar = 1.0 - ay;
    by = -yp2 * ypi;
    bybar = ypi2 * yp;

    //z
    zpi = 1.0 - zp;
    zp2 = zp * zp;
    zpi2 = zpi * zpi;
    az = zp2 * (3.0 - 2.0 * zp);
    azbar = 1.0 - az;
    bz = -zp2 * zpi;
    bzbar = zpi2 * zp;

    iadr = 0;

    //derivatives
    axp = 6.0 * xp * xpi;
    axbarp = -axp;
    bxp = xp * (3.0 * xp - 2.0);
    bxbarp = xpi * (3.0 * xpi - 2.0);

    ayp = 6.0 * yp * ypi;
    aybarp = -ayp;
    byp = yp * (3.0 * yp - 2.0);
    bybarp = ypi * (3.0 * ypi - 2.0);

    azp = 6.0 * zp * zpi;
    azbarp = -azp;
    bzp = zp * (3.0 * zp - 2.0);
    bzbarp = zpi * (3.0 * zpi - 2.0);

    int s1 = 8;
    if (ict[0] == 1) {
        // iadr = iadr + 1;
        sum = azbar * (
                axbar * (aybar * transposeLookup4D(fin, 0, i, j, k, s1, inf2, inf3, nz) +
                         ay * transposeLookup4D(fin, 0, i, j + 1, k, s1, inf2, inf3, nz)) +
                ax * (aybar * transposeLookup4D(fin, 0, i + 1, j, k, s1, inf2, inf3, nz) +
                      ay * transposeLookup4D(fin, 0, i + 1, j + 1, k, s1, inf2, inf3, nz))) +
              +az * (
                      axbar * (aybar * transposeLookup4D(fin, 0, i, j, k + 1, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 0, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                      ax * (aybar * transposeLookup4D(fin, 0, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                            ay * transposeLookup4D(fin, 0, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)));

        sum = sum + hx * (
                azbar * (
                        bxbar * (aybar * transposeLookup4D(fin, 1, i, j, k, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 1, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 1, i + 1, j, k, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 1, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        bxbar * (aybar * transposeLookup4D(fin, 1, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 1, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 1, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 1, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hy * (
                azbar * (
                        axbar * (bybar * transposeLookup4D(fin, 2, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 2, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 2, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 2, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        axbar * (bybar * transposeLookup4D(fin, 2, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 2, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 2, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 2, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hz * (
                bzbar * (
                        axbar * (aybar * transposeLookup4D(fin, 3, i, j, k, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 3, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (aybar * transposeLookup4D(fin, 3, i + 1, j, k, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 3, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        axbar * (aybar * transposeLookup4D(fin, 3, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 3, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (aybar * transposeLookup4D(fin, 3, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 3, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hx * hy * (
                azbar * (
                        bxbar * (bybar * transposeLookup4D(fin, 4, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 4, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 4, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 4, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        bxbar * (bybar * transposeLookup4D(fin, 4, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 4, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 4, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 4, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hx * hz * (
                bzbar * (
                        bxbar * (aybar * transposeLookup4D(fin, 5, i, j, k, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 5, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 5, i + 1, j, k, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 5, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        bxbar * (aybar * transposeLookup4D(fin, 5, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 5, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 5, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 5, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hy * hz * (
                bzbar * (
                        axbar * (bybar * transposeLookup4D(fin, 6, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 6, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 6, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 6, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        axbar * (bybar * transposeLookup4D(fin, 6, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 6, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 6, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 6, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hx * hy * hz * (
                bzbar * (
                        bxbar * (bybar * transposeLookup4D(fin, 7, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 7, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 7, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 7, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        bxbar * (bybar * transposeLookup4D(fin, 7, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 7, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 7, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 7, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        //v is 0 so ignore
        fval[iadr] = sum;
    }

    if (ict[1] == 1) {
        iadr = iadr + 1;

        sum = hxi * (
                azbar * (
                        axbarp * (aybar * transposeLookup4D(fin, 0, i, j, k, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 0, i, j + 1, k, s1, inf2, inf3, nz)) +
                        axp * (aybar * transposeLookup4D(fin, 0, i + 1, j, k, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 0, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        axbarp * (aybar * transposeLookup4D(fin, 0, i, j, k + 1, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 0, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        axp * (aybar * transposeLookup4D(fin, 0, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 0, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + (
                azbar * (
                        bxbarp * (aybar * transposeLookup4D(fin, 1, i, j, k, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 1, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bxp * (aybar * transposeLookup4D(fin, 1, i + 1, j, k, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 1, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        bxbarp * (aybar * transposeLookup4D(fin, 1, i, j, k + 1, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 1, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bxp * (aybar * transposeLookup4D(fin, 1, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 1, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hxi * hy * (
                azbar * (
                        axbarp * (bybar * transposeLookup4D(fin, 2, i, j, k, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 2, i, j + 1, k, s1, inf2, inf3, nz)) +
                        axp * (bybar * transposeLookup4D(fin, 2, i + 1, j, k, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 2, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        axbarp * (bybar * transposeLookup4D(fin, 2, i, j, k + 1, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 2, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        axp * (bybar * transposeLookup4D(fin, 2, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 2, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hxi * hz * (
                bzbar * (
                        axbarp * (aybar * transposeLookup4D(fin, 3, i, j, k, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 3, i, j + 1, k, s1, inf2, inf3, nz)) +
                        axp * (aybar * transposeLookup4D(fin, 3, i + 1, j, k, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 3, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        axbarp * (aybar * transposeLookup4D(fin, 3, i, j, k + 1, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 3, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        axp * (aybar * transposeLookup4D(fin, 3, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 3, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hy * (
                azbar * (
                        bxbarp * (bybar * transposeLookup4D(fin, 4, i, j, k, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 4, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bxp * (bybar * transposeLookup4D(fin, 4, i + 1, j, k, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 4, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        bxbarp * (bybar * transposeLookup4D(fin, 4, i, j, k + 1, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 4, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bxp * (bybar * transposeLookup4D(fin, 4, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 4, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hz * (
                bzbar * (
                        bxbarp * (aybar * transposeLookup4D(fin, 5, i, j, k, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 5, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bxp * (aybar * transposeLookup4D(fin, 5, i + 1, j, k, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 5, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        bxbarp * (aybar * transposeLookup4D(fin, 5, i, j, k + 1, s1, inf2, inf3, nz) +
                                  ay * transposeLookup4D(fin, 5, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bxp * (aybar * transposeLookup4D(fin, 5, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               ay * transposeLookup4D(fin, 5, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hxi * hy * hz * (
                bzbar * (
                        axbarp * (bybar * transposeLookup4D(fin, 6, i, j, k, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 6, i, j + 1, k, s1, inf2, inf3, nz)) +
                        axp * (bybar * transposeLookup4D(fin, 6, i + 1, j, k, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 6, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        axbarp * (bybar * transposeLookup4D(fin, 6, i, j, k + 1, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 6, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        axp * (bybar * transposeLookup4D(fin, 6, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 6, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        sum = sum + hy * hz * (
                bzbar * (
                        bxbarp * (bybar * transposeLookup4D(fin, 7, i, j, k, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 7, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bxp * (bybar * transposeLookup4D(fin, 7, i + 1, j, k, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 7, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        bxbarp * (bybar * transposeLookup4D(fin, 7, i, j, k + 1, s1, inf2, inf3, nz) +
                                  by * transposeLookup4D(fin, 7, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bxp * (bybar * transposeLookup4D(fin, 7, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                               by * transposeLookup4D(fin, 7, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );

        fval[iadr] = sum;
    }

    if (ict[2] == 1) {
        iadr = iadr + 1;
        sum = hyi * (
                azbar * (
                        axbar * (aybarp * transposeLookup4D(fin, 0, i, j, k, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 0, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (aybarp * transposeLookup4D(fin, 0, i + 1, j, k, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 0, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        axbar * (aybarp * transposeLookup4D(fin, 0, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 0, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (aybarp * transposeLookup4D(fin, 0, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 0, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hyi * hx * (
                azbar * (
                        bxbar * (aybarp * transposeLookup4D(fin, 1, i, j, k, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 1, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (aybarp * transposeLookup4D(fin, 1, i + 1, j, k, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 1, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        bxbar * (aybarp * transposeLookup4D(fin, 1, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 1, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (aybarp * transposeLookup4D(fin, 1, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 1, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + (
                azbar * (
                        axbar * (bybarp * transposeLookup4D(fin, 2, i, j, k, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 2, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (bybarp * transposeLookup4D(fin, 2, i + 1, j, k, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 2, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        axbar * (bybarp * transposeLookup4D(fin, 2, i, j, k + 1, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 2, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (bybarp * transposeLookup4D(fin, 2, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 2, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hyi * hz * (
                bzbar * (
                        axbar * (aybarp * transposeLookup4D(fin, 3, i, j, k, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 3, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (aybarp * transposeLookup4D(fin, 3, i + 1, j, k, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 3, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        axbar * (aybarp * transposeLookup4D(fin, 3, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 3, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (aybarp * transposeLookup4D(fin, 3, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 3, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hx * (
                azbar * (
                        bxbar * (bybarp * transposeLookup4D(fin, 4, i, j, k, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 4, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (bybarp * transposeLookup4D(fin, 4, i + 1, j, k, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 4, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + az * (
                        bxbar * (bybarp * transposeLookup4D(fin, 4, i, j, k + 1, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 4, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (bybarp * transposeLookup4D(fin, 4, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 4, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hx * hyi * hz * (
                bzbar * (
                        bxbar * (aybarp * transposeLookup4D(fin, 5, i, j, k, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 5, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (aybarp * transposeLookup4D(fin, 5, i + 1, j, k, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 5, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        bxbar * (aybarp * transposeLookup4D(fin, 5, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ayp * transposeLookup4D(fin, 5, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (aybarp * transposeLookup4D(fin, 5, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ayp * transposeLookup4D(fin, 5, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hz * (
                bzbar * (
                        axbar * (bybarp * transposeLookup4D(fin, 6, i, j, k, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 6, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (bybarp * transposeLookup4D(fin, 6, i + 1, j, k, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 6, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        axbar * (bybarp * transposeLookup4D(fin, 6, i, j, k + 1, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 6, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (bybarp * transposeLookup4D(fin, 6, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 6, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hx * hz * (
                bzbar * (
                        bxbar * (bybarp * transposeLookup4D(fin, 7, i, j, k, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 7, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (bybarp * transposeLookup4D(fin, 7, i + 1, j, k, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 7, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bz * (
                        bxbar * (bybarp * transposeLookup4D(fin, 7, i, j, k + 1, s1, inf2, inf3, nz) +
                                 byp * transposeLookup4D(fin, 7, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (bybarp * transposeLookup4D(fin, 7, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              byp * transposeLookup4D(fin, 7, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        fval[iadr] = sum;
    }
    if (ict[3] == 1) {
        iadr = iadr + 1;
        sum = hzi * (
                azbarp * (
                        axbar * (aybar * transposeLookup4D(fin, 0, i, j, k, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 0, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (aybar * transposeLookup4D(fin, 0, i + 1, j, k, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 0, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + azp * (
                        axbar * (aybar * transposeLookup4D(fin, 0, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 0, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (aybar * transposeLookup4D(fin, 0, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 0, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hzi * hx * (
                azbarp * (
                        bxbar * (aybar * transposeLookup4D(fin, 1, i, j, k, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 1, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 1, i + 1, j, k, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 1, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + azp * (
                        bxbar * (aybar * transposeLookup4D(fin, 1, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 1, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 1, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 1, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hzi * hy * (
                azbarp * (
                        axbar * (bybar * transposeLookup4D(fin, 2, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 2, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 2, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 2, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + azp * (
                        axbar * (bybar * transposeLookup4D(fin, 2, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 2, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 2, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 2, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + (
                bzbarp * (
                        axbar * (aybar * transposeLookup4D(fin, 3, i, j, k, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 3, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (aybar * transposeLookup4D(fin, 3, i + 1, j, k, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 3, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bzp * (
                        axbar * (aybar * transposeLookup4D(fin, 3, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 3, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (aybar * transposeLookup4D(fin, 3, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 3, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hzi * hx * hy * (
                azbarp * (
                        bxbar * (bybar * transposeLookup4D(fin, 4, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 4, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 4, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 4, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + azp * (
                        bxbar * (bybar * transposeLookup4D(fin, 4, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 4, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 4, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 4, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hx * (
                bzbarp * (
                        bxbar * (aybar * transposeLookup4D(fin, 5, i, j, k, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 5, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 5, i + 1, j, k, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 5, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bzp * (
                        bxbar * (aybar * transposeLookup4D(fin, 5, i, j, k + 1, s1, inf2, inf3, nz) +
                                 ay * transposeLookup4D(fin, 5, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (aybar * transposeLookup4D(fin, 5, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              ay * transposeLookup4D(fin, 5, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hy * (
                bzbarp * (
                        axbar * (bybar * transposeLookup4D(fin, 6, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 6, i, j + 1, k, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 6, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 6, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bzp * (
                        axbar * (bybar * transposeLookup4D(fin, 6, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 6, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        ax * (bybar * transposeLookup4D(fin, 6, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 6, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        sum = sum + hx * hy * (
                bzbarp * (
                        bxbar * (bybar * transposeLookup4D(fin, 7, i, j, k, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 7, i, j + 1, k, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 7, i + 1, j, k, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 7, i + 1, j + 1, k, s1, inf2, inf3, nz)))
                + bzp * (
                        bxbar * (bybar * transposeLookup4D(fin, 7, i, j, k + 1, s1, inf2, inf3, nz) +
                                 by * transposeLookup4D(fin, 7, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
                        bx * (bybar * transposeLookup4D(fin, 7, i + 1, j, k + 1, s1, inf2, inf3, nz) +
                              by * transposeLookup4D(fin, 7, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)))
        );
        fval[iadr] = sum;
    }
}
