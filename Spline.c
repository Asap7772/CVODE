//
// Created by Anikait Singh on 2019-08-01.
// for interpolation
//

//ivec = 1
//arr[i][j] to arr[j][i]
//r is rows
//c is columns
double transposeLookup2D(double *arr, int i, int j, int r, int c) {
    return arr[j * r + i];
}

//arr[i][j][k][l] to arr[l][k][j][i]
//s1,s2,s3,s4 is size of i, j, k, l respetively
double transposeLookup4D(double *arr, int i, int j, int k, int l, int s1, int s2, int s3, int s4) {
    return arr[l * s3 * s2 * s1 + k * s2 * s1 + j * s1 + i];
}

void
r8herm3fcn(int *ict, int *ivec_pointer, int *ivecd_pointer, double *fval, int *ii, int *jj, int *kk, double *xparam,
           double *yparam, double *zparam, double *hx, double *hxi, double *hy_pointer, double *hyi_pointer,
           double *hz_pointer, double *hzi_pointer, double *fin, int *inf2_pointer, int *inf3_pointer, int *nz_pointer) {
    int ivec = *ivec_pointer;
    int ivecd = *ivecd_pointer;
    int inf2 = *inf2_pointer;
    int inf3 = *inf3_pointer;
    int nz = *nz_pointer;

    double hy = *hy_pointer;
    double hyi = *hyi_pointer;
    double hz = *hz_pointer;
    double hzi = *hzi_pointer;

    int i, j, k, iadr;
    double xp, xpi, xp2, xpi2, ax, axbar, bx, bxbar, yp, ypi, yp2, ypi2, ay;
    double aybar, by, bybar, zp, zpi, zp2, zpi2, az, azbar, bz, bzbar, axp;
    double axbarp, bxp, bxbarp, ayp, aybarp, byp, bybarp, azp, azbarp, bzp;
    double bzbarp;

    double sum = 0;
    for (int v = 0; v < ivec; v++) {
        i = ii[v];
        j = jj[v];
        k = kk[v];

        //x
        xp = xparam[v];
        xpi = 1.0 - xp;
        xp2 = xp * xp;
        xpi2 = xpi * xpi;
        ax = xp2 * (3.0 - 2.0 * xp);
        axbar = 1.0 - ax;
        bx = -xp2 * xpi;
        bxbar = xpi2 * xp;

        //y
        yp = yparam[v];
        ypi = 1.0 - yp;
        yp2 = yp * yp;
        ypi2 = ypi * ypi;
        ay = yp2 * (3.0 - 2.0 * yp);
        aybar = 1.0 - ay;
        by = -yp2 * ypi;
        bybar = ypi2 * yp;

        //z
        zp = zparam[v];
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
            iadr = iadr + 1;
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

            sum = sum + hx[v] * (
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

            sum = sum + hx[v] * hy * (
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

            sum = sum + hx[v] * hz * (
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

            sum = sum + hx[v] * hy * hz * (
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

            sum = hxi[v] * (
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

            sum = sum + hxi[v] * hy * (
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

            sum = sum + hxi[v] * hz * (
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

            sum = sum + hxi[v] * hy * hz * (
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
            sum = sum + hyi * hx[v] * (
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
            sum = sum + hx[v] * (
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
            sum = sum + hx[v] * hyi * hz * (
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
            sum = sum + hx[v] * hz * (
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
            sum = sum + hzi * hx[v] * (
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
            sum = sum + hzi * hx[v] * hy * (
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
            sum = sum + hx[v] * (
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
            sum = sum + hx[v] * hy * (
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
}

