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
void evaluateSpline(int ivec, int ivecd, double *fval, int i, int j, int k, double xp,
                    double yp, double zp, double hx, double hxi, double hy, double hyi,
                    double hz, double hzi, double *fin, int inf2, int inf3, int nz) {

    double sum = 0;
    int ict[8] = {1, 1, 1, 1, 0, 0, 0, 0};
    int s1 = 8;


    //Currently allocation running serially on device

    //don't know if it would be better to allocate on host and copy or allocate directly on device
    double* constants_d, sum_d;
    const int chunkSize = 8
    dim3 grid_size = {4};
    dim3 block_size = {8};
    const int N = 256;

    cudaMalloc(&constants_d, N*sizeof(constants));
    cudaMalloc(&sum_d, grid_size.x*block_size.x*sizeof(sum_d));
    allocateArray<<<1,1>>>(xp, yp, zp, hx, hxi, hy, hyi, hz, hzi, constants_d, N);

    summationLookupKernel<<<grid_size,block_size>>(fval, i, j, k, fin, inf2, inf3, nz, constants_d, chunkSize, sum_d);
}

//check if done correctly
#ifdef __NVCC__
__global__
#endif
void summationLookupKernel(double *fval, int i, int j, int k, double *fin, int inf2, int inf3, int nz, double * constants, int chunkSize, double* sum){
    int tid = blockId.x * blockDim.x + threadId.x;
    int i = 0;

    sum[tid] = constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i, j, k, s1, inf2, inf3, nz) +
    constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i, j + 1, k, s1, inf2, inf3, nz)) +
    constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i + 1, j, k, s1, inf2, inf3, nz) +
    constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i + 1, j + 1, k, s1, inf2, inf3, nz))) +
    constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i, j, k + 1, s1, inf2, inf3, nz) +
    constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i, j + 1, k + 1, s1, inf2, inf3, nz)) +
    constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i + 1, j, k + 1, s1, inf2, inf3, nz) +
    constants[tid * chunkSize + i++] * transposeLookup4D(fin, threadId.x, i + 1, j + 1, k + 1, s1, inf2, inf3, nz)));

    //
    if(threadId.x == 0){
        fval[blockId.x] = 0;
        for(int i = 0; i< blockDim.x; i++){
            fval[blockId.x] += sum[blockId.x*blockDim.x + i];
        }
    }
}

#ifdef __NVCC__
__global__
#endif
void allocateArray(double xp, double yp, double zp, double hx, double hxi,double hy, double hyi, double hz,
        double hzi, double * constants, const int size){
    int iadr;
    double xpi, xp2, xpi2, ax, axbar, bx, bxbar, ypi, yp2, ypi2, ay;
    double aybar, by, bybar, zpi, zp2, zpi2, az, azbar, bz, bzbar, axp;
    double axbarp, bxp, bxbarp, ayp, aybarp, byp, bybarp, azp, azbarp, bzp;
    double bzbarp;

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

    int i = 0;

    //---------
    //BLOCK ONE
    //---------

    //thread one
    constants[i++] = azbar * axbar * aybar;
    constants[i++] = azbar * axbar * ay;
    constants[i++] = azbar * ax * aybar;
    constants[i++] = azbar * ax * ay;

    constants[i++] = az * axbar * aybar;
    constants[i++] = az * axbar * ay;
    constants[i++] = az * ax * aybar;
    constants[i++] = az * ax * ay;

    //thread two
    constants[i++] = hx * azbar * bxbar * aybar;
    constants[i++] = hx * azbar * bxbar * ay;
    constants[i++] = hx * azbar * bx * aybar;
    constants[i++] = hx * azbar * bx * ay;

    constants[i++] = hx * az * bxbar * aybar;
    constants[i++] = hx * az * bxbar * ay;
    constants[i++] = hx * az * bx * aybar;
    constants[i++] = hx * az * bx * ay;

    //thread three
    constants[i++] = hy * azbar * axbar * bybar;
    constants[i++] = hy * azbar * axbar * by;
    constants[i++] = hy * azbar * ax * bybar;
    constants[i++] = hy * azbar * ax * by;

    constants[i++] = hy * az * axbar * bybar;
    constants[i++] = hy * az * axbar * by;
    constants[i++] = hy * az * ax * bybar;
    constants[i++] = hy * az * ax * by;

    //thread four
    constants[i++] = hz * bzbar * axbar * aybar;
    constants[i++] = hz * bzbar * axbar * ay;
    constants[i++] = hz * bzbar * ax * aybar;
    constants[i++] = hz * bzbar * ax * ay;

    constants[i++] = hz * bz * axbar * aybar;
    constants[i++] = hz * bz * axbar * ay;
    constants[i++] = hz * bz * ax * aybar;
    constants[i++] = hz * bz * ax * ay;

    //thread five
    constants[i++] = hx * hy * azbar * bxbar * bybar;
    constants[i++] = hx * hy * azbar * bxbar * by;
    constants[i++] = hx * hy * azbar * bx * bybar;
    constants[i++] = hx * hy * azbar * bx * by;

    constants[i++] = hx * hy * az * bxbar * bybar;
    constants[i++] = hx * hy * az * bxbar * by;
    constants[i++] = hx * hy * az * bx * bybar;
    constants[i++] = hx * hy * az * bx * by;

    //thread six
    constants[i++] = hx * hz * bzbar * bxbar * aybar;
    constants[i++] = hx * hz * bzbar * bxbar * ay;
    constants[i++] = hx * hz * bzbar * bx * aybar;
    constants[i++] = hx * hz * bzbar * bx * ay;

    constants[i++] = hx * hz * bz * bxbar * aybar;
    constants[i++] = hx * hz * bz * bxbar * ay;
    constants[i++] = hx * hz * bz * bx * aybar;
    constants[i++] = hx * hz * bz * bx * ay;

    //thread seven
    constants[i++] = hy * hz * bzbar * axbar * bybar;
    constants[i++] = hy * hz * bzbar * axbar * by;
    constants[i++] = hy * hz * bzbar * ax * bybar;
    constants[i++] = hy * hz * bzbar * ax * by;

    constants[i++] = hy * hz * bz * axbar * bybar;
    constants[i++] = hy * hz * bz * axbar * by;
    constants[i++] = hy * hz * bz * ax * bybar;
    constants[i++] = hy * hz * bz * ax * by;

    //thread eight
    constants[i++] = hx * hy * hz * bzbar * bxbar * bybar;
    constants[i++] = hx * hy * hz * bzbar * bxbar * by;
    constants[i++] = hx * hy * hz * bzbar * bx * bybar;
    constants[i++] = hx * hy * hz * bzbar * bx * by;

    constants[i++] = hx * hy * hz * bz * bxbar * bybar;
    constants[i++] = hx * hy * hz * bz * bxbar * by;
    constants[i++] = hx * hy * hz * bz * bx * bybar;
    constants[i++] = hx * hy * hz * bz * bx * by;

    //---------
    //BLOCK TWO
    //---------

    //thread one
    constants[i++] = hxi * azbar * axbarp * aybar;
    constants[i++] = hxi * azbar * axbarp * ay;
    constants[i++] = hxi * azbar * axp * aybar;
    constants[i++] = hxi * azbar * axp * ay;

    constants[i++] = hxi * az * axbarp * aybar;
    constants[i++] = hxi * az * axbarp * ay;
    constants[i++] = hxi * az * axp * aybar;
    constants[i++] = hxi * az * axp * ay;

    //thread two
    constants[i++] = azbar * bxbarp * aybar;
    constants[i++] = azbar * bxbarp * ay;
    constants[i++] = azbar * bxp * aybar;
    constants[i++] = azbar * bxp * ay;

    constants[i++] = az * bxbarp * aybar;
    constants[i++] = az * bxbarp * ay;
    constants[i++] = az * bxp * aybar;
    constants[i++] = az * bxp * ay;

    //thread three
    constants[i++] = hxi * hy * azbar * axbarp * bybar;
    constants[i++] = hxi * hy * azbar * axbarp * by;
    constants[i++] = hxi * hy * azbar * axp * bybar;
    constants[i++] = hxi * hy * azbar * axp * by;

    constants[i++] = hxi * hy * az * axbarp * bybar;
    constants[i++] = hxi * hy * az * axbarp * by;
    constants[i++] = hxi * hy * az * axp * bybar;
    constants[i++] = hxi * hy * az * axp * by;

    //thread four
    constants[i++] = hxi * hz * bzbar * axbarp * aybar;
    constants[i++] = hxi * hz * bzbar * axbarp * ay;
    constants[i++] = hxi * hz * bzbar * axp * aybar;
    constants[i++] = hxi * hz * bzbar * axp * ay;

    constants[i++] = hxi * hz * bz * axbarp * ay;
    constants[i++] = hxi * hz * bz * axbarp * aybar;
    constants[i++] = hxi * hz * bz * axp * aybar;
    constants[i++] = hxi * hz * bz * axp * ay;

    //thread five
    constants[i++] = hy * azbar * bxbarp * bybar;
    constants[i++] = hy * azbar * bxbarp * by;
    constants[i++] = hy * azbar * bxp * bybar;
    constants[i++] = hy * azbar * bxp * by;

    constants[i++] = hy * az * bxbarp * bybar;
    constants[i++] = hy * az * bxbarp * by;
    constants[i++] = hy * az * bxp * bybar;
    constants[i++] = hy * az * bxp * by;

    //thread six
    constants[i++] = hz * bzbar * bxbarp * aybar;
    constants[i++] = hz * bzbar * bxbarp * ay;
    constants[i++] = hz * bzbar * bxp * aybar;
    constants[i++] = hz * bzbar * bxp * ay;

    constants[i++] = hz * bz * bxbarp * aybar;
    constants[i++] = hz * bz * bxbarp * ay;
    constants[i++] = hz * bz * bxp * aybar;
    constants[i++] = hz * bz * bxp * ay;

    //thread seven
    constants[i++] = hxi * hy * hz * bzbar * axbarp * bybar;
    constants[i++] = hxi * hy * hz * bzbar * axbarp * by;
    constants[i++] = hxi * hy * hz * bzbar * axp * bybar;
    constants[i++] = hxi * hy * hz * bzbar * axp * by;

    constants[i++] = hxi * hy * hz * bz * axbarp * bybar;
    constants[i++] = hxi * hy * hz * bz * axbarp * by;
    constants[i++] = hxi * hy * hz * bz * axp * bybar;
    constants[i++] = hxi * hy * hz * bz * axp * by;

    //thread eight
    constants[i++] = hy * hz * bzbar * bxbarp * bybar;
    constants[i++] = hy * hz * bzbar * bxbarp * by;
    constants[i++] = hy * hz * bzbar * bx * bybar;
    constants[i++] = hy * hz * bzbar * bxp * by;

    constants[i++] = hy * hz * bz * bxbarp * bybar;
    constants[i++] = hy * hz * bz * bxbarp * by;
    constants[i++] = hy * hz * bz * bxp * bybar;
    constants[i++] = hy * hz * bz * bxp * by;

    //---------
    //BLOCK THREE
    //---------

    //thread one
    constants[i++] = hyi * azbar * axbar * aybarp;
    constants[i++] = hyi * azbar * axbar * ayp;
    constants[i++] = hyi * azbar * axp * aybarp;
    constants[i++] = hyi * azbar * axp * ayp;

    constants[i++] = hyi * az * axbar * aybarp;
    constants[i++] = hyi * az * axbar * ayp;
    constants[i++] = hyi * az * axp * aybarp;
    constants[i++] = hyi * az * axp * ayp;

    //thread two
    constants[i++] = hyi * hx * azbar * bxbar * aybarp;
    constants[i++] = hyi * hx * azbar * bxbar * ayp;
    constants[i++] = hyi * hx * azbar * bxp * aybarp;
    constants[i++] = hyi * hx * azbar * bxp * ayp;

    constants[i++] = hyi * hx * az * bxbar * aybarp;
    constants[i++] = hyi * hx * az * bxbar * ayp;
    constants[i++] = hyi * hx * az * bxp * aybarp;
    constants[i++] = hyi * hx * az * bxp * ayp;

    //thread three
    constants[i++] = azbar * axbar * bybarp;
    constants[i++] = azbar * axbar * byp;
    constants[i++] = azbar * axp * bybarp;
    constants[i++] = azbar * axp * byp;

    constants[i++] = az * axbar * bybarp;
    constants[i++] = az * axbar * byp;
    constants[i++] = az * axp * bybarp;
    constants[i++] = az * axp * byp;

    //thread four
    constants[i++] = hyi * hz * bzbar * axbar * aybarp;
    constants[i++] = hyi * hz * bzbar * axbar * ayp;
    constants[i++] = hyi * hz * bzbar * axp * aybarp;
    constants[i++] = hyi * hz * bzbar * axp * ayp;

    constants[i++] = hyi * hz * bz * axbar * ayp;
    constants[i++] = hyi * hz * bz * axbar * aybarp;
    constants[i++] = hyi * hz * bz * axp * aybarp;
    constants[i++] = hyi * hz * bz * axp * ayp;

    //thread five
    constants[i++] = hx * azbar * bxbar * bybar;
    constants[i++] = hx * azbar * bxbar * by;
    constants[i++] = hx * azbar * bxp * bybar;
    constants[i++] = hx * azbar * bxp * by;

    constants[i++] = hx * az * bxbar * bybar;
    constants[i++] = hx * az * bxbar * by;
    constants[i++] = hx * az * bxp * bybar;
    constants[i++] = hx * az * bxp * by;

    //thread six
    constants[i++] = hx * hyi * hz * bzbar * bxbar * aybar;
    constants[i++] = hx * hyi * hz * bzbar * bxbar * ay;
    constants[i++] = hx * hyi * hz * bzbar * bxp * aybar;
    constants[i++] = hx * hyi * hz * bzbar * bxp * ay;

    constants[i++] = hx * hyi * hz * bz * bxbar * aybar;
    constants[i++] = hx * hyi * hz * bz * bxbar * ay;
    constants[i++] = hx * hyi * hz * bz * bxp * aybar;
    constants[i++] = hx * hyi * hz * bz * bxp * ay;

    //thread seven
    constants[i++] = hz * bzbar * axbar * bybar;
    constants[i++] = hz * bzbar * axbar * by;
    constants[i++] = hz * bzbar * axp * bybar;
    constants[i++] = hz * bzbar * axp * by;

    constants[i++] = hz * bz * axbar * bybar;
    constants[i++] = hz * bz * axbar * by;
    constants[i++] = hz * bz * axp * bybar;
    constants[i++] = hz * bz * axp * by;

    //thread eight
    constants[i++] = hx * hz * bzbar * bxbar * bybar;
    constants[i++] = hx * hz * bzbar * bxbar * by;
    constants[i++] = hx * hz * bzbar * bx * bybar;
    constants[i++] = hx * hz * bzbar * bxp * by;

    constants[i++] = hx * hz * bz * bxbar * bybar;
    constants[i++] = hx * hz * bz * bxbar * by;
    constants[i++] = hx * hz * bz * bxp * bybar;
    constants[i++] = hx * hz * bz * bxp * by;

    //---------
    //BLOCK FOUR
    //---------

    //thread one
    constants[i++] = hzi * azbarp * axbar * aybar;
    constants[i++] = hzi * azbarp * axbar * ay;
    constants[i++] = hzi * azbarp * ax * aybar;
    constants[i++] = hzi * azbarp * ax * ay;

    constants[i++] = hzi * azp * axbar * aybar;
    constants[i++] = hzi * azp * axbar * ay;
    constants[i++] = hzi * azp * ax * aybar;
    constants[i++] = hzi * azp * ax * ay;

    //thread two
    constants[i++] = hzi * hx * azbarp * bxbar * aybar;
    constants[i++] = hzi * hx * azbarp * bxbar * ay;
    constants[i++] = hzi * hx * azbarp * bx * aybar;
    constants[i++] = hzi * hx * azbarp * bx * ay;

    constants[i++] = hzi * hx * azp * bxbar * aybar;
    constants[i++] = hzi * hx * azp * bxbar * ay;
    constants[i++] = hzi * hx * azp * bx * aybar;
    constants[i++] = hzi * hx * azp * bx * ay;

    //thread three
    constants[i++] = hzi * hy * azbarp * axbar * bybar;
    constants[i++] = hzi * hy * azbarp * axbar * by;
    constants[i++] = hzi * hy * azbarp * ax * bybar;
    constants[i++] = hzi * hy * azbarp * ax * by;

    constants[i++] = hzi * hy * azp * axbar * bybar;
    constants[i++] = hzi * hy * azp * axbar * by;
    constants[i++] = hzi * hy * azp * ax * bybar;
    constants[i++] = hzi * hy * azp * ax * by;

    //thread four
    constants[i++] = bzbarp * axbar * aybar;
    constants[i++] = bzbarp * axbar * ay;
    constants[i++] = bzbarp * ax * aybar;
    constants[i++] = bzbarp * ax * ay;

    constants[i++] = bzp * axbar * aybar;
    constants[i++] = bzp * axbar * ay;
    constants[i++] = bzp * ax * aybar;
    constants[i++] = bzp * ax * ay;

    //thread five
    constants[i++] = hzi * hx * hy * azbarp * bxbar * bybar;
    constants[i++] = hzi * hx * hy * azbarp * bxbar * by;
    constants[i++] = hzi * hx * hy * azbarp * bx * bybar;
    constants[i++] = hzi * hx * hy * azbarp * bx * by;

    constants[i++] = hzi * hx * hy * azp * bxbar * bybar;
    constants[i++] = hzi * hx * hy * azp * bxbar * by;
    constants[i++] = hzi * hx * hy * azp * bx * bybar;
    constants[i++] = hzi * hx * hy * azp * bx * by;

    //thread six
    constants[i++] = hx * bzbarp * bxbar * aybar;
    constants[i++] = hx * bzbarp * bxbar * ay;
    constants[i++] = hx * bzbarp * bx * aybar;
    constants[i++] = hx * bzbarp * bx * ay;

    constants[i++] = hx * bzp * bxbar * aybar;
    constants[i++] = hx * bzp * bxbar * ay;
    constants[i++] = hx * bzp * bx * aybar;
    constants[i++] = hx * bzp * bx * ay;

    //thread seven
    constants[i++] = hy * bzbarp * axbar * bybar;
    constants[i++] = hy * bzbarp * axbar * by;
    constants[i++] = hy * bzbarp * ax * bybar;
    constants[i++] = hy * bzbarp * ax * by;

    constants[i++] = hy * bzp * axbar * bybar;
    constants[i++] = hy * bzp * axbar * by;
    constants[i++] = hy * bzp * ax * bybar;
    constants[i++] = hy * bzp * ax * by;

    //thread eight
    constants[i++] = hx * hy * bzbarp * bxbar * bybar;
    constants[i++] = hx * hy * bzbarp * bxbar * by;
    constants[i++] = hx * hy * bzbarp * bx * bybar;
    constants[i++] = hx * hy * bzbarp * bx * by;

    constants[i++] = hx * hy * bzp * bxbar * bybar;
    constants[i++] = hx * hy * bzp * bxbar * by;
    constants[i++] = hx * hy * bzp * bx * bybar;
    constants[i++] = hx * hy * bzp * bx * by;
}
