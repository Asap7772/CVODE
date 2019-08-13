//
// Created by Anikait Singh on 2019-08-06.
// realtype
//
#include "fblin.h"
#include "jacobian.h"
#include "sundials_nvector.h"
#include "sundials_types.h"
#include "nvector_serial.h"
#include "cvode.h"
#include "sunlinsol_spgmr.h"
#include <cuda_runtime.h>
#include <nvector/nvector_cuda.h>
#include <sundials/sundials_math.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
// typedef cudaStream_t cudaStream_t;


#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct _UserData {
    //constants
    int neq, nr, nphi, nz;
    double rmin, rmax, phimin, phimax, zmin, zmax, eps1, eps2, eps3, delta_phi;
    //static arrays
    double *raxis, *phiaxis, *zaxis, *BR4D, *BZ4D;
    double *raxis_d, *phiaxis_d, *zaxis_d, *BR4D_d, *BZ4D_d;
};

typedef struct _UserData *UserData;
//typedef double realtype;
//typedef double *N_Vector;

/* Problem setup and initialization functions */
static UserData SetUserData(int neq, double rmin, double rmax, double phimin, double phimax, double zmin, double zmax,
                            int nr, int nphi, int nz, double eps1, double eps2, double eps3, double *raxis,
                            double *phiaxis, double *zaxis, double *BR4D, double *BZ4D, double *raxis_d,
                            double *phiaxis_d, double *zaxis_d, double *BR4D_d, double *BZ4D_d, double delta_phi) {

    UserData ud = (UserData) malloc(sizeof(*ud));
//    if (check_retval((void *) ud, "AllocUserData", 2)) return (NULL);
    ud->neq = neq;
    ud->rmin = rmin;
    ud->rmax = rmax;
    ud->phimin = phimin;
    ud->phimax = phimax;
    ud->zmin = zmin;
    ud->zmax = zmax;
    ud->nr = nr;
    ud->nphi = nphi;
    ud->nz = nz;
    ud->eps1 = eps1;
    ud->eps2 = eps2;
    ud->eps3 = eps3;
    ud->raxis = raxis;
    ud->raxis_d = raxis_d;
    ud->zaxis = zaxis;
    ud->zaxis_d = zaxis_d;
    ud->phiaxis = phiaxis;
    ud->phiaxis_d = phiaxis_d;
    ud->BR4D = BR4D;
    ud->BR4D_d = BR4D_d;
    ud->BZ4D = BZ4D;
    ud->BZ4D_d = BZ4D_d;
    ud->delta_phi = delta_phi;
    return ud;
}

/* Functions Called by the Solver */
static int f(realtype t, N_Vector u, N_Vector udot, void *user_data) {
    UserData data = (UserData) user_data;
    //NVECTOR SYNTAX
    //  NON-CUDA
//    double *u_data = NV_DATA_S(u);
//    double *udot_data = NV_DATA_S(udot);
    //  CUDA
    double *u_data_d = N_VGetDeviceArrayPointer_Cuda(u);
    double *u_data = N_VGetHostArrayPointer_Cuda(u);
    double *udot_data_d = N_VGetDeviceArrayPointer_Cuda(udot);
    double *udot_data = N_VGetHostArrayPointer_Cuda(udot);

   rhside_lsode_kernel(t, u_data,u_data_d, udot_data, udot_data_d, data->rmin, data->rmax, data->phimin, data->phimax, data->zmin,
                        data->zmax, data->nr, data->nphi, data->nz, data->eps1, data->eps2, data->eps3, data->raxis,
                        data->phiaxis, data->zaxis, data->BR4D, data->BZ4D, data->raxis_d, data->phiaxis_d, data->zaxis_d,
			data->BR4D_d, data->BZ4D_d, data->delta_phi);
    return 0;
}

static int jtv(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *user_data, N_Vector temp) {
    UserData data = (UserData) user_data;
    //NVECTOR SYNTAX
    //  NON-CUDA
//    double *u_data = NV_DATA_S(u);
//    double *udot_data = NV_DATA_S(fu);
//    double *vec_data = NV_DATA_S(v);
//    double *out_data = NV_DATA_S(Jv);
    //  CUDA
    double *u_data = N_VGetHostArrayPointer_Cuda(u);
    double *u_data_d = N_VGetDeviceArrayPointer_Cuda(u);
    double *udot_data = N_VGetHostArrayPointer_Cuda(fu);
    double *vec_data = N_VGetHostArrayPointer_Cuda(v);
    double *out_data = N_VGetHostArrayPointer_Cuda(Jv);

    int nrpd = data->neq * data->neq;
    double pd[nrpd];

    jacobian_lsode_kernelC(data->neq, t, u_data, u_data_d, pd, nrpd, data->rmin, data->rmax, data->phimin, data->phimax,
                           data->zmin, data->zmax, data->nr, data->nphi, data->nz, data->eps1, data->eps2, data->eps3,
                           data->raxis, data->phiaxis, data->zaxis, data->BR4D, data->BZ4D,data->raxis_d, data->phiaxis_d,
			    data->zaxis_d, data->BR4D_d, data->BZ4D_d, data->delta_phi);

    // 2 x 2 matrix times 2 x 1 vector
    out_data[0] = pd[0] * vec_data[0] + pd[2] * vec_data[1];
    out_data[0] = pd[1] * vec_data[0] + pd[3] * vec_data[1];
    return 0;
}

//method to call from fortran
extern "C"
void evaluatecvode_(int *neq_pointer, double *uval, double *t_pointer, double *tout_pointer, double *reltol_pointer,
                    double *abstol_pointer, double *rmin_pointer, double *rmax_pointer, double *phimin_pointer,
                    double *phimax_pointer, double *zmin_pointer, double *zmax_pointer, int *nr_pointer,
                    int *nphi_pointer, int *nz_pointer, double *eps1_pointer, double *eps2_pointer,
                    double *eps3_pointer, double *raxis, double *phiaxis, double *zaxis, double *BR4D, double *BZ4D,
                    double *delta_phi_pointer) {

    int neq = *neq_pointer;
    double t = *t_pointer;
    double tout = *tout_pointer;
    double reltol = *reltol_pointer;
    double abstol = *abstol_pointer;
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

    N_Vector u;
    int iout, retval;
    void *cvode_mem;
    SUNLinearSolver LS;
    UserData data;


    //    ifaccess("a.dat" , F_OK ) == -1 ) {
    //FILE *fp = fopen("a.dat","w+");
    //fprintf(fp, "neq = %d;", neq);
    //fprintf(fp, "neq = %d;", neq);
    //fclose(fp);
    //}
    // fprintf(stderr,"%f %f\n", t, tout);

double *raxis_d, *phiaxis_d, *zaxis_d, *BR4D_d, *BZ4D_d;
 fprintf(stderr, "Before malloc cuda\n");
 fprintf(stderr, "nr = %d, nphi =%d, nz = %d\n", nr, nphi, nz);
//malloc cuda
 fprintf(stderr, "one\n");
 HANDLE_ERROR(cudaMalloc(&raxis_d,(size_t)(nr*sizeof(double))));
 fprintf(stderr, "two\n");
 HANDLE_ERROR(cudaMalloc(&phiaxis_d,(size_t)(nphi*sizeof(double))));
 fprintf(stderr, "three\n");
 HANDLE_ERROR(cudaMalloc(&zaxis_d,(size_t)(nz*sizeof(double))));
 fprintf(stderr, "four\n");
 HANDLE_ERROR(cudaMalloc(&BR4D_d,(size_t)(nr*nphi*nz*8*sizeof(double))));
 fprintf(stderr, "five\n");
 HANDLE_ERROR(cudaMalloc(&BZ4D_d,(size_t)(nr*nphi*nz*8*sizeof(double))));
 fprintf(stderr, "After malloc cuda\n");
//memcpy cuda
 HANDLE_ERROR(cudaMemcpy(raxis_d, raxis, (size_t)(nr * sizeof(double)), cudaMemcpyHostToDevice));
 HANDLE_ERROR(cudaMemcpy(phiaxis_d, phiaxis, (size_t)(nphi * sizeof(double)), cudaMemcpyHostToDevice));
 HANDLE_ERROR(cudaMemcpy(zaxis_d, zaxis, (size_t)(nz * sizeof(double)), cudaMemcpyHostToDevice));
 HANDLE_ERROR(cudaMemcpy(BR4D_d, BR4D, (size_t)(8*nr*nphi*nz * sizeof(double)), cudaMemcpyHostToDevice));
 HANDLE_ERROR(cudaMemcpy(BZ4D_d, BZ4D, (size_t)(8*nr*nphi*nz * sizeof(double)), cudaMemcpyHostToDevice));
 fprintf(stderr, "After memcpy cuda\n");
 data = SetUserData(neq, rmin, rmax, phimin, phimax, zmin, zmax, nr, nphi, nz, eps1, eps2, eps3,raxis, phiaxis, zaxis, BR4D, BZ4D,
 		   raxis_d, phiaxis_d,zaxis_d, BR4D_d, BZ4D_d, delta_phi);
 fprintf(stderr, "After SetUserData\n");
    u = N_VMake_Serial(data->neq, uval);
 double *u_d;
 HANDLE_ERROR(cudaMalloc(&u_d, (size_t)(neq * sizeof(double))));
 fprintf(stderr, "before Nvect stuff uval %p u_d %p\n",uval, u_d);
 u = N_VMake_Cuda(neq, uval, u_d);
 fprintf(stderr,"before cvode uval %p u_d %p\n",uval, u_d);
 HANDLE_ERROR(cudaMemcpy(u_d, uval, (size_t)(neq * sizeof(double)), cudaMemcpyHostToDevice));

 cvode_mem = CVodeCreate(CV_BDF);
 retval = CVodeInit(cvode_mem, f, t, u);
 retval = CVodeSStolerances(cvode_mem, reltol, abstol);
 retval = CVodeSetUserData(cvode_mem, data);
 LS = SUNLinSol_SPGMR(u, PREC_NONE, 0);
 retval = CVodeSetLinearSolver(cvode_mem, LS, NULL);
 retval = CVodeSetJacTimes(cvode_mem, NULL, jtv);

 retval = CVode(cvode_mem, tout, u, &t, CV_NORMAL);
 retval = CVode(cvode_mem, t + delta_phi, u, &tout, CV_NORMAL);
 //double* x = NV_DATA_S(u);
 u_d = N_VGetDeviceArrayPointer_Cuda(u);
 fprintf(stderr,"after cvode  uval %p u_d %p\n",uval, u_d);
 //N_VCopyFromDevice_Cuda(u);
 HANDLE_ERROR(cudaMemcpy(uval, u_d, (size_t)(neq * sizeof(double)), cudaMemcpyDeviceToHost));
 //    for(int i = 0; i<neq; i++){
 //        uval[i] = x[i];
 //    }
 //    retval = CVodeGetNumSteps(cvode_mem, &nst);
 fprintf(stderr, "leaving cvodeTest\n");
 cudaFree(raxis_d);
 cudaFree(phiaxis_d);
 cudaFree(zaxis_d);
 cudaFree(BR4D_d);
 cudaFree(BZ4D_d);
 cudaFree(u_d);
 free(data);
}
