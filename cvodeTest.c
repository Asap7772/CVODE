//
// Created by Anikait Singh on 2019-08-06.
// realtype
//
#include "fblin.h"
#include "jacobian.h"
#include "sundials_nvector.h"
#include "sundials_types.h"
#include "nvector_serial.h"
#include "nvector_cuda.h"
#include "cvode.h"
#include "sunlinsol_spgmr.h"
#include <stdlib.h>

struct _UserData {
    //constants
    int neq, nr, nphi, nz;
    double rmin, rmax, phimin, phimax, zmin, zmax, eps1, eps2, eps3, delta_phi;
    //static arrays
    double *raxis, *phiaxis, *zaxis, *BR4D, *BZ4D;
};

typedef struct _UserData *UserData;
//typedef double realtype;
//typedef double *N_Vector;

/* Problem setup and initialization functions */
static UserData SetUserData(int neq, double rmin, double rmax, double phimin, double phimax, double zmin, double zmax,
                            int nr, int nphi, int nz, double eps1, double eps2, double eps3, double *raxis,
                            double *phiaxis, double *zaxis, double *BR4D, double *BZ4D, double delta_phi) {

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
    ud->zaxis = zaxis;
    ud->phiaxis = phiaxis;
    ud->BR4D = BR4D;
    ud->BZ4D = BZ4D;
    ud->delta_phi = delta_phi;
    return ud;
}

/* Functions Called by the Solver */
static int f(realtype t, N_Vector u, N_Vector udot, void *user_data) {
    UserData data = (UserData) user_data;
    //NVECTOR SYNTAX
    //  NON-CUDA
    //    double *u_data = u->data;
    //    double *udot_data = udot->data;
    //  CUDA
    //    double *u_data = N_VGetHostArrayPointer_Cuda(u);
    //    double *udot_data = N_VGetHostArrayPointer_Cuda(udot);

    //alias for now
    double *u_data = u;
    double *udot_data = udot;

    rhside_lsode_kernel(t, u_data, udot_data, data->rmin, data->rmax, data->phimin, data->phimax, data->zmin,
                        data->zmax, data->nr, data->nphi, data->nz, data->eps1, data->eps2, data->eps3, data->raxis,
                        data->zaxis, data->phiaxis, data->BR4D, data->BZ4D, data->delta_phi);
    return 0;
}

static int jtv(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *user_data) {
    UserData data = (UserData) user_data;
    //NVECTOR SYNTAX
    //  NON-CUDA
    //    double *u_data = u->data;
    //    double *udot_data = fu->data;
    //    double *vec_data = v->data;
    //    double *out_data = Jv->data;
    //  CUDA
    //    double *u_data = N_VGetHostArrayPointer_Cuda(u);
    //    double *udot_data = N_VGetHostArrayPointer_Cuda(fu);
    //    double *vec_data = N_VGetHostArrayPointer_Cuda(v);
    //    double *out_data = N_VGetHostArrayPointer_Cuda(Jv);

    //alias for now
    double *u_data = u;
    double *udot_data = fu;
    double *vec_data = v;
    double *out_data = Jv;

    int nrpd = data->neq * data->neq;
    double pd[nrpd];

    jacobian_lsode_kernelC(data->neq,t, u_data, pd, nrpd, data->rmin, data->rmax, data->phimin, data->phimax, data->zmin,
                           data->zmax, data->nr, data->nphi,data->nz, data->eps1,data->eps2, data->eps3, data->raxis, data->phiaxis,
                           data->zaxis, data->BR4D, data->BZ4D, data -> delta_phi);

    // 2 x 2 matrix times 2 x 1 vector
    out_data[0] = pd[0] * v[0] + pd[2] * v[1];
    out_data[0] = pd[1] * v[0] + pd[3] * v[1];
    return 0;
}

//void jacobian_lsode_kernelC(int neq, double phi, double *q, double *pd, int nrpd, double rmin, double rmax, double phimin, double phimax,
//                            double zmin, double zmax, int nr, int nphi, int nz, double eps1, double eps2, double eps3,
//                            double *raxis, double *phiaxis, double *zaxis, double *BR4D, double *BZ4D,
//                            double delta_phi);

//method to call from fortran
void evaluateCvode_(int *neq_pointer, double *u, double *t_pointer, double *tout_pointer, double *reltol_pointer,
                    double *abstol_pointer, double *rmin_pointer, double *rmax_pointer, double *phimin_pointer,
                    double *phimax_pointer, double *zmin_pointer, double *zmax_pointer, int *nr_pointer,
                    int *nphi_pointer, int *nz_pointer, double *eps1_pointer, double *eps2_pointer,
                    double *eps3_pointer, double *raxis, double *phiaxis, double *zaxis, double *BR4D, double *BZ4D,
                    double *delta_phi_pointer) {

    int neq = *neq_pointer;
    int t = *t_pointer;
    int tout = *tout_pointer;
    int reltol = *reltol_pointer;
    int abstol = *abstol_pointer;
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

    N_Vector uvec;
    int iout, retval;
    void *cvode_mem;
    SUNLinearSolver LS;
    UserData data;

    data = SetUserData(neq, rmin, rmax, phimin, phimax, zmin, zmax, nr, nphi, nz, eps1, eps2, eps3, raxis, phiaxis, zaxis, BR4D, BZ4D, delta_phi);

    uvec = N_VMake_Serial(data->NEQ, u);
    cvode_mem = CVodeCreate(CV_BDF);
    retval = CVodeInit(cvode_mem, f, t, u);
    retval = CVodeSStolerances(cvode_mem, reltol, abstol);
    retval = CVodeSetUserData(cvode_mem, data);
    LS = SUNLinSol_SPGMR(u, PREC_NONE, 0);
    retval = CVodeSetLinearSolver(cvode_mem, LS, NULL);
    retval = CVodeSetJacTimes(cvode_mem, NULL, jtv);

    retval = CVode(cvode_mem, tout, u, &t, CV_NORMAL);
//    retval = CVodeGetNumSteps(cvode_mem, &nst);
    free(data)
}
