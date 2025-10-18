#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* Utility: allocate matrix (rows x cols) */
static double *mat_alloc(int rows, int cols) {
    double *m = (double*)calloc((size_t)rows * cols, sizeof(double));
    if (!m) { perror("calloc"); exit(1); }
    return m;
}

/* C = A * B
 * A: r x m, B: m x c => C: r x c
 */
static void mat_mul(const double *A, int r, int m, const double *B, int c, double *C) {
    for (int i=0;i<r*c;i++) C[i]=0.0;
    for (int i=0;i<r;i++){
        for (int k=0;k<m;k++){
            double av = A[i*m + k];
            for (int j=0;j<c;j++){
                C[i*c + j] += av * B[k*c + j];
            }
        }
    }
}

/* out = A * v  (A: r x c, v: c x 1) => out: r x 1 */
static void mat_vec_mul(const double *A, int r, int c, const double *v, double *out) {
    for (int i=0;i<r;i++){
        double s = 0.0;
        for (int j=0;j<c;j++) s += A[i*c + j] * v[j];
        out[i] = s;
    }
}

/* infinity-norm (max absolute element) */
static double vec_inf_norm(const double *v, int n) {
    double mx = 0.0;
    for (int i=0;i<n;i++){
        double a = fabs(v[i]);
        if (a > mx) mx = a;
    }
    return mx;
}

/* Main FalCAN attack-vector synthesis routine */
void compute_attack_vector(
    int n, int m, int p,
    const double *A, const double *B, const double *C, const double *K, const double *L,
    const double *B_inv, const double *C_inv,
    const double *x_tilde_k,
    const double *u_hat,
    const double *y_a_k,
    double theta_u_val,
    double theta_u_grad,
    double theta_y_val,
    double theta_y_grad,
    double theta_res,
    double *a_out,
    double *u_tilde_out
) {
    if (p != 1) {
        fprintf(stderr, "This implementation currently requires a single control input (p=1).\n");
        exit(1);
    }

    /* 1) ESTIMATE x_tilde_{k+1} using Eq.5 */
    double *Ax = mat_alloc(n,1);
    mat_vec_mul(A, n, n, x_tilde_k, Ax);

    double *Bu = mat_alloc(n,1);
    mat_vec_mul(B, n, p, u_hat, Bu);

    double *Axb = mat_alloc(n,1);
    for (int i=0;i<n;i++) Axb[i] = Ax[i] + Bu[i];

    double *C_Axb = mat_alloc(m,1);
    mat_vec_mul(C, m, n, Axb, C_Axb);

    double *resid = mat_alloc(m,1);
    for (int i=0;i<m;i++) resid[i] = y_a_k[i] - C_Axb[i];

    double *L_resid = mat_alloc(n,1);
    mat_vec_mul(L, n, m, resid, L_resid);

    double *x_tilde_k1 = mat_alloc(n,1);
    for (int i=0;i<n;i++) x_tilde_k1[i] = Axb[i] + L_resid[i];

    /* 2) Compute -K * x_tilde_{k+1} (nominal control) */
    double *Kx = mat_alloc(p,1);
    mat_vec_mul(K, p, n, x_tilde_k1, Kx);

    double *u_nom = mat_alloc(p,1);
    for (int i=0;i<p;i++) u_nom[i] = -Kx[i];

    /* 3) Compute UB1..UB5 */
    double *CA = mat_alloc(m, n); mat_mul(C, m, n, A, n, CA);
    double *LCA = mat_alloc(n, n); mat_mul(L, n, m, CA, n, LCA);
    double *A_minus_LCA = mat_alloc(n, n);
    for (int i=0;i<n*n;i++) A_minus_LCA[i] = A[i] - LCA[i];

    double *CB = mat_alloc(m, p); mat_mul(C, m, n, B, p, CB);
    double *LCB = mat_alloc(n, p); mat_mul(L, n, m, CB, p, LCB);
    double *B_minus_LCB = mat_alloc(n, p);
    for (int i=0;i<n*p;i++) B_minus_LCB[i] = B[i] - LCB[i];

    double *term1 = mat_alloc(n,1); mat_vec_mul(A_minus_LCA, n, n, x_tilde_k, term1);
    double *term2 = mat_alloc(n,1); mat_vec_mul(B_minus_LCB, n, p, u_hat, term2);
    double *Lya = mat_alloc(n,1); mat_vec_mul(L, n, m, y_a_k, Lya);
    double *inner = mat_alloc(n,1);
    for (int i=0;i<n;i++) inner[i] = term1[i] + term2[i] + Lya[i];

    double inner_K_scalar = 0.0;
    for (int j=0;j<n;j++) inner_K_scalar += K[j] * inner[j];

    double UB1 = theta_u_val + fabs(inner_K_scalar);
    double UB2 = theta_u_grad + fabs(inner_K_scalar + u_hat[0]);

    double *BK = mat_alloc(n, n); mat_mul(B, n, p, K, n, BK);
    double *A_minus_BK = mat_alloc(n, n);
    for (int i=0;i<n*n;i++) A_minus_BK[i] = A[i] - BK[i];

    double *tmp_n = mat_alloc(n,1); mat_vec_mul(A_minus_BK, n, n, x_tilde_k1, tmp_n);
    double *tmp_m = mat_alloc(m,1); mat_vec_mul(C, m, n, tmp_n, tmp_m);
    double tmp_m_norm = vec_inf_norm(tmp_m, m);

    double Binv_norm = vec_inf_norm(B_inv, p*n);
    double Cinv_norm = vec_inf_norm(C_inv, n*m);
    double BCinv_norm = Binv_norm * Cinv_norm;

    double rhs3 = theta_y_val - tmp_m_norm;
    double UB3 = (rhs3 > 0.0) ? BCinv_norm * rhs3 : 0.0;

    double *diff_m = mat_alloc(m,1);
    for (int i=0;i<m;i++) diff_m[i] = tmp_m[i] - y_a_k[i];
    double diff_norm = vec_inf_norm(diff_m, m);
    double rhs4 = theta_y_grad - diff_norm;
    double UB4 = (rhs4 > 0.0) ? BCinv_norm * rhs4 : 0.0;

    double UB5 = BCinv_norm * theta_res;

    /* 4) Final attack magnitude (Eq. 6) */
    double best = UB1;
    if (UB2 < best) best = UB2;
    if (UB3 < best) best = UB3;
    if (UB4 < best) best = UB4;
    if (UB5 < best) best = UB5;

    double sign = (u_nom[0] >= 0.0) ? 1.0 : -1.0;
    double a_k1 = sign * best;
    double u_tilde_k1 = u_nom[0] + a_k1;

    fprintf(stderr, "DEBUG: UB1=%g UB2=%g UB3=%g UB4=%g UB5=%g => best=%g\n",
            UB1, UB2, UB3, UB4, UB5, best);
    fprintf(stderr, "DEBUG: u_nom = %g, chosen a_k1 = %g, u_tilde_k1 = %g\n",
            u_nom[0], a_k1, u_tilde_k1);

    a_out[0] = a_k1;
    u_tilde_out[0] = u_tilde_k1;

    free(Ax); free(Bu); free(Axb); free(C_Axb); free(resid); free(L_resid);
    free(x_tilde_k1); free(Kx); free(u_nom);
    free(CA); free(LCA); free(A_minus_LCA);
    free(CB); free(LCB); free(B_minus_LCB);
    free(term1); free(term2); free(Lya); free(inner);
    free(BK); free(A_minus_BK); free(tmp_n); free(tmp_m);
    free(diff_m);
}

/* Example main using a 2-state system model */
int main() {
    // System dimensions for TTC/ESP: n=2 states, m=1 measurement, p=1 control input
    int n=2, m=1, p=1;

    // --- Static System Parameters ---
    // A (n x n): State transition matrix
    double A[] = {1.0, 0.1, 0.0, 1.0};
    // B (n x p): Control input matrix
    double B[] = {0.0, 0.5};
    // C (m x n): Measurement matrix
    double C[] = {1.0, 0.0};
    // K (p x n): Controller gain matrix
    double K[] = {0.5, 0.2};
    // L (n x m): Estimator gain matrix
    double L[] = {0.2, 0.1};

    // Pseudo-inverses (B_inv: p x n, C_inv: n x m)
    double B_inv[] = {0.0, 2.0};
    double C_inv[] = {1.0, 0.0};

    // Monitor thresholds from Table 1 (using TTC values)
    double theta_u_val = 30.0;
    double theta_u_grad = 10.0;
    double theta_y_val = 25.0;
    double theta_y_grad = 30.0;
    double theta_res = 4.35;

    // --- Dynamic "Real-Time" Data ---
    // Simulating a snapshot in time
    double x_tilde_k[] = {0.5, -0.2}; // Estimated state vector [deviation, velocity]
    double u_hat[] = {1.5};           // Last known control signal [acceleration]
    double y_a_k[] = {0.55};          // Current sensor measurement [deviation]

    // --- Output Vectors ---
    double a_out[p], u_tilde_out[p];

    // --- Run the calculation ---
    compute_attack_vector(n, m, p,
        A, B, C, K, L,
        B_inv, C_inv,
        x_tilde_k, u_hat, y_a_k,
        theta_u_val, theta_u_grad, theta_y_val, theta_y_grad, theta_res,
        a_out, u_tilde_out);

    // --- Print Final Results ---
    printf("Computed attack a = %f, u_tilde = %f\n", a_out[0], u_tilde_out[0]);

    return 0;
}