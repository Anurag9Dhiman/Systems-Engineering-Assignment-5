/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * The driver (main) below is unchanged from the starter. See attn_naive.c for the
 * input and output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): cache and register tiling.                                    */
/* ------------------------------------------------------------------------- */
/*
 * Register tiling. The M4 P-core has 4 FMA pipes of 3-cycle latency, and the measured
 * throughput only reaches 4 FMLA/cycle with 16 independent accumulators (12 give 2.9).
 * Both kernels therefore keep 16 accumulators in registers instead of updating memory:
 *
 *   kernel_qk: 4 rows of Q x 4 rows of K. acc[r][c] += Q[r][k..k+3] * K[c][k..k+3].
 *              Needs no transposed K, and 8 loads per 16 FMAs stays under the 3 loads/cycle
 *              limit. Pairwise additions turn the accumulators into S[r][c] at the end.
 *   kernel_sv: 4 rows of O x 16 columns of O (4 vectors each). acc[r][m] += s[r][c] * V[c][m],
 *              where one register of S supplies 4 multipliers through fmla-by-element.
 *
 * Cache tiling. Rows of S are computed TILE_I at a time and K, V are swept TILE_J rows at a
 * time, with TILE_J * d * 4 bytes sized for the 128 KB L1, so every K and V row is reused by
 * all of the rows in the i-tile instead of streaming from L2 once per row.
 */
#include <arm_neon.h>

#ifndef TILE_I
#define TILE_I 64
#endif
#ifndef TILE_J_BYTES
#define TILE_J_BYTES 65536
#endif

static inline void kernel_qk(const float *restrict Q, const float *restrict K, float *restrict S,
                             int N, int d, float scale)
{
    float32x4_t a00 = vdupq_n_f32(0.0f), a01 = a00, a02 = a00, a03 = a00;
    float32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    float32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    float32x4_t a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    const float *q0 = Q, *q1 = Q + d, *q2 = Q + 2 * d, *q3 = Q + 3 * d;
    const float *k0 = K, *k1 = K + d, *k2 = K + 2 * d, *k3 = K + 3 * d;
    int k = 0;
    for (; k + 4 <= d; k += 4) {
        const float32x4_t x0 = vld1q_f32(q0 + k), x1 = vld1q_f32(q1 + k);
        const float32x4_t x2 = vld1q_f32(q2 + k), x3 = vld1q_f32(q3 + k);
        const float32x4_t y0 = vld1q_f32(k0 + k), y1 = vld1q_f32(k1 + k);
        const float32x4_t y2 = vld1q_f32(k2 + k), y3 = vld1q_f32(k3 + k);
        a00 = vfmaq_f32(a00, x0, y0); a01 = vfmaq_f32(a01, x0, y1);
        a02 = vfmaq_f32(a02, x0, y2); a03 = vfmaq_f32(a03, x0, y3);
        a10 = vfmaq_f32(a10, x1, y0); a11 = vfmaq_f32(a11, x1, y1);
        a12 = vfmaq_f32(a12, x1, y2); a13 = vfmaq_f32(a13, x1, y3);
        a20 = vfmaq_f32(a20, x2, y0); a21 = vfmaq_f32(a21, x2, y1);
        a22 = vfmaq_f32(a22, x2, y2); a23 = vfmaq_f32(a23, x2, y3);
        a30 = vfmaq_f32(a30, x3, y0); a31 = vfmaq_f32(a31, x3, y1);
        a32 = vfmaq_f32(a32, x3, y2); a33 = vfmaq_f32(a33, x3, y3);
    }
    float32x4_t s0 = vpaddq_f32(vpaddq_f32(a00, a01), vpaddq_f32(a02, a03));
    float32x4_t s1 = vpaddq_f32(vpaddq_f32(a10, a11), vpaddq_f32(a12, a13));
    float32x4_t s2 = vpaddq_f32(vpaddq_f32(a20, a21), vpaddq_f32(a22, a23));
    float32x4_t s3 = vpaddq_f32(vpaddq_f32(a30, a31), vpaddq_f32(a32, a33));
    vst1q_f32(S, vmulq_n_f32(s0, scale));
    vst1q_f32(S + N, vmulq_n_f32(s1, scale));
    vst1q_f32(S + 2 * (size_t)N, vmulq_n_f32(s2, scale));
    vst1q_f32(S + 3 * (size_t)N, vmulq_n_f32(s3, scale));
    for (; k < d; k++) /* d not a multiple of 4 */
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                S[(size_t)r * N + c] += Q[r * d + k] * K[c * d + k] * scale;
}

static inline void kernel_sv(const float *restrict S, const float *restrict V, float *restrict O,
                             int nj, int N, int d)
{
    float *o0 = O, *o1 = O + d, *o2 = O + 2 * d, *o3 = O + 3 * d;
    float32x4_t a00 = vld1q_f32(o0), a01 = vld1q_f32(o0 + 4), a02 = vld1q_f32(o0 + 8), a03 = vld1q_f32(o0 + 12);
    float32x4_t a10 = vld1q_f32(o1), a11 = vld1q_f32(o1 + 4), a12 = vld1q_f32(o1 + 8), a13 = vld1q_f32(o1 + 12);
    float32x4_t a20 = vld1q_f32(o2), a21 = vld1q_f32(o2 + 4), a22 = vld1q_f32(o2 + 8), a23 = vld1q_f32(o2 + 12);
    float32x4_t a30 = vld1q_f32(o3), a31 = vld1q_f32(o3 + 4), a32 = vld1q_f32(o3 + 8), a33 = vld1q_f32(o3 + 12);
    const float *s0 = S, *s1 = S + N, *s2 = S + 2 * (size_t)N, *s3 = S + 3 * (size_t)N;
    for (int j = 0; j < nj; j += 4) {
        const float32x4_t x0 = vld1q_f32(s0 + j), x1 = vld1q_f32(s1 + j);
        const float32x4_t x2 = vld1q_f32(s2 + j), x3 = vld1q_f32(s3 + j);
#define ROW(c)                                                                 \
    do {                                                                       \
        const float *v = V + (size_t)(j + (c)) * d;                            \
        const float32x4_t v0 = vld1q_f32(v), v1 = vld1q_f32(v + 4);            \
        const float32x4_t v2 = vld1q_f32(v + 8), v3 = vld1q_f32(v + 12);       \
        a00 = vfmaq_laneq_f32(a00, v0, x0, c); a01 = vfmaq_laneq_f32(a01, v1, x0, c); \
        a02 = vfmaq_laneq_f32(a02, v2, x0, c); a03 = vfmaq_laneq_f32(a03, v3, x0, c); \
        a10 = vfmaq_laneq_f32(a10, v0, x1, c); a11 = vfmaq_laneq_f32(a11, v1, x1, c); \
        a12 = vfmaq_laneq_f32(a12, v2, x1, c); a13 = vfmaq_laneq_f32(a13, v3, x1, c); \
        a20 = vfmaq_laneq_f32(a20, v0, x2, c); a21 = vfmaq_laneq_f32(a21, v1, x2, c); \
        a22 = vfmaq_laneq_f32(a22, v2, x2, c); a23 = vfmaq_laneq_f32(a23, v3, x2, c); \
        a30 = vfmaq_laneq_f32(a30, v0, x3, c); a31 = vfmaq_laneq_f32(a31, v1, x3, c); \
        a32 = vfmaq_laneq_f32(a32, v2, x3, c); a33 = vfmaq_laneq_f32(a33, v3, x3, c); \
    } while (0)
        ROW(0);
        ROW(1);
        ROW(2);
        ROW(3);
#undef ROW
    }
    vst1q_f32(o0, a00); vst1q_f32(o0 + 4, a01); vst1q_f32(o0 + 8, a02); vst1q_f32(o0 + 12, a03);
    vst1q_f32(o1, a10); vst1q_f32(o1 + 4, a11); vst1q_f32(o1 + 8, a12); vst1q_f32(o1 + 12, a13);
    vst1q_f32(o2, a20); vst1q_f32(o2 + 4, a21); vst1q_f32(o2 + 8, a22); vst1q_f32(o2 + 12, a23);
    vst1q_f32(o3, a30); vst1q_f32(o3 + 4, a31); vst1q_f32(o3 + 8, a32); vst1q_f32(o3 + 12, a33);
}

static void attention_impl(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    int tile_j = (TILE_J_BYTES / (d * (int)sizeof(float))) & ~3;
    if (tile_j < 4)
        tile_j = 4;

    /* 1. S = Q K^T * scale */
    for (int ib = 0; ib < N; ib += TILE_I) {
        const int iend = ib + TILE_I < N ? ib + TILE_I : N;
        for (int jb = 0; jb < N; jb += tile_j) {
            const int jend = jb + tile_j < N ? jb + tile_j : N;
            int i = ib;
            for (; i + 4 <= iend; i += 4) {
                int j = jb;
                for (; j + 4 <= jend; j += 4)
                    kernel_qk(Q + (size_t)i * d, K + (size_t)j * d, S + (size_t)i * N + j, N, d, scale);
                for (; j < jend; j++) /* fringe columns */
                    for (int r = i; r < i + 4; r++) {
                        float acc = 0.0f;
                        for (int k = 0; k < d; k++)
                            acc += Q[(size_t)r * d + k] * K[(size_t)j * d + k];
                        S[(size_t)r * N + j] = acc * scale;
                    }
            }
            for (; i < iend; i++) /* fringe rows */
                for (int j = jb; j < jend; j++) {
                    float acc = 0.0f;
                    for (int k = 0; k < d; k++)
                        acc += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
                    S[(size_t)i * N + j] = acc * scale;
                }
        }
    }

    /* 2. softmax of each row of S */
    for (int i = 0; i < N; i++) {
        float *restrict row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            max = fmaxf(max, row[j]);
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }

    /* 3. O = S V */
    memset(O, 0, (size_t)N * d * sizeof(float));
    for (int ib = 0; ib < N; ib += TILE_I) {
        const int iend = ib + TILE_I < N ? ib + TILE_I : N;
        for (int jb = 0; jb < N; jb += tile_j) {
            const int jend = jb + tile_j < N ? jb + tile_j : N;
            const int nj = (jend - jb) & ~3;
            for (int i = ib; i + 4 <= iend; i += 4)
                for (int kb = 0; kb + 16 <= d; kb += 16)
                    kernel_sv(S + (size_t)i * N + jb, V + (size_t)jb * d + kb, O + (size_t)i * d + kb, nj, N, d);
            /* every (row, column) pair the 4x16 kernel did not reach */
            const int rows4 = (iend - ib) & ~3;
            const int k16 = d & ~15;
            for (int i = ib; i < iend; i++) {
                float *restrict o = O + (size_t)i * d;
                const float *restrict row = S + (size_t)i * N;
                const int in4 = i < ib + rows4;
                for (int j = jb; j < jend; j++) {
                    const int j4 = j < jb + nj;
                    const float s = row[j];
                    const float *restrict v = V + (size_t)j * d;
                    for (int k = (in4 && j4) ? k16 : 0; k < d; k++)
                        o[k] += s * v[k];
                }
            }
        }
    }

    free(S);
}

/*
 * The driver's single untimed warm-up call is meant to reach a steady state, but a cold
 * process needs about 100 ms for the clock to ramp from 1.6 to 4.4 GHz (report.pdf, "How I measured"),
 * longer than this variant may take. The first call therefore repeats the work until 200 ms
 * have passed, so that timing starts at the steady clock, as it already does for naive.
 * Only that untimed call changes. Build with -DNO_CLOCK_WARMUP to turn this off.
 */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
#ifndef NO_CLOCK_WARMUP
    static int warmed;
    if (!warmed) {
        struct timespec a, b;
        warmed = 1;
        clock_gettime(CLOCK_MONOTONIC, &a);
        do {
            attention_impl(Q, K, V, O, N, d);
            clock_gettime(CLOCK_MONOTONIC, &b);
        } while ((double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec) < 0.2);
    }
#endif
    attention_impl(Q, K, V, O, N, d);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
