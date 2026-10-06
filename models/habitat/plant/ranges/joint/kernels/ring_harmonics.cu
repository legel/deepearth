// Ring circular harmonics of a multi-channel raster pyramid: the fused CUDA path of field.HarmonicField.
//
// For each location b and ring j (radius R_j in level-0 cells) the ring is read at A angles theta_a = 2 pi a / A
// (a = 0 north, clockwise through east) by bilinear interpolation on two pyramid levels, k0 = floor(l) and
// k1 = k0 + 1 with l = clamp(log2 R_j - 1, 0, levels - 1), blended by t = l - k0. Missing values (int8 -128, or
// outside the raster) are left out of a bilinear sample and its corner weights renormalized over the valid corners;
// a sample counts as valid where its blended corner weight exceeds 0.5. With d_a = v(theta_a) - v_centre per
// channel and ok_a the validity, the kernel returns per (b, j, channel):
//
//   S0 = sum_a ok_a d_a,   Sc_m = sum_a ok_a d_a cos(m theta_a),   Ss_m = sum_a ok_a d_a sin(m theta_a)   (m = 1..M),
//   n  = sum_a ok_a,
//
// as [B, R, C, 2M + 2] = (S0, Sc_1..M, Ss_1..M, n). The backward pass returns the derivative of these sums with
// respect to the radii (field values and centre values fixed): the bilinear gradient (quotient rule over the valid
// corners) along d(position)/dR = (-cos theta, sin theta) / 2^k in level-k cells, plus the level-blend term
// (v1 - v0) dt/dR with dt/dR = 1 / (R ln 2) inside the clamp. n is piecewise constant in R and carries no gradient.
// One warp per (location, ring), lanes = angles (A <= 32). Equal to the PyTorch reference
// (field.ring_sums_reference; tests/test_field.py).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int MAXC = 24;              // channels of a field pyramid
constexpr int MAXM = 8;               // harmonic orders
constexpr int MAXL = 12;              // pyramid levels

struct Level { const int8_t* data; int H; int W; };   // pixel-major [H, W, C]: a corner's channels are contiguous

__device__ __forceinline__ void bilinear(const Level& L, float r, float c, int C, float* v, float* gr, float* gc,
                                         float* ws) {
    // per channel: value (SD units; int8 in steps of 1/16), its gradient in (r, c), and the summed bilinear weight
    // of that channel's valid corners, at fractional cell coordinates (cell centres at +0.5)
    const float y = r - 0.5f, x = c - 0.5f;
    const float y0 = floorf(y), x0 = floorf(x);
    const float wy = y - y0, wx = x - x0;
    float wr[MAXC], wc[MAXC];
    for (int ch = 0; ch < C; ch++) { v[ch] = 0.f; gr[ch] = 0.f; gc[ch] = 0.f; ws[ch] = 0.f; wr[ch] = 0.f; wc[ch] = 0.f; }
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            const int yy = (int)y0 + dy, xx = (int)x0 + dx;
            if (yy < 0 || yy >= L.H || xx < 0 || xx >= L.W) continue;
            const float fy = dy ? wy : 1.f - wy, fx = dx ? wx : 1.f - wx;
            const float dfy = dy ? 1.f : -1.f, dfx = dx ? 1.f : -1.f;   // d fy / dr, d fx / dc
            const int8_t* px = L.data + ((size_t)yy * L.W + xx) * C;
            for (int ch = 0; ch < C; ch++) {
                const int8_t q = px[ch];
                if (q == -128) continue;
                const float val = q / 16.f;
                v[ch] += fy * fx * val; gr[ch] += dfy * fx * val; gc[ch] += fy * dfx * val;
                ws[ch] += fy * fx; wr[ch] += dfy * fx; wc[ch] += fy * dfx;
            }
        }
    }
    for (int ch = 0; ch < C; ch++) {                           // normalize by the valid weight (quotient rule)
        if (ws[ch] > 1e-6f) {
            const float num = v[ch], w2 = ws[ch] * ws[ch];
            v[ch] = num / ws[ch];
            gr[ch] = (gr[ch] * ws[ch] - num * wr[ch]) / w2;
            gc[ch] = (gc[ch] * ws[ch] - num * wc[ch]) / w2;
        } else { v[ch] = 0.f; gr[ch] = 0.f; gc[ch] = 0.f; }
    }
}

__device__ __forceinline__ void bilinear_value(const Level& L, float r, float c, int C, float* v, float* ws) {
    // forward only: per-channel value and valid-corner weight (no gradients: fewer registers)
    const float y = r - 0.5f, x = c - 0.5f;
    const float y0 = floorf(y), x0 = floorf(x);
    const float wy = y - y0, wx = x - x0;
    for (int ch = 0; ch < C; ch++) { v[ch] = 0.f; ws[ch] = 0.f; }
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            const int yy = (int)y0 + dy, xx = (int)x0 + dx;
            if (yy < 0 || yy >= L.H || xx < 0 || xx >= L.W) continue;
            const float f = (dy ? wy : 1.f - wy) * (dx ? wx : 1.f - wx);
            const int8_t* px = L.data + ((size_t)yy * L.W + xx) * C;
            for (int ch = 0; ch < C; ch++) {
                const int8_t q = px[ch];
                if (q == -128) continue;
                v[ch] += f * (q / 16.f); ws[ch] += f;
            }
        }
    }
    for (int ch = 0; ch < C; ch++) v[ch] = ws[ch] > 1e-6f ? v[ch] / ws[ch] : 0.f;
}

struct Pyramid { Level lev[MAXL]; int n; };

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__global__ void ring_forward(Pyramid P, const float* __restrict__ rc, const float* __restrict__ radius,
                             const float* __restrict__ vc, float* __restrict__ out, int B, int R, int C, int A, int M) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= B * R) return;
    const int b = warp / R, j = warp % R;
    const float Rj = radius[j];
    float l = log2f(Rj) - 1.f; l = fminf(fmaxf(l, 0.f), (float)(P.n - 1));
    const int k0 = (int)floorf(l), k1 = min(k0 + 1, P.n - 1);
    const float t = l - k0;
    const bool active = lane < A;
    const float th = 2.f * 3.14159265358979f * lane / A;
    const float ct = cosf(th), st = sinf(th);
    float d[MAXC], ok[MAXC];
    for (int ch = 0; ch < C; ch++) { d[ch] = 0.f; ok[ch] = 0.f; }
    if (active) {
        const float r0 = rc[2 * b] - Rj * ct, c0 = rc[2 * b + 1] + Rj * st;    // north = decreasing row
        float v0[MAXC], w0[MAXC], v1[MAXC], w1[MAXC];
        bilinear_value(P.lev[k0], r0 / (1 << k0), c0 / (1 << k0), C, v0, w0);
        const bool two = k1 != k0;
        if (two) bilinear_value(P.lev[k1], r0 / (1 << k1), c0 / (1 << k1), C, v1, w1);
        for (int ch = 0; ch < C; ch++) {
            const float w = two ? (1.f - t) * w0[ch] + t * w1[ch] : w0[ch];
            const float v = two ? (1.f - t) * v0[ch] + t * v1[ch] : v0[ch];
            ok[ch] = w > 0.5f ? 1.f : 0.f;
            d[ch] = ok[ch] * (v - vc[b * C + ch]);
        }
    }
    const int H = 2 * M + 2;                                   // S0, Sc_1..M, Ss_1..M, n
    float* o = out + ((size_t)b * R + j) * C * H;
    for (int ch = 0; ch < C; ch++) {
        const float s0 = warp_sum(d[ch]), n = warp_sum(ok[ch]);
        if (lane == 0) { o[ch * H] = s0; o[ch * H + H - 1] = n; }
        for (int m = 1; m <= M; m++) {
            const float cm = cosf(m * th), sm = sinf(m * th);
            const float sc = warp_sum(d[ch] * cm), ss = warp_sum(d[ch] * sm);
            if (lane == 0) { o[ch * H + m] = sc; o[ch * H + M + m] = ss; }
        }
    }
}

__global__ void ring_backward(Pyramid P, const float* __restrict__ rc, const float* __restrict__ radius,
                              const float* __restrict__ gout, float* __restrict__ gR_bj, int B, int R, int C, int A,
                              int M) {
    // gR_bj[b, j] = sum over channels c and outputs h of gout[b, j, c, h] * d S_h / d R_j
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= B * R) return;
    const int b = warp / R, j = warp % R;
    const float Rj = radius[j];
    const float lr = log2f(Rj) - 1.f;
    const float l = fminf(fmaxf(lr, 0.f), (float)(P.n - 1));
    const bool inside = lr > 0.f && lr < (float)(P.n - 1);    // the blend moves with R only inside the clamp
    const int k0 = (int)floorf(l), k1 = min(k0 + 1, P.n - 1);
    const float t = l - k0;
    const int H = 2 * M + 2;
    float acc = 0.f;
    if (lane < A) {
        const float th = 2.f * 3.14159265358979f * lane / A;
        const float ct = cosf(th), st = sinf(th);
        const float r0 = rc[2 * b] - Rj * ct, c0 = rc[2 * b + 1] + Rj * st;
        float v0[MAXC], g0r[MAXC], g0c[MAXC], w0[MAXC], v1[MAXC], g1r[MAXC], g1c[MAXC], w1[MAXC];
        bilinear(P.lev[k0], r0 / (1 << k0), c0 / (1 << k0), C, v0, g0r, g0c, w0);
        const bool two = k1 != k0;
        if (two) bilinear(P.lev[k1], r0 / (1 << k1), c0 / (1 << k1), C, v1, g1r, g1c, w1);
        const float dtdR = (two && inside) ? 1.f / (Rj * 0.69314718056f) : 0.f;
        const float* g = gout + ((size_t)b * R + j) * C * H;
        for (int ch = 0; ch < C; ch++) {
            const float w = two ? (1.f - t) * w0[ch] + t * w1[ch] : w0[ch];
            if (!(w > 0.5f)) continue;                          // invalid samples carry no gradient
            // dv/dR: the sample moves along (-cos, +sin) in level-0 cells, i.e. / 2^k in level-k cells
            float dv = (two ? (1.f - t) : 1.f) * (g0r[ch] * (-ct) + g0c[ch] * st) / (1 << k0);
            if (two) dv += t * (g1r[ch] * (-ct) + g1c[ch] * st) / (1 << k1) + (v1[ch] - v0[ch]) * dtdR;
            float basis = g[ch * H];
            for (int m = 1; m <= M; m++) basis += g[ch * H + m] * cosf(m * th) + g[ch * H + M + m] * sinf(m * th);
            acc += dv * basis;
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) gR_bj[(size_t)b * R + j] = acc;
}

static Pyramid make_pyramid(const std::vector<at::Tensor>& levels) {
    Pyramid P; P.n = (int)levels.size();
    TORCH_CHECK(P.n >= 1 && P.n <= MAXL, "1 to ", MAXL, " pyramid levels");
    const int C = (int)levels[0].size(2);
    for (int k = 0; k < P.n; k++) {
        TORCH_CHECK(levels[k].is_cuda() && levels[k].dtype() == at::kChar && levels[k].is_contiguous() &&
                    levels[k].dim() == 3 && levels[k].size(2) == C, "levels: contiguous int8 CUDA tensors [H, W, C]");
        P.lev[k] = {levels[k].data_ptr<int8_t>(), (int)levels[k].size(0), (int)levels[k].size(1)};
    }
    return P;
}

at::Tensor forward(const std::vector<at::Tensor>& levels, at::Tensor rc, at::Tensor radius, at::Tensor vc, int A, int M) {
    const c10::cuda::OptionalCUDAGuard guard(rc.device());
    const int B = rc.size(0), R = radius.size(0), C = levels[0].size(2);
    TORCH_CHECK(radius.dim() == 1 && C <= MAXC && A >= 1 && A <= 32 && M <= MAXM, "radius [R], C <= ", MAXC,
                ", A <= 32, M <= ", MAXM);
    TORCH_CHECK(rc.is_contiguous() && radius.is_contiguous() && vc.is_contiguous() && vc.size(0) == B && vc.size(1) == C);
    auto out = at::zeros({B, R, C, 2 * M + 2}, rc.options());
    if (B == 0) return out;
    const int threads = 256, blocks = (int)(((size_t)B * R * 32 + threads - 1) / threads);
    ring_forward<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(make_pyramid(levels), rc.data_ptr<float>(),
        radius.data_ptr<float>(), vc.data_ptr<float>(), out.data_ptr<float>(), B, R, C, A, M);
    return out;
}

at::Tensor backward(const std::vector<at::Tensor>& levels, at::Tensor rc, at::Tensor radius, at::Tensor gout, int A, int M) {
    const c10::cuda::OptionalCUDAGuard guard(rc.device());
    const int B = rc.size(0), R = radius.size(0), C = levels[0].size(2);
    auto gR = at::zeros({B, R}, rc.options());
    if (B == 0) return gR.sum(0);
    auto g = gout.contiguous();
    const int threads = 256, blocks = (int)(((size_t)B * R * 32 + threads - 1) / threads);
    ring_backward<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(make_pyramid(levels), rc.data_ptr<float>(),
        radius.data_ptr<float>(), g.data_ptr<float>(), gR.data_ptr<float>(), B, R, C, A, M);
    return gR.sum(0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ring circular-harmonic sums [B, R, C, 2M + 2]");
    m.def("backward", &backward, "gradient of the sums with respect to the ring radii [R]");
}
