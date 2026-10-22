// Experimental OpenCL MoE Q4 2D-image kernels, compiled by ggml-opencl.cpp.
// Input q is the upstream trans4_ns layout: [expert][qword][output].
static const char * ggml_cl_moe_wimg2d_kernel_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define MOE_IMG_NSG 4
#define MOE_IMG_SG  64

inline int2 moe_img_xy(uint e, uint oc, uint qw, uint M, uint QW,
                       uint layout, uint epc, uint BM, uint BKQ) {
    uint w, h, x, y;
    if (layout == 0) {              // row-major, qword contiguous
        w = QW; h = M;
        x = qw; y = oc;
    } else if (layout == 1) {       // output-channel-major
        w = M; h = QW;
        x = oc; y = qw;
    } else {                        // 2D block, output-channel inner
        uint tk = QW / BKQ;
        w = BM * BKQ;
        h = (M / BM) * tk;
        x = (qw % BKQ) * BM + (oc % BM);
        y = (oc / BM) * tk + (qw / BKQ);
    }
    return (int2)((int)((e % epc) * w + x),
                  (int)((e / epc) * h + y));
}

__kernel void kernel_moe_q_wimg2d_pack(
        __global const uint * src, __global uint * dst,
        uint QW, uint M, uint E, uint layout, uint epc,
        uint BM, uint BKQ, uint image_width) {
    const ulong i = get_global_id(0);
    const ulong stride = (ulong)QW * M;
    if (i >= stride * E) return;
    const uint e = (uint)(i / stride);
    const uint qw = (uint)((i % stride) / M);
    const uint oc = (uint)(i % M);
    const int2 xy = moe_img_xy(e, oc, qw, M, QW, layout, epc, BM, BKQ);
    dst[(size_t)xy.y * image_width + xy.x] = src[i];
}

inline uint moe_img_read_q(read_only image2d_t img,
        uint e, uint oc, uint qw, uint M, uint QW,
        uint layout, uint epc, uint BM, uint BKQ) {
    return read_imageui(img, moe_img_xy(e, oc, qw, M, QW, layout, epc, BM, BKQ)).x;
}

// Row-major specialization: adjacent qwords within each output channel
// occupy consecutive image x coordinates. Coordinates are independent of
// BM/BKQ; the atlas origin is still selected by experts_per_row.
inline uint moe_img_read_q_row(read_only image2d_t img,
        uint e, uint oc, uint qw, uint M, uint QW, uint epc) {
    const int2 xy = (int2)((int)((e % epc) * QW + qw),
                           (int)((e / epc) * M  + oc));
    return read_imageui(img, xy).x;
}

inline float8 moe_img_q4_8(uint v, float scale, float bias) {
    return (float8)(
        (float)((v >>  0) & 15u) * scale + bias,
        (float)((v >>  4) & 15u) * scale + bias,
        (float)((v >>  8) & 15u) * scale + bias,
        (float)((v >> 12) & 15u) * scale + bias,
        (float)((v >> 16) & 15u) * scale + bias,
        (float)((v >> 20) & 15u) * scale + bias,
        (float)((v >> 24) & 15u) * scale + bias,
        (float)((v >> 28) & 15u) * scale + bias);
}

inline float moe_img_dot8(read_only image1d_buffer_t activations,
        uint xoff, float8 w) {
    return dot(read_imagef(activations, (int)xoff), w.lo)
         + dot(read_imagef(activations, (int)(xoff + 1)), w.hi);
}

inline void moe_img_reduce_store(float sum, uint oc, uint route,
        uint M, ulong offsetd, __global float * dst,
        uint sgid, uint lane, __local float * scratch) {
    if (sgid != 0) scratch[(sgid - 1) * MOE_IMG_SG + lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) {
        sum += scratch[lane];
        sum += scratch[MOE_IMG_SG + lane];
        sum += scratch[2 * MOE_IMG_SG + lane];
        dst[(offsetd >> 2) + route * M + oc] = sum;
    }
}

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_0_f32_img2d(
        read_only image2d_t src0_q,
        __global half * src0_d,
        read_only image1d_buffer_t src1,
        __global uint * src2,
        __global float * dst,
        ulong offsetd, int ne00, int ne01, int ne11,
        uint layout, uint epc, uint BM, uint BKQ) {
    const uint oc = get_global_id(0);
    const uint route = get_global_id(2);
    const uint sgid = get_local_id(1);
    const uint lane = get_sub_group_local_id();
    if (oc >= (uint)ne01) return;
    const uint i11 = route % (uint)ne11;
    const uint e = src2[route];
    const uint QW = (uint)ne00 / 8;
    const uint nblocks = (uint)ne00 / 32;
    const size_t d_base = (size_t)e * nblocks * (uint)ne01;
    float sum = 0.0f;
    for (uint ib = sgid; ib < nblocks; ib += MOE_IMG_NSG) {
        const float scale = (float)src0_d[d_base + ib * (uint)ne01 + oc];
        for (uint j = 0; j < 4; ++j) {
            const uint q = moe_img_read_q(src0_q, e, oc, ib * 4 + j,
                    ne01, QW, layout, epc, BM, BKQ);
            const float8 w = moe_img_q4_8(q, scale, -8.0f * scale);
            const uint xoff = i11 * (uint)ne00 / 4 + ib * 8 + j * 2;
            sum += moe_img_dot8(src1, xoff, w);
        }
    }
    __local float scratch[MOE_IMG_SG * (MOE_IMG_NSG - 1)];
    moe_img_reduce_store(sum, oc, route, ne01, offsetd, dst, sgid, lane, scratch);
}

// Specialized row-major Q4_0 MoE decode (no runtime layout branch).
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_0_f32_img2d_row(
        read_only image2d_t src0_q,
        __global half * src0_d,
        read_only image1d_buffer_t src1,
        __global uint * src2,
        __global float * dst,
        ulong offsetd, int ne00, int ne01, int ne11,
        uint layout, uint epc, uint BM, uint BKQ) {
    const uint oc = get_global_id(0);
    const uint route = get_global_id(2);
    const uint sgid = get_local_id(1);
    const uint lane = get_sub_group_local_id();
    if (oc >= (uint)ne01) return;
    const uint i11 = route % (uint)ne11;
    const uint e = src2[route];
    // Keep the same kernel ABI as the shared layout path.
    (void)layout; (void)BM; (void)BKQ;
    const uint QW = (uint)ne00 / 8;
    const uint nblocks = (uint)ne00 / 32;
    const size_t d_base = (size_t)e * nblocks * (uint)ne01;
    float sum = 0.0f;
    for (uint ib = sgid; ib < nblocks; ib += MOE_IMG_NSG) {
        const float scale = (float)src0_d[d_base + ib * (uint)ne01 + oc];
        for (uint j = 0; j < 4; ++j) {
            const uint q = moe_img_read_q_row(src0_q, e, oc, ib * 4 + j,
                    ne01, QW, epc);
            const float8 w = moe_img_q4_8(q, scale, -8.0f * scale);
            const uint xoff = i11 * (uint)ne00 / 4 + ib * 8 + j * 2;
            sum += moe_img_dot8(src1, xoff, w);
        }
    }
    __local float scratch[MOE_IMG_SG * (MOE_IMG_NSG - 1)];
    moe_img_reduce_store(sum, oc, route, ne01, offsetd, dst, sgid, lane, scratch);
}

inline void moe_img_scale_min_k4(uint j, __global const uchar * s,
                                 __private uchar * sv, __private uchar * mn) {
    if (j < 4) {
        *sv = s[j] & 63;
        *mn = s[j + 4] & 63;
    } else {
        *sv = (s[j + 4] & 15) | ((s[j - 4] & 0xc0) >> 2);
        *mn = ((s[j + 4] >> 4) & 15) | ((s[j] & 0xc0) >> 2);
    }
}

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_k_f32_img2d(
        read_only image2d_t src0_q,
        __global half * src0_d,
        __global half * src0_dm,
        __global uchar * src0_s,
        read_only image1d_buffer_t src1,
        __global uint * src2,
        __global float * dst,
        ulong offsetd, int ne00, int ne01, int ne11,
        uint layout, uint epc, uint BM, uint BKQ) {
    const uint oc = get_global_id(0);
    const uint route = get_global_id(2);
    const uint sgid = get_local_id(1);
    const uint lane = get_sub_group_local_id();
    if (oc >= (uint)ne01) return;
    const uint i11 = route % (uint)ne11;
    const uint e = src2[route];
    const uint QW = (uint)ne00 / 8;
    const uint nsb = (uint)ne00 / 256;
    const uint nb = (uint)ne00 / 32;
    const size_t d_base = (size_t)e * nsb * (uint)ne01;
    const uint scales_per_row = nsb * 12;
    float sum = 0.0f;
    for (uint ib = sgid; ib < nb; ib += MOE_IMG_NSG) {
        const uint sb = ib / 8;
        const uint j = ib % 8;
        const float d = (float)src0_d[d_base + sb * (uint)ne01 + oc];
        const float dm = (float)src0_dm[d_base + sb * (uint)ne01 + oc];
        __global const uchar * sc = src0_s +
            ((size_t)e * (uint)ne01 + oc) * scales_per_row + sb * 12;
        uchar sv, mn;
        moe_img_scale_min_k4(j, sc, &sv, &mn);
        const float scale = d * (float)sv;
        const float bias = -dm * (float)mn;
        for (uint v = 0; v < 4; ++v) {
            const uint q = moe_img_read_q(src0_q, e, oc, ib * 4 + v,
                    ne01, QW, layout, epc, BM, BKQ);
            const float8 w = moe_img_q4_8(q, scale, bias);
            const uint xoff = i11 * (uint)ne00 / 4 + ib * 8 + v * 2;
            sum += moe_img_dot8(src1, xoff, w);
        }
    }
    __local float scratch[MOE_IMG_SG * (MOE_IMG_NSG - 1)];
    moe_img_reduce_store(sum, oc, route, ne01, offsetd, dst, sgid, lane, scratch);
}
// Specialized row-major Q4_K MoE decode (no runtime layout branch).
__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_k_f32_img2d_row(
        read_only image2d_t src0_q,
        __global half * src0_d,
        __global half * src0_dm,
        __global uchar * src0_s,
        read_only image1d_buffer_t src1,
        __global uint * src2,
        __global float * dst,
        ulong offsetd, int ne00, int ne01, int ne11,
        uint layout, uint epc, uint BM, uint BKQ) {
    const uint oc = get_global_id(0);
    const uint route = get_global_id(2);
    const uint sgid = get_local_id(1);
    const uint lane = get_sub_group_local_id();
    if (oc >= (uint)ne01) return;
    const uint i11 = route % (uint)ne11;
    const uint e = src2[route];
    // Keep the same kernel ABI as the shared layout path.
    (void)layout; (void)BM; (void)BKQ;
    const uint QW = (uint)ne00 / 8;
    const uint nsb = (uint)ne00 / 256;
    const uint nb = (uint)ne00 / 32;
    const size_t d_base = (size_t)e * nsb * (uint)ne01;
    const uint scales_per_row = nsb * 12;
    float sum = 0.0f;
    for (uint ib = sgid; ib < nb; ib += MOE_IMG_NSG) {
        const uint sb = ib / 8;
        const uint j = ib % 8;
        const float d = (float)src0_d[d_base + sb * (uint)ne01 + oc];
        const float dm = (float)src0_dm[d_base + sb * (uint)ne01 + oc];
        __global const uchar * sc = src0_s +
            ((size_t)e * (uint)ne01 + oc) * scales_per_row + sb * 12;
        uchar sv, mn;
        moe_img_scale_min_k4(j, sc, &sv, &mn);
        const float scale = d * (float)sv;
        const float bias = -dm * (float)mn;
        for (uint v = 0; v < 4; ++v) {
            const uint q = moe_img_read_q_row(src0_q, e, oc, ib * 4 + v,
                    ne01, QW, epc);
            const float8 w = moe_img_q4_8(q, scale, bias);
            const uint xoff = i11 * (uint)ne00 / 4 + ib * 8 + v * 2;
            sum += moe_img_dot8(src1, xoff, w);
        }
    }
    __local float scratch[MOE_IMG_SG * (MOE_IMG_NSG - 1)];
    moe_img_reduce_store(sum, oc, route, ne01, offsetd, dst, sgid, lane, scratch);
}
)CLC";