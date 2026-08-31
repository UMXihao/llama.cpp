#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_extra_vector_types : enable

#define TILESIZE_K 16
#define TILESIZE_M 64
#define TILESIZE_N 32


#define dequantize_q4_0(q4, a_f16, scale) \
    a_f16.s0 = (half)((q4.s0 & 0x000F) - 8) * scale; \
    a_f16.s1 = (half)(((q4.s0 & 0x00F0) >> 4) - 8) * scale; \
    a_f16.s2 = (half)(((q4.s0 & 0x0F00) >> 8) - 8) * scale; \
    a_f16.s3 = (half)(((q4.s0 & 0xF000) >> 12) - 8) * scale; \
    a_f16.s4 = (half)((q4.s1 & 0x000F) - 8) * scale; \
    a_f16.s5 = (half)(((q4.s1 & 0x00F0) >> 4) - 8) * scale; \
    a_f16.s6 = (half)(((q4.s1 & 0x0F00) >> 8) - 8) * scale; \
    a_f16.s7 = (half)(((q4.s1 & 0xF000) >> 12) - 8) * scale; \
    a_f16.s8 = (half)((q4.s2 & 0x000F) - 8) * scale; \
    a_f16.s9 = (half)(((q4.s2 & 0x00F0) >> 4) - 8) * scale; \
    a_f16.sa = (half)(((q4.s2 & 0x0F00) >> 8) - 8) * scale; \
    a_f16.sb = (half)(((q4.s2 & 0xF000) >> 12) - 8) * scale; \
    a_f16.sc = (half)((q4.s3 & 0x000F) - 8) * scale; \
    a_f16.sd = (half)(((q4.s3 & 0x00F0) >> 4) - 8) * scale; \
    a_f16.se = (half)(((q4.s3 & 0x0F00) >> 8) - 8) * scale; \
    a_f16.sf = (half)(((q4.s3 & 0xF000) >> 12) - 8) * scale; \


#define dotx16_reduce8(a_reg, b_lm, c_reg, lm_offset) \
    acc.s0 = dot(a_reg.s0123, b_lm[lm_offset + 0]); \
    acc.s1 = dot(a_reg.s0123, b_lm[lm_offset + 1]); \
    acc.s2 = dot(a_reg.s0123, b_lm[lm_offset + 2]); \
    acc.s3 = dot(a_reg.s0123, b_lm[lm_offset + 3]); \
    acc.s4 = dot(a_reg.s0123, b_lm[lm_offset + 4]); \
    acc.s5 = dot(a_reg.s0123, b_lm[lm_offset + 5]); \
    acc.s6 = dot(a_reg.s0123, b_lm[lm_offset + 6]); \
    acc.s7 = dot(a_reg.s0123, b_lm[lm_offset + 7]); \
    acc.s8 = dot(a_reg.s0123, b_lm[lm_offset + 8]); \
    acc.s9 = dot(a_reg.s0123, b_lm[lm_offset + 9]); \
    acc.sa = dot(a_reg.s0123, b_lm[lm_offset + 10]); \
    acc.sb = dot(a_reg.s0123, b_lm[lm_offset + 11]); \
    acc.sc = dot(a_reg.s0123, b_lm[lm_offset + 12]); \
    acc.sd = dot(a_reg.s0123, b_lm[lm_offset + 13]); \
    acc.se = dot(a_reg.s0123, b_lm[lm_offset + 14]); \
    acc.sf = dot(a_reg.s0123, b_lm[lm_offset + 15]); \
    acc.s0 += dot(a_reg.s4567, b_lm[lm_offset + 32]); \
    acc.s1 += dot(a_reg.s4567, b_lm[lm_offset + 33]); \
    acc.s2 += dot(a_reg.s4567, b_lm[lm_offset + 34]); \
    acc.s3 += dot(a_reg.s4567, b_lm[lm_offset + 35]); \
    acc.s4 += dot(a_reg.s4567, b_lm[lm_offset + 36]); \
    acc.s5 += dot(a_reg.s4567, b_lm[lm_offset + 37]); \
    acc.s6 += dot(a_reg.s4567, b_lm[lm_offset + 38]); \
    acc.s7 += dot(a_reg.s4567, b_lm[lm_offset + 39]); \
    acc.s8 += dot(a_reg.s4567, b_lm[lm_offset + 40]); \
    acc.s9 += dot(a_reg.s4567, b_lm[lm_offset + 41]); \
    acc.sa += dot(a_reg.s4567, b_lm[lm_offset + 42]); \
    acc.sb += dot(a_reg.s4567, b_lm[lm_offset + 43]); \
    acc.sc += dot(a_reg.s4567, b_lm[lm_offset + 44]); \
    acc.sd += dot(a_reg.s4567, b_lm[lm_offset + 45]); \
    acc.se += dot(a_reg.s4567, b_lm[lm_offset + 46]); \
    acc.sf += dot(a_reg.s4567, b_lm[lm_offset + 47]); \
    c_reg.lo += convert_float8(acc.lo); \
    c_reg.hi += convert_float8(acc.hi); \
    acc.s0 = dot(a_reg.s89ab, b_lm[lm_offset + 64]); \
    acc.s1 = dot(a_reg.s89ab, b_lm[lm_offset + 65]); \
    acc.s2 = dot(a_reg.s89ab, b_lm[lm_offset + 66]); \
    acc.s3 = dot(a_reg.s89ab, b_lm[lm_offset + 67]); \
    acc.s4 = dot(a_reg.s89ab, b_lm[lm_offset + 68]); \
    acc.s5 = dot(a_reg.s89ab, b_lm[lm_offset + 69]); \
    acc.s6 = dot(a_reg.s89ab, b_lm[lm_offset + 70]); \
    acc.s7 = dot(a_reg.s89ab, b_lm[lm_offset + 71]); \
    acc.s8 = dot(a_reg.s89ab, b_lm[lm_offset + 72]); \
    acc.s9 = dot(a_reg.s89ab, b_lm[lm_offset + 73]); \
    acc.sa = dot(a_reg.s89ab, b_lm[lm_offset + 74]); \
    acc.sb = dot(a_reg.s89ab, b_lm[lm_offset + 75]); \
    acc.sc = dot(a_reg.s89ab, b_lm[lm_offset + 76]); \
    acc.sd = dot(a_reg.s89ab, b_lm[lm_offset + 77]); \
    acc.se = dot(a_reg.s89ab, b_lm[lm_offset + 78]); \
    acc.sf = dot(a_reg.s89ab, b_lm[lm_offset + 79]); \
    acc.s0 += dot(a_reg.scdef, b_lm[lm_offset + 96]); \
    acc.s1 += dot(a_reg.scdef, b_lm[lm_offset + 97]); \
    acc.s2 += dot(a_reg.scdef, b_lm[lm_offset + 98]); \
    acc.s3 += dot(a_reg.scdef, b_lm[lm_offset + 99]); \
    acc.s4 += dot(a_reg.scdef, b_lm[lm_offset + 100]); \
    acc.s5 += dot(a_reg.scdef, b_lm[lm_offset + 101]); \
    acc.s6 += dot(a_reg.scdef, b_lm[lm_offset + 102]); \
    acc.s7 += dot(a_reg.scdef, b_lm[lm_offset + 103]); \
    acc.s8 += dot(a_reg.scdef, b_lm[lm_offset + 104]); \
    acc.s9 += dot(a_reg.scdef, b_lm[lm_offset + 105]); \
    acc.sa += dot(a_reg.scdef, b_lm[lm_offset + 106]); \
    acc.sb += dot(a_reg.scdef, b_lm[lm_offset + 107]); \
    acc.sc += dot(a_reg.scdef, b_lm[lm_offset + 108]); \
    acc.sd += dot(a_reg.scdef, b_lm[lm_offset + 109]); \
    acc.se += dot(a_reg.scdef, b_lm[lm_offset + 110]); \
    acc.sf += dot(a_reg.scdef, b_lm[lm_offset + 111]); \
    c_reg.lo += convert_float8(acc.lo); \
    c_reg.hi += convert_float8(acc.hi); \

// Quarter-tile variant: computes 8 output columns (one skip-group) into a float8
// accumulator. Same reduction order / flush cadence as dotx16_reduce8, so the
// non-skipped path is byte-identical; it just lets the caller skip empty
// 8-column groups at finer granularity. Uses a private half8 `acc8`.
#define dotx8_reduce4(a_reg, b_lm, c_reg, lm_offset) \
    acc8.s0 = dot(a_reg.s0123, b_lm[lm_offset + 0]); \
    acc8.s1 = dot(a_reg.s0123, b_lm[lm_offset + 1]); \
    acc8.s2 = dot(a_reg.s0123, b_lm[lm_offset + 2]); \
    acc8.s3 = dot(a_reg.s0123, b_lm[lm_offset + 3]); \
    acc8.s4 = dot(a_reg.s0123, b_lm[lm_offset + 4]); \
    acc8.s5 = dot(a_reg.s0123, b_lm[lm_offset + 5]); \
    acc8.s6 = dot(a_reg.s0123, b_lm[lm_offset + 6]); \
    acc8.s7 = dot(a_reg.s0123, b_lm[lm_offset + 7]); \
    acc8.s0 += dot(a_reg.s4567, b_lm[lm_offset + 32]); \
    acc8.s1 += dot(a_reg.s4567, b_lm[lm_offset + 33]); \
    acc8.s2 += dot(a_reg.s4567, b_lm[lm_offset + 34]); \
    acc8.s3 += dot(a_reg.s4567, b_lm[lm_offset + 35]); \
    acc8.s4 += dot(a_reg.s4567, b_lm[lm_offset + 36]); \
    acc8.s5 += dot(a_reg.s4567, b_lm[lm_offset + 37]); \
    acc8.s6 += dot(a_reg.s4567, b_lm[lm_offset + 38]); \
    acc8.s7 += dot(a_reg.s4567, b_lm[lm_offset + 39]); \
    c_reg += convert_float8(acc8); \
    acc8.s0 = dot(a_reg.s89ab, b_lm[lm_offset + 64]); \
    acc8.s1 = dot(a_reg.s89ab, b_lm[lm_offset + 65]); \
    acc8.s2 = dot(a_reg.s89ab, b_lm[lm_offset + 66]); \
    acc8.s3 = dot(a_reg.s89ab, b_lm[lm_offset + 67]); \
    acc8.s4 = dot(a_reg.s89ab, b_lm[lm_offset + 68]); \
    acc8.s5 = dot(a_reg.s89ab, b_lm[lm_offset + 69]); \
    acc8.s6 = dot(a_reg.s89ab, b_lm[lm_offset + 70]); \
    acc8.s7 = dot(a_reg.s89ab, b_lm[lm_offset + 71]); \
    acc8.s0 += dot(a_reg.scdef, b_lm[lm_offset + 96]); \
    acc8.s1 += dot(a_reg.scdef, b_lm[lm_offset + 97]); \
    acc8.s2 += dot(a_reg.scdef, b_lm[lm_offset + 98]); \
    acc8.s3 += dot(a_reg.scdef, b_lm[lm_offset + 99]); \
    acc8.s4 += dot(a_reg.scdef, b_lm[lm_offset + 100]); \
    acc8.s5 += dot(a_reg.scdef, b_lm[lm_offset + 101]); \
    acc8.s6 += dot(a_reg.scdef, b_lm[lm_offset + 102]); \
    acc8.s7 += dot(a_reg.scdef, b_lm[lm_offset + 103]); \
    c_reg += convert_float8(acc8); \


__attribute__((qcom_wave_pair_mode(1))) // 1=force single 2=force pair
kernel void kernel_gemm_moe_q4_0_f32_ns(
        __read_only  image1d_buffer_t src0_q,
        __global     half *           src0_d,
        __read_only  image1d_buffer_t src1,
        __global     uint *           src2,
        __global     ushort *         src2_emap,
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        uint is_ragged,
        uint skip_gran
) {
    uint block_id_m = get_global_id(1); // m_tile
    uint block_id_n = get_global_id(2); // n_tile

    // Boundary check
    if (block_id_n >= total_tiles[0]) {
        return;
    }

    // Ragged tile-skip: when is_ragged and the upper 16 token-slots of this tile are all
    // padding (router 0xFFFFFFFF), skip the second (reg_c.hi) dotx16_reduce8 half -> ~half
    // the GEMM dot for sparse tiles. Numerically identical (the skipped lanes are padding).
    // Ragged tile-skip: tokens are packed contiguously per expert (moe_scatter fills
    // lanes 0..V-1, moe_fill pre-pads the rest), so router padding (0xFFFFFFFF) is always
    // trailing. Find the valid-token count V and round it UP to the skip granularity
    // skip_gran (columns per skip-group: 8 = quarter, 16 = half/legacy, 32 = disabled).
    // A 8-column group g is all-padding iff its first column (8*g) >= n_active, so its
    // dotx8_reduce4 is skipped. Numerically identical (skipped lanes are padding).
    uint n_active = TILESIZE_N;
    if (is_ragged && skip_gran < TILESIZE_N) {
        uint n_valid = TILESIZE_N;
        for (uint _t = 0; _t < TILESIZE_N; ++_t) {
            if (src2[block_id_n * TILESIZE_N + _t] == 0xFFFFFFFFu) { n_valid = _t; break; }
        }
        n_active = min((uint)TILESIZE_N, ((n_valid + skip_gran - 1) / skip_gran) * skip_gran);
    }
    // Group 0 (cols 0-7) always runs; groups 1-3 skip when fully padding.
    bool skip_g1 = (8u  >= n_active);
    bool skip_g2 = (16u >= n_active);
    bool skip_g3 = (24u >= n_active);

    __private half16 reg_a;
    __private float32 reg_c = (float32)(0);
    __local half4 shared_b[128];

    const ushort expert_id = src2_emap[block_id_n];

    const uint row = block_id_m * TILESIZE_M;
    const uint col = block_id_n * TILESIZE_N;

    uint sub_block_id_m = get_local_id(0);
    uint2 b_global_offset;
    b_global_offset.x = ((sub_block_id_m & 3) << 2) + (sub_block_id_m >> 2) * ne00;
    b_global_offset.y = b_global_offset.x + (16 * ne00);
    uint2 b_local_offset;
    b_local_offset.x = (sub_block_id_m & 3) * 32 + (sub_block_id_m >> 2);
    b_local_offset.y = b_local_offset.x + 16;

    // Loop along K axis, 32 elements (one block) for each iteration, divided into 2 sub-blocks
    for (uint step = 0; step < ne00; step += TILESIZE_K * 2) {
        // First sub-block
        uint q_sub_offset = row + ((ne01 * step) >> 3) + ((expert_id * ne00 * ne01) >> 3);
        uint s_sub_offset = row + ((ne01 * step) >> 5) + ((expert_id * ne00 * ne01) >> 5);
        uint b_sub_offset = col * ne00 + step;

        // Load scale for current Q4_0 block
        uint s_offset = s_sub_offset + get_global_id(0);
        half s = src0_d[s_offset];

        // Load 16 q (64-bits) in transposed layout
        uint2 q4x16;
        q4x16.x = read_imageui(src0_q, q_sub_offset + sub_block_id_m).x;
        q4x16.y = read_imageui(src0_q, q_sub_offset + sub_block_id_m + ne01).x;

        // Load 16x32 floats from matrix B, each fiber out of 64 in a sub-group loads 8 elements
        float8 bx8_f32;
        bx8_f32.lo = read_imagef(src1, (b_sub_offset + b_global_offset.x) / 4);
        bx8_f32.hi = read_imagef(src1, (b_sub_offset + b_global_offset.y) / 4);
        // Convert to half and store to LM to share within the subgroup
        half8 bx8_f16 = convert_half8(bx8_f32);
        shared_b[b_local_offset.x] = bx8_f16.lo;
        shared_b[b_local_offset.y] = bx8_f16.hi;

        // Dequantization
        dequantize_q4_0(as_ushort4(q4x16), reg_a, s);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // 32 16x16 fp16 dot product with 8 elements reduction for better precision
        half8 acc8;
        dotx8_reduce4(reg_a, shared_b, reg_c.lo.lo, 0);
        if (!skip_g1) { dotx8_reduce4(reg_a, shared_b, reg_c.lo.hi, 8); }
        if (!skip_g2) { dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16); }
        if (!skip_g3) { dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24); }

        // Repeat for second sub-block
        uint half_step = step + TILESIZE_K;
        q_sub_offset = row + ((ne01 * half_step) >> 3) + ((expert_id * ne00 * ne01) >> 3);
        b_sub_offset = col * ne00 + half_step;

        // Load next 16 q (64-bits) in transposed layout
        q4x16.x = read_imageui(src0_q, q_sub_offset + sub_block_id_m).x;
        q4x16.y = read_imageui(src0_q, q_sub_offset + sub_block_id_m + ne01).x;

        // Load 16x32 floats from matrix B, each fiber out of 64 in a sub-group loads 8 elements
        bx8_f32.lo = read_imagef(src1, (b_sub_offset + b_global_offset.x) / 4);
        bx8_f32.hi = read_imagef(src1, (b_sub_offset + b_global_offset.y) / 4);
        // Convert to half and store to LM to share within the subgroup
        bx8_f16 = convert_half8(bx8_f32);
        shared_b[b_local_offset.x] = bx8_f16.lo;
        shared_b[b_local_offset.y] = bx8_f16.hi;

        // Dequantization
        dequantize_q4_0(as_ushort4(q4x16), reg_a, s);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // 32 16x16 fp16 dot product with 3-levels reduction for better precision
        dotx8_reduce4(reg_a, shared_b, reg_c.lo.lo, 0);
        if (!skip_g1) { dotx8_reduce4(reg_a, shared_b, reg_c.lo.hi, 8); }
        if (!skip_g2) { dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16); }
        if (!skip_g3) { dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24); }
    }

    if ((get_global_id(0) + block_id_m * TILESIZE_M) >= ne01) {
        return;
    }

    // Load poster router and share in LM
    __local uint out_idx[TILESIZE_N];

    if (get_local_id(0) < TILESIZE_N) {
        uint idx = src2[block_id_n * TILESIZE_N + get_local_id(0)];
        if (idx == 0xFFFFFFFF) {
            idx = src2[block_id_n * TILESIZE_N + 0];
        }
        out_idx[get_local_id(0)] = idx * ne01;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Scatter results back to original position in output grid
    uint m_offset = row + get_local_id(0);

    write_imagef(dst, out_idx[1] + m_offset, (reg_c.s1));
    write_imagef(dst, out_idx[2] + m_offset, (reg_c.s2));
    write_imagef(dst, out_idx[3] + m_offset, (reg_c.s3));
    write_imagef(dst, out_idx[4] + m_offset, (reg_c.s4));
    write_imagef(dst, out_idx[5] + m_offset, (reg_c.s5));
    write_imagef(dst, out_idx[6] + m_offset, (reg_c.s6));
    write_imagef(dst, out_idx[7] + m_offset, (reg_c.s7));
    write_imagef(dst, out_idx[8] + m_offset, (reg_c.s8));
    write_imagef(dst, out_idx[9] + m_offset, (reg_c.s9));
    write_imagef(dst, out_idx[10] + m_offset, (reg_c.sa));
    write_imagef(dst, out_idx[11] + m_offset, (reg_c.sb));
    write_imagef(dst, out_idx[12] + m_offset, (reg_c.sc));
    write_imagef(dst, out_idx[13] + m_offset, (reg_c.sd));
    write_imagef(dst, out_idx[14] + m_offset, (reg_c.se));
    write_imagef(dst, out_idx[15] + m_offset, (reg_c.sf));
    write_imagef(dst, out_idx[16] + m_offset, (reg_c.sg));
    write_imagef(dst, out_idx[17] + m_offset, (reg_c.sh));
    write_imagef(dst, out_idx[18] + m_offset, (reg_c.si));
    write_imagef(dst, out_idx[19] + m_offset, (reg_c.sj));
    write_imagef(dst, out_idx[20] + m_offset, (reg_c.sk));
    write_imagef(dst, out_idx[21] + m_offset, (reg_c.sl));
    write_imagef(dst, out_idx[22] + m_offset, (reg_c.sm));
    write_imagef(dst, out_idx[23] + m_offset, (reg_c.sn));
    write_imagef(dst, out_idx[24] + m_offset, (reg_c.so));
    write_imagef(dst, out_idx[25] + m_offset, (reg_c.sp));
    write_imagef(dst, out_idx[26] + m_offset, (reg_c.sq));
    write_imagef(dst, out_idx[27] + m_offset, (reg_c.sr));
    write_imagef(dst, out_idx[28] + m_offset, (reg_c.ss));
    write_imagef(dst, out_idx[29] + m_offset, (reg_c.st));
    write_imagef(dst, out_idx[30] + m_offset, (reg_c.su));
    write_imagef(dst, out_idx[31] + m_offset, (reg_c.sv));

    // Store zero padding parts to the index of first output in tile, override correct result in the end
    barrier(CLK_GLOBAL_MEM_FENCE);
    write_imagef(dst, out_idx[0] + m_offset, (reg_c.s0));
}

// Diagnostic-only: replay one token tile and touch only the Q4_0 expert weights.
// Host-side CL event timing of a z-size=1 launch is reported as that tile's
// isolated expert-weight load wall time.
kernel void kernel_moe_q4_0_weight_load_probe(
        __read_only image1d_buffer_t src0_q,
        __global half *              src0_d,
        __global ushort *            src2_emap,
        __global int *               total_tiles,
        __global uint *              sink,
        uint ne00,
        uint ne01) {
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    if (block_id_n >= (uint) total_tiles[0]) {
        return;
    }

    const ushort expert_id = src2_emap[block_id_n];
    const uint row = block_id_m * TILESIZE_M;
    const uint sub_block_id_m = get_local_id(0);

    uint checksum = 0x9e3779b9u ^ (uint) expert_id;

    // Match the source FP16 MoE kernel's expert-weight traversal.
    for (uint step = 0; step < ne00; step += TILESIZE_K * 2) {
        uint q_sub_offset =
            row +
            ((ne01 * step) >> 3) +
            ((expert_id * ne00 * ne01) >> 3);

        const uint s_sub_offset =
            row +
            ((ne01 * step) >> 5) +
            ((expert_id * ne00 * ne01) >> 5);

        // One Q4_0 scale for this 32-element K block.
        const half s = src0_d[s_sub_offset + get_global_id(0)];

        // First 16 q values.
        uint q0 = read_imageui(src0_q, q_sub_offset + sub_block_id_m).x;
        uint q1 = read_imageui(src0_q, q_sub_offset + sub_block_id_m + ne01).x;

        checksum ^= q0;
        checksum = rotate(checksum, 5u) ^ q1;
        checksum ^= (uint) as_ushort(s);

        // Second 16 q values.
        const uint half_step = step + TILESIZE_K;
        q_sub_offset =
            row +
            ((ne01 * half_step) >> 3) +
            ((expert_id * ne00 * ne01) >> 3);

        q0 = read_imageui(src0_q, q_sub_offset + sub_block_id_m).x;
        q1 = read_imageui(src0_q, q_sub_offset + sub_block_id_m + ne01).x;

        checksum = rotate(checksum, 7u) ^ q0;
        checksum = rotate(checksum, 11u) ^ q1;
    }

    // Observable side effect: prevents dead-load elimination.
    sink[block_id_m * TILESIZE_M + get_local_id(0)] = checksum;
}

// Experimental Q4_0 MoE GEMM for the 4 experts x 8 token packing layout.
// The work-group geometry is unchanged (64 threads, TILESIZE_N=32). The B tile
// is loaded once, while each 8-column quarter selects its own expert weights.
#define Q4_0_LOAD_EXPERT_4X8(EXPERT_ID, Q_STEP, SCALE) do { \
    const uint _q_sub_offset = row + ((ne01 * (Q_STEP)) >> 3) + (((uint)(EXPERT_ID) * ne00 * ne01) >> 3); \
    uint2 _q4x16; \
    _q4x16.x = read_imageui(src0_q, _q_sub_offset + sub_block_id_m).x; \
    _q4x16.y = read_imageui(src0_q, _q_sub_offset + sub_block_id_m + ne01).x; \
    dequantize_q4_0(as_ushort4(_q4x16), reg_a, (SCALE)); \
} while (0)

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_moe_q4_0_f32_ns_4x8(
        __read_only  image1d_buffer_t src0_q,
        __global     half *           src0_d,
        __read_only  image1d_buffer_t src1,
        __global     uint *           src2,
        __global     ushort *         src2_emap,
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        uint is_ragged,
        uint skip_gran
) {
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);
    if (block_id_n >= total_tiles[0]) {
        return;
    }

    // This kernel obtains sparsity directly from each 8-slot group. Keep the
    // two legacy arguments so host-side argument numbering remains identical.
    (void)is_ragged;
    (void)skip_gran;

    const uint router_base = block_id_n * TILESIZE_N;
    const bool active_g0 = src2[router_base +  0] != 0xFFFFFFFFu;
    const bool active_g1 = src2[router_base +  8] != 0xFFFFFFFFu;
    const bool active_g2 = src2[router_base + 16] != 0xFFFFFFFFu;
    const bool active_g3 = src2[router_base + 24] != 0xFFFFFFFFu;

    // Empty groups have no valid emap entry. Never dereference those entries.
    const ushort expert0 = active_g0 ? src2_emap[block_id_n * 4 + 0] : (ushort)0;
    const ushort expert1 = active_g1 ? src2_emap[block_id_n * 4 + 1] : (ushort)0;
    const ushort expert2 = active_g2 ? src2_emap[block_id_n * 4 + 2] : (ushort)0;
    const ushort expert3 = active_g3 ? src2_emap[block_id_n * 4 + 3] : (ushort)0;

    __private half16 reg_a;
    __private float32 reg_c = (float32)(0);
    __local half4 shared_b[128];

    const uint row = block_id_m * TILESIZE_M;
    const uint col = block_id_n * TILESIZE_N;
    const uint sub_block_id_m = get_local_id(0);

    uint2 b_global_offset;
    b_global_offset.x = ((sub_block_id_m & 3) << 2) + (sub_block_id_m >> 2) * ne00;
    b_global_offset.y = b_global_offset.x + (16 * ne00);

    uint2 b_local_offset;
    b_local_offset.x = (sub_block_id_m & 3) * 32 + (sub_block_id_m >> 2);
    b_local_offset.y = b_local_offset.x + 16;

    for (uint step = 0; step < ne00; step += TILESIZE_K * 2) {
        // A Q4_0 scale covers a full 32-element block, so the same scale is
        // reused for the first and second 16-K halves of this iteration.
        const uint scale_step_offset = row + ((ne01 * step) >> 5) + get_global_id(0);
        const half s0 = active_g0
            ? src0_d[scale_step_offset + (((uint)expert0 * ne00 * ne01) >> 5)]
            : (half)0;
        const half s1 = active_g1
            ? ((active_g0 && expert1 == expert0)
                ? s0
                : src0_d[scale_step_offset + (((uint)expert1 * ne00 * ne01) >> 5)])
            : (half)0;
        const half s2 = active_g2
            ? ((active_g1 && expert2 == expert1)
                ? s1
                : src0_d[scale_step_offset + (((uint)expert2 * ne00 * ne01) >> 5)])
            : (half)0;
        const half s3 = active_g3
            ? ((active_g2 && expert3 == expert2)
                ? s2
                : src0_d[scale_step_offset + (((uint)expert3 * ne00 * ne01) >> 5)])
            : (half)0;

        // First 16-K half: load the whole 32-column B tile once.
        uint b_sub_offset = col * ne00 + step;
        float8 bx8_f32;
        bx8_f32.lo = read_imagef(src1, (b_sub_offset + b_global_offset.x) / 4);
        bx8_f32.hi = read_imagef(src1, (b_sub_offset + b_global_offset.y) / 4);
        half8 bx8_f16 = convert_half8(bx8_f32);
        shared_b[b_local_offset.x] = bx8_f16.lo;
        shared_b[b_local_offset.y] = bx8_f16.hi;
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        half8 acc8;
        // Router packing keeps repeated experts contiguous. Load/dequantize A
        // once per expert run, then reuse reg_a for every 8-column group in it.
        if (active_g0) {
            Q4_0_LOAD_EXPERT_4X8(expert0, step, s0);
            dotx8_reduce4(reg_a, shared_b, reg_c.lo.lo, 0);
            if (active_g1 && expert1 == expert0) {
                dotx8_reduce4(reg_a, shared_b, reg_c.lo.hi, 8);
            }
            if (active_g2 && expert1 == expert0 && expert2 == expert0) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16);
            }
            if (active_g3 && expert1 == expert0 && expert2 == expert0 && expert3 == expert0) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
            }
        }
        if (active_g1 && (!active_g0 || expert1 != expert0)) {
            Q4_0_LOAD_EXPERT_4X8(expert1, step, s1);
            dotx8_reduce4(reg_a, shared_b, reg_c.lo.hi, 8);
            if (active_g2 && expert2 == expert1) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16);
            }
            if (active_g3 && expert2 == expert1 && expert3 == expert1) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
            }
        }
        if (active_g2 && (!active_g1 || expert2 != expert1)) {
            Q4_0_LOAD_EXPERT_4X8(expert2, step, s2);
            dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16);
            if (active_g3 && expert3 == expert2) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
            }
        }
        if (active_g3 && (!active_g2 || expert3 != expert2)) {
            Q4_0_LOAD_EXPERT_4X8(expert3, step, s3);
            dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
        }

        // Second 16-K half.
        const uint half_step = step + TILESIZE_K;
        b_sub_offset = col * ne00 + half_step;
        bx8_f32.lo = read_imagef(src1, (b_sub_offset + b_global_offset.x) / 4);
        bx8_f32.hi = read_imagef(src1, (b_sub_offset + b_global_offset.y) / 4);
        bx8_f16 = convert_half8(bx8_f32);
        shared_b[b_local_offset.x] = bx8_f16.lo;
        shared_b[b_local_offset.y] = bx8_f16.hi;
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        if (active_g0) {
            Q4_0_LOAD_EXPERT_4X8(expert0, half_step, s0);
            dotx8_reduce4(reg_a, shared_b, reg_c.lo.lo, 0);
            if (active_g1 && expert1 == expert0) {
                dotx8_reduce4(reg_a, shared_b, reg_c.lo.hi, 8);
            }
            if (active_g2 && expert1 == expert0 && expert2 == expert0) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16);
            }
            if (active_g3 && expert1 == expert0 && expert2 == expert0 && expert3 == expert0) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
            }
        }
        if (active_g1 && (!active_g0 || expert1 != expert0)) {
            Q4_0_LOAD_EXPERT_4X8(expert1, half_step, s1);
            dotx8_reduce4(reg_a, shared_b, reg_c.lo.hi, 8);
            if (active_g2 && expert2 == expert1) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16);
            }
            if (active_g3 && expert2 == expert1 && expert3 == expert1) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
            }
        }
        if (active_g2 && (!active_g1 || expert2 != expert1)) {
            Q4_0_LOAD_EXPERT_4X8(expert2, half_step, s2);
            dotx8_reduce4(reg_a, shared_b, reg_c.hi.lo, 16);
            if (active_g3 && expert3 == expert2) {
                dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
            }
        }
        if (active_g3 && (!active_g2 || expert3 != expert2)) {
            Q4_0_LOAD_EXPERT_4X8(expert3, half_step, s3);
            dotx8_reduce4(reg_a, shared_b, reg_c.hi.hi, 24);
        }
    }
    
    // Share the post-router once per work-group. All work-items must reach the
    // barrier, including the lanes outside ne01 in the last M tile.
    __local uint out_idx[TILESIZE_N];
    if (get_local_id(0) < TILESIZE_N) {
        out_idx[get_local_id(0)] = src2[router_base + get_local_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint m_offset = row + get_local_id(0);
    if (m_offset >= ne01) {
        return;
    }

    // qcom_extra_vector_types float32 exposes named components s0..sv but is
    // not subscriptable on the Qualcomm OpenCL compiler. Also, unlike the
    // legacy 1x32 kernel, padding must not alias slot 0 because each 8-column
    // quarter can belong to a different expert.
    if (out_idx[0] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[0] * ne01 + m_offset, (reg_c.s0));
    }
    if (out_idx[1] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[1] * ne01 + m_offset, (reg_c.s1));
    }
    if (out_idx[2] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[2] * ne01 + m_offset, (reg_c.s2));
    }
    if (out_idx[3] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[3] * ne01 + m_offset, (reg_c.s3));
    }
    if (out_idx[4] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[4] * ne01 + m_offset, (reg_c.s4));
    }
    if (out_idx[5] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[5] * ne01 + m_offset, (reg_c.s5));
    }
    if (out_idx[6] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[6] * ne01 + m_offset, (reg_c.s6));
    }
    if (out_idx[7] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[7] * ne01 + m_offset, (reg_c.s7));
    }
    if (out_idx[8] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[8] * ne01 + m_offset, (reg_c.s8));
    }
    if (out_idx[9] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[9] * ne01 + m_offset, (reg_c.s9));
    }
    if (out_idx[10] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[10] * ne01 + m_offset, (reg_c.sa));
    }
    if (out_idx[11] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[11] * ne01 + m_offset, (reg_c.sb));
    }
    if (out_idx[12] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[12] * ne01 + m_offset, (reg_c.sc));
    }
    if (out_idx[13] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[13] * ne01 + m_offset, (reg_c.sd));
    }
    if (out_idx[14] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[14] * ne01 + m_offset, (reg_c.se));
    }
    if (out_idx[15] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[15] * ne01 + m_offset, (reg_c.sf));
    }
    if (out_idx[16] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[16] * ne01 + m_offset, (reg_c.sg));
    }
    if (out_idx[17] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[17] * ne01 + m_offset, (reg_c.sh));
    }
    if (out_idx[18] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[18] * ne01 + m_offset, (reg_c.si));
    }
    if (out_idx[19] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[19] * ne01 + m_offset, (reg_c.sj));
    }
    if (out_idx[20] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[20] * ne01 + m_offset, (reg_c.sk));
    }
    if (out_idx[21] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[21] * ne01 + m_offset, (reg_c.sl));
    }
    if (out_idx[22] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[22] * ne01 + m_offset, (reg_c.sm));
    }
    if (out_idx[23] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[23] * ne01 + m_offset, (reg_c.sn));
    }
    if (out_idx[24] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[24] * ne01 + m_offset, (reg_c.so));
    }
    if (out_idx[25] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[25] * ne01 + m_offset, (reg_c.sp));
    }
    if (out_idx[26] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[26] * ne01 + m_offset, (reg_c.sq));
    }
    if (out_idx[27] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[27] * ne01 + m_offset, (reg_c.sr));
    }
    if (out_idx[28] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[28] * ne01 + m_offset, (reg_c.ss));
    }
    if (out_idx[29] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[29] * ne01 + m_offset, (reg_c.st));
    }
    if (out_idx[30] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[30] * ne01 + m_offset, (reg_c.su));
    }
    if (out_idx[31] != 0xFFFFFFFFu) {
        write_imagef(dst, out_idx[31] * ne01 + m_offset, (reg_c.sv));
    }
}

#undef Q4_0_LOAD_EXPERT_4X8