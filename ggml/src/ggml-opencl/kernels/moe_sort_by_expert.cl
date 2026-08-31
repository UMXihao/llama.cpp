#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void kernel_moe_histogram(
    __global const int * input,
    __global int * hist,
    uint N,
    uint topK,
    uint n_experts
) {
    uint n = get_global_id(0);
    uint k = get_global_id(1);

    if (n >= N || k >= topK) {
        return;
    }

    int expert_id = input[n * n_experts + k];
    atomic_inc(&hist[expert_id]);
}

__kernel void kernel_moe_scan(
    __global int * hist,
    __global int * tile_offset,
    __global int * total_tiles,
    __global int * slot_counter,
    int tile_size,
    uint n_experts
) {
    int offset = 0;
    for (int v = 0; v < n_experts; v++) {
        int count = hist[v];
        int tiles = (count + tile_size - 1) / tile_size;
        tile_offset[v] = offset;
        offset += tiles;
        hist[v] = 0;
        slot_counter[v] = 0;
    }

    *total_tiles = offset;
}

__kernel void kernel_moe_scatter(
    __global const int * input,
    __global int * post_router,
    __global ushort * emap,
    __global const int * tile_offset,
    __global int * slot_counter,
    int N,
    int topK,
    uint n_experts,
    int tile_size
) {
    uint n = get_global_id(0);
    uint k = get_global_id(1);

    if (n >= N || k >= topK) {
        return;
    }

    int val = input[n * n_experts + k];

    int local_slot = atomic_inc(&slot_counter[val]);

    int tile_idx  = tile_offset[val] + (local_slot / tile_size);
    int lane      = local_slot % tile_size;
    int out_pos   = tile_idx * tile_size + lane;

    post_router[out_pos] = n * topK + k;
    emap[tile_idx] = val;
}

__kernel void kernel_moe_fill(
    __global int * post_router,
    __global int * total_tiles,
    int tile_size
) {
    int tile_id = get_global_id(0);
    int vec_id_in_tile = get_global_id(1);

    if (tile_id < total_tiles[0]) {
        post_router[tile_id * tile_size + vec_id_in_tile] = 0xFFFFFFFF;
    }
}

// Q4_0 experimental MoE packing: one 32-slot physical tile contains up to
// four different experts, each owning one contiguous 8-slot group.
//
// The legacy kernel_moe_fill is intentionally reused unchanged: it initializes
// every physical slot to 0xFFFFFFFF before scatter fills the valid positions.
__kernel void kernel_moe_scan_4x8(
    __global int * hist,
    __global int * group_offset,
    __global int * group_map,
    __global int * total_tiles,
    __global int * slot_counter,
    int tile_size,
    int expert_slot_size,
    uint n_experts
) {
    const int groups_per_tile = tile_size / expert_slot_size;

    int logical_groups = 0;
    int max_groups = 0;

    // group_offset[e] is the first logical 8-token group for expert e.
    // Keep hist intact until the physical round-robin mapping is built.
    for (int e = 0; e < n_experts; ++e) {
        const int count = hist[e];
        const int groups = (count + expert_slot_size - 1) / expert_slot_size;

        group_offset[e] = logical_groups;
        logical_groups += groups;
        if (groups > max_groups) {
            max_groups = groups;
        }
        slot_counter[e] = 0;
    }

    // Pack one group per active expert in each round. A round always starts on
    // a fresh physical tile, therefore the same expert can never occupy two of
    // the four 8-token groups in one tile.
    int physical_group = 0;
    for (int round = 0; round < max_groups; ++round) {
        int groups_in_tile = 0;

        for (int e = 0; e < n_experts; ++e) {
            const int groups = (hist[e] + expert_slot_size - 1) / expert_slot_size;
            if (round >= groups) {
                continue;
            }

            const int logical_group = group_offset[e] + round;
            group_map[logical_group] = physical_group++;

            ++groups_in_tile;
            if (groups_in_tile == groups_per_tile) {
                groups_in_tile = 0;
            }
        }

        if (groups_in_tile != 0) {
            physical_group += groups_per_tile - groups_in_tile;
        }
    }

    *total_tiles = physical_group / groups_per_tile;

    // Preserve the legacy behavior: histogram and slot counters are ready for
    // the next router reorder invocation.
    for (int e = 0; e < n_experts; ++e) {
        hist[e] = 0;
        slot_counter[e] = 0;
    }
}

__kernel void kernel_moe_scatter_4x8(
    __global const int * input,
    __global int * post_router,
    __global ushort * emap,
    __global const int * group_offset,
    __global const int * group_map,
    __global int * slot_counter,
    int N,
    int topK,
    uint n_experts,
    int tile_size,
    int expert_slot_size
) {
    const uint n = get_global_id(0);
    const uint k = get_global_id(1);

    if (n >= N || k >= topK) {
        return;
    }

    const int expert_id = input[n * n_experts + k];
    const int local_slot = atomic_inc(&slot_counter[expert_id]);
    const int local_group = local_slot / expert_slot_size;
    const int lane_in_group = local_slot % expert_slot_size;

    const int logical_group = group_offset[expert_id] + local_group;
    const int physical_group = group_map[logical_group];
    const int groups_per_tile = tile_size / expert_slot_size;
    const int tile_idx = physical_group / groups_per_tile;
    const int group_in_tile = physical_group % groups_per_tile;
    const int lane = group_in_tile * expert_slot_size + lane_in_group;

    post_router[tile_idx * tile_size + lane] = n * topK + k;

    // emap is indexed by physical 8-token group, i.e. four entries per tile.
    // Multiple lanes of the same group store the same expert id, matching the
    // benign same-value race used by the legacy per-tile emap scatter.
    emap[physical_group] = (ushort)expert_id;
}