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

// Q4_0 experimental MoE packing: one 32-slot physical tile contains four
// contiguous 8-slot groups. Packing minimizes physical tiles first and expert
// runs second, so common patterns are 4, 3+1, 2+2, 2+1+1 and 1+1+1+1.
// The legacy kernel_moe_fill is intentionally reused unchanged: it initializes
// every physical slot to 0xFFFFFFFF before scatter fills the valid positions.
inline int moe_find_remaining_4x8(
        __global const int * remaining,
        uint n_experts,
        int want,
        int exclude0,
        int exclude1,
        int exclude2) {
    for (int e = 0; e < (int)n_experts; ++e) {
        if (e == exclude0 || e == exclude1 || e == exclude2) {
            continue;
        }
        if (remaining[e] == want) {
            return e;
        }
    }
    return -1;
}

inline int moe_map_take_4x8(
        __global int * group_map,
        __global const int * group_offset,
        __global int * mapped_groups,
        int expert,
        int take,
        int physical_group) {
    const int logical_base = group_offset[expert] + mapped_groups[expert];
    for (int i = 0; i < take; ++i) {
        group_map[logical_base + i] = physical_group + i;
    }
    mapped_groups[expert] += take;
    return physical_group + take;
}

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

    // This packing policy is intentionally specialized for 32 = 4 x 8.
    if (groups_per_tile != 4) {
        *total_tiles = 0;
        return;
    }

    int logical_groups = 0;
    for (int e = 0; e < (int)n_experts; ++e) {
        const int count = hist[e];
        const int groups = (count + expert_slot_size - 1) / expert_slot_size;

        group_offset[e] = logical_groups;
        logical_groups += groups;

        // During scan hist[] is reused as the number of groups not yet packed,
        // and slot_counter[] as the number already mapped for this expert.
        hist[e] = groups;
        slot_counter[e] = 0;
    }

    int physical_group = 0;
    // 1) Four groups from one expert are ideal: no padding and one weight load.
    for (int e = 0; e < (int)n_experts; ++e) {
        while (hist[e] >= 4) {
            physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e, 4, physical_group);
            hist[e] -= 4;
        }
    }

    // 2) Pack whole remainders into exact 4-group tiles. The order favors the
    //    requested low-weight-load patterns: 3+1 and 2+2 before 3/4-expert tiles.
    while (1) {
        const int e3 = moe_find_remaining_4x8(hist, n_experts, 3, -1, -1, -1);
        const int e1 = moe_find_remaining_4x8(hist, n_experts, 1, e3, -1, -1);
        if (e3 < 0 || e1 < 0) {
            break;
        }
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e3, 3, physical_group);
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e1, 1, physical_group);
        hist[e3] = 0;
        hist[e1] = 0;
    }

    while (1) {
        const int e20 = moe_find_remaining_4x8(hist, n_experts, 2, -1, -1, -1);
        const int e21 = moe_find_remaining_4x8(hist, n_experts, 2, e20, -1, -1);
        if (e20 < 0 || e21 < 0) {
            break;
        }
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e20, 2, physical_group);
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e21, 2, physical_group);
        hist[e20] = 0;
        hist[e21] = 0;
    }

    while (1) {
        const int e2 = moe_find_remaining_4x8(hist, n_experts, 2, -1, -1, -1);
        const int e10 = moe_find_remaining_4x8(hist, n_experts, 1, e2, -1, -1);
        const int e11 = moe_find_remaining_4x8(hist, n_experts, 1, e2, e10, -1);
        if (e2 < 0 || e10 < 0 || e11 < 0) {
            break;
        }
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e2, 2, physical_group);
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e10, 1, physical_group);
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e11, 1, physical_group);
        hist[e2] = 0;
        hist[e10] = 0;
        hist[e11] = 0;
    }

    while (1) {
        const int e0 = moe_find_remaining_4x8(hist, n_experts, 1, -1, -1, -1);
        const int e1 = moe_find_remaining_4x8(hist, n_experts, 1, e0, -1, -1);
        const int e2 = moe_find_remaining_4x8(hist, n_experts, 1, e0, e1, -1);
        const int e3 = moe_find_remaining_4x8(hist, n_experts, 1, e0, e1, e2);
        if (e0 < 0 || e1 < 0 || e2 < 0 || e3 < 0) {
            break;
        }
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e0, 1, physical_group);
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e1, 1, physical_group);
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e2, 1, physical_group);
        physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, e3, 1, physical_group);
        hist[e0] = 0;
        hist[e1] = 0;
        hist[e2] = 0;
        hist[e3] = 0;
    }

    // 3) If exact whole-remainder combinations are exhausted, fill the minimum
    //    number of remaining tiles. Split an expert remainder only when needed
    //    to fill a non-final tile. Within every tile each expert stays contiguous.
    int remaining_groups = 0;
    for (int e = 0; e < (int)n_experts; ++e) {
        remaining_groups += hist[e];
    }
    while (remaining_groups > 0) {
        int capacity = min(4, remaining_groups);

        while (capacity > 0 && remaining_groups > 0) {
            int expert = moe_find_remaining_4x8(hist, n_experts, capacity, -1, -1, -1);
            int take = capacity;

            if (expert < 0) {
                // Prefer consuming a whole remainder that fits, largest first.
                int best_groups = 0;
                for (int e = 0; e < (int)n_experts; ++e) {
                    const int groups = hist[e];
                    if (groups > best_groups && groups <= capacity) {
                        best_groups = groups;
                        expert = e;
                    }
                }
                if (expert >= 0) {
                    take = hist[expert];
                } else {
                    // Every remaining expert is larger than the free space.
                    // Splitting here is required to achieve ceil(groups/4) tiles.
                    int largest_groups = 0;
                    for (int e = 0; e < (int)n_experts; ++e) {
                        if (hist[e] > largest_groups) {
                            largest_groups = hist[e];
                            expert = e;
                        }
                    }
                    take = capacity;
                }
            }
            physical_group = moe_map_take_4x8(group_map, group_offset, slot_counter, expert, take, physical_group);
            hist[expert] -= take;
            capacity -= take;
            remaining_groups -= take;
        }

        // Only the final partial tile should normally need this padding.
        if ((physical_group & 3) != 0) {
            physical_group += 4 - (physical_group & 3);
        }
    }

    *total_tiles = physical_group / groups_per_tile;
    // Derive statistics from the final mapping. A run is a maximal sequence of
    // groups belonging to the same expert that is contiguous inside one
    // physical tile. This is exactly the unit for which the Q4_0 4x8 GEMM can
    // reuse one weight load/dequantization.
    int expert_runs = 0;
    for (int e = 0; e < (int)n_experts; ++e) {
        const int mapped = slot_counter[e];
        if (mapped <= 0) {
            continue;
        }

        int prev = group_map[group_offset[e]];
        ++expert_runs;
        for (int i = 1; i < mapped; ++i) {
            const int cur = group_map[group_offset[e] + i];
            const bool contiguous = (cur == prev + 1) &&
                                    ((cur / groups_per_tile) == (prev / groups_per_tile));
            if (!contiguous) {
                ++expert_runs;
            }
            prev = cur;
        }
    }

    // total_tiles[0] remains ABI-compatible with kernel_moe_fill and all GEMM
    // kernels. The two extra words are diagnostic data for 4x8 only.
    total_tiles[0] = physical_group / groups_per_tile;
    total_tiles[1] = logical_groups;
    total_tiles[2] = expert_runs;

    for (int e = 0; e < (int)n_experts; ++e) {
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