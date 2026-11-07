; ModuleID = 'f32_matmul_lds'
target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-v32:32:32-v4:16:16-v2:16:16-v1:8:8-v512:512:512-v1024:1024:1024-v2048:2048:2048"

; ==============================================================================
; Native Float32 2D LDS-Tiled Matrix Multiplication Kernel (gfx90c)
; Computes Out[batch_size, c_out] = X[batch_size, c_in] @ W[c_out, c_in]^T
; Used for LSA 512-token block PCA projection ([512, 1024] @ [64, 1024]^T)
; and LSA cosine Gram matrix ([512, 64] @ [512, 64]^T / [128, 64] @ [128, 64]^T).
; ==============================================================================

@lds_X = local_unnamed_addr addrspace(3) global [512 x float] undef, align 16
@lds_W = local_unnamed_addr addrspace(3) global [576 x float] undef, align 16

define amdgpu_kernel void @f32_matmul(
    ptr addrspace(1) %W_ptr,
    ptr addrspace(1) %scales_unused,
    ptr addrspace(1) %x_ptr,
    ptr addrspace(1) %out_ptr,
    ptr addrspace(1) %completion_flag,
    i32 %c_in,
    i32 %c_out,
    i32 %grid_stride,
    i32 %num_workgroups,
    i32 %batch_size,
    i32 %out_stride
) {
entry:
    %tid = call i32 @llvm.amdgcn.workitem.id.x()
    %wg_id = call i32 @llvm.amdgcn.workgroup.id.x()

    %tok_row = udiv i32 %tid, 16
    %weight_row = urem i32 %tid, 16

    %c_in_blocks = udiv i32 %c_in, 32

    %c_out_plus_15 = add i32 %c_out, 15
    %num_row_tiles = udiv i32 %c_out_plus_15, 16
    %row_tile_idx = urem i32 %wg_id, %num_row_tiles
    %token_tile_idx = udiv i32 %wg_id, %num_row_tiles

    %row_base = mul i32 %row_tile_idx, 16
    %token_base = mul i32 %token_tile_idx, 16

    %global_row = add i32 %row_base, %weight_row
    %global_token = add i32 %token_base, %tok_row

    %wg_oob_row = icmp uge i32 %row_base, %c_out
    %wg_oob_token = icmp uge i32 %token_base, %batch_size
    %wg_oob = or i1 %wg_oob_row, %wg_oob_token
    br i1 %wg_oob, label %exit, label %loop_init

loop_init:
    %w_coop_global_row = add i32 %row_base, %tok_row
    %is_w_row_in_bounds = icmp ult i32 %w_coop_global_row, %c_out
    %is_token_in_bounds = icmp ult i32 %global_token, %batch_size
    %lds_row_offset = mul i32 %tok_row, 32
    %lds_w_row_offset = mul i32 %tok_row, 36
    %weight_row_plus_16 = add i32 %weight_row, 16
    %lds_idx_0 = add i32 %lds_row_offset, %weight_row
    %lds_idx_1 = add i32 %lds_row_offset, %weight_row_plus_16
    %lds_w_idx_0 = add i32 %lds_w_row_offset, %weight_row
    %lds_w_idx_1 = add i32 %lds_w_row_offset, %weight_row_plus_16
    br label %outer_loop

outer_loop:
    %b = phi i32 [ 0, %loop_init ], [ %b_next, %outer_loop_end ]
    %acc = phi float [ 0.0, %loop_init ], [ %acc_next, %outer_loop_end ]

    %cmp_b = icmp ult i32 %b, %c_in_blocks
    br i1 %cmp_b, label %cooperative_load, label %outer_loop_exit

cooperative_load:
    %b_mul_32 = mul i32 %b, 32
    %col_0 = add i32 %b_mul_32, %weight_row
    %col_1 = add i32 %b_mul_32, %weight_row_plus_16

    ; Cooperative load X[global_token, col_0] and X[global_token, col_1]
    %x_row_base = mul i32 %global_token, %c_in
    %x_idx_0 = add i32 %x_row_base, %col_0
    %x_idx_1 = add i32 %x_row_base, %col_1
    %x_idx_0_safe = select i1 %is_token_in_bounds, i32 %x_idx_0, i32 0
    %x_idx_1_safe = select i1 %is_token_in_bounds, i32 %x_idx_1, i32 0
    %x_gptr_0 = getelementptr inbounds float, ptr addrspace(1) %x_ptr, i32 %x_idx_0_safe
    %x_gptr_1 = getelementptr inbounds float, ptr addrspace(1) %x_ptr, i32 %x_idx_1_safe
    %vx0_raw = load float, ptr addrspace(1) %x_gptr_0, align 4
    %vx1_raw = load float, ptr addrspace(1) %x_gptr_1, align 4
    %vx0 = select i1 %is_token_in_bounds, float %vx0_raw, float 0.0
    %vx1 = select i1 %is_token_in_bounds, float %vx1_raw, float 0.0

    %lds_x_ptr_0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_idx_0
    %lds_x_ptr_1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_idx_1
    store float %vx0, ptr addrspace(3) %lds_x_ptr_0, align 4
    store float %vx1, ptr addrspace(3) %lds_x_ptr_1, align 4

    ; Cooperative load W[w_coop_global_row, col_0] and W[w_coop_global_row, col_1]
    %w_row_base = mul i32 %w_coop_global_row, %c_in
    %w_idx_0 = add i32 %w_row_base, %col_0
    %w_idx_1 = add i32 %w_row_base, %col_1
    %w_idx_0_safe = select i1 %is_w_row_in_bounds, i32 %w_idx_0, i32 0
    %w_idx_1_safe = select i1 %is_w_row_in_bounds, i32 %w_idx_1, i32 0
    %w_gptr_0 = getelementptr inbounds float, ptr addrspace(1) %W_ptr, i32 %w_idx_0_safe
    %w_gptr_1 = getelementptr inbounds float, ptr addrspace(1) %W_ptr, i32 %w_idx_1_safe
    %vw0_raw = load float, ptr addrspace(1) %w_gptr_0, align 4
    %vw1_raw = load float, ptr addrspace(1) %w_gptr_1, align 4
    %vw0 = select i1 %is_w_row_in_bounds, float %vw0_raw, float 0.0
    %vw1 = select i1 %is_w_row_in_bounds, float %vw1_raw, float 0.0

    %lds_w_ptr_0 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_idx_0
    %lds_w_ptr_1 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_idx_1
    store float %vw0, ptr addrspace(3) %lds_w_ptr_0, align 4
    store float %vw1, ptr addrspace(3) %lds_w_ptr_1, align 4

    call void @llvm.amdgcn.s.barrier()

    ; Math block calculation (32 floats, stride 36 avoids LDS bank conflicts)
    %lds_w_math_base = mul i32 %weight_row, 36
    %lw0 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_math_base
    %wv0 = load <4 x float>, ptr addrspace(3) %lw0, align 16
    %lwi1 = add i32 %lds_w_math_base, 4
    %lw1 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lwi1
    %wv1 = load <4 x float>, ptr addrspace(3) %lw1, align 16
    %lwi2 = add i32 %lds_w_math_base, 8
    %lw2 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lwi2
    %wv2 = load <4 x float>, ptr addrspace(3) %lw2, align 16
    %lwi3 = add i32 %lds_w_math_base, 12
    %lw3 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lwi3
    %wv3 = load <4 x float>, ptr addrspace(3) %lw3, align 16
    %lwi4 = add i32 %lds_w_math_base, 16
    %lw4 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lwi4
    %wv4 = load <4 x float>, ptr addrspace(3) %lw4, align 16
    %lwi5 = add i32 %lds_w_math_base, 20
    %lw5 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lwi5
    %wv5 = load <4 x float>, ptr addrspace(3) %lw5, align 16
    %lwi6 = add i32 %lds_w_math_base, 24
    %lw6 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lwi6
    %wv6 = load <4 x float>, ptr addrspace(3) %lw6, align 16
    %lwi7 = add i32 %lds_w_math_base, 28
    %lw7 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lwi7
    %wv7 = load <4 x float>, ptr addrspace(3) %lw7, align 16

    %lx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_row_offset
    %xv0 = load <4 x float>, ptr addrspace(3) %lx0, align 16
    %lxi1 = add i32 %lds_row_offset, 4
    %lx1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lxi1
    %xv1 = load <4 x float>, ptr addrspace(3) %lx1, align 16
    %lxi2 = add i32 %lds_row_offset, 8
    %lx2 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lxi2
    %xv2 = load <4 x float>, ptr addrspace(3) %lx2, align 16
    %lxi3 = add i32 %lds_row_offset, 12
    %lx3 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lxi3
    %xv3 = load <4 x float>, ptr addrspace(3) %lx3, align 16
    %lxi4 = add i32 %lds_row_offset, 16
    %lx4 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lxi4
    %xv4 = load <4 x float>, ptr addrspace(3) %lx4, align 16
    %lxi5 = add i32 %lds_row_offset, 20
    %lx5 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lxi5
    %xv5 = load <4 x float>, ptr addrspace(3) %lx5, align 16
    %lxi6 = add i32 %lds_row_offset, 24
    %lx6 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lxi6
    %xv6 = load <4 x float>, ptr addrspace(3) %lx6, align 16
    %lxi7 = add i32 %lds_row_offset, 28
    %lx7 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lxi7
    %xv7 = load <4 x float>, ptr addrspace(3) %lx7, align 16

    %p0 = fmul <4 x float> %wv0, %xv0
    %p1 = fmul <4 x float> %wv1, %xv1
    %p2 = fmul <4 x float> %wv2, %xv2
    %p3 = fmul <4 x float> %wv3, %xv3
    %p4 = fmul <4 x float> %wv4, %xv4
    %p5 = fmul <4 x float> %wv5, %xv5
    %p6 = fmul <4 x float> %wv6, %xv6
    %p7 = fmul <4 x float> %wv7, %xv7

    %s01 = fadd <4 x float> %p0, %p1
    %s23 = fadd <4 x float> %p2, %p3
    %s45 = fadd <4 x float> %p4, %p5
    %s67 = fadd <4 x float> %p6, %p7
    %s03 = fadd <4 x float> %s01, %s23
    %s47 = fadd <4 x float> %s45, %s67
    %sv = fadd <4 x float> %s03, %s47

    %e0 = extractelement <4 x float> %sv, i32 0
    %e1 = extractelement <4 x float> %sv, i32 1
    %e2 = extractelement <4 x float> %sv, i32 2
    %e3 = extractelement <4 x float> %sv, i32 3
    %e01 = fadd float %e0, %e1
    %e23 = fadd float %e2, %e3
    %blk_sum = fadd float %e01, %e23
    %acc_curr = fadd float %acc, %blk_sum
    br label %outer_loop_end

outer_loop_end:
    %acc_next = phi float [ %acc_curr, %cooperative_load ]
    %b_next = add nuw nsw i32 %b, 1
    call void @llvm.amdgcn.s.barrier()
    br label %outer_loop

outer_loop_exit:
    %has_stride = icmp ugt i32 %out_stride, 0
    %row_stride = select i1 %has_stride, i32 %out_stride, i32 %c_out
    %out_idx_0 = mul i32 %global_token, %row_stride
    %out_idx = add i32 %out_idx_0, %global_row

    %is_thread_in_bounds_row = icmp ult i32 %global_row, %c_out
    %is_thread_in_bounds_token = icmp ult i32 %global_token, %batch_size
    %is_thread_in_bounds = and i1 %is_thread_in_bounds_row, %is_thread_in_bounds_token
    br i1 %is_thread_in_bounds, label %write_output, label %exit_sync

write_output:
    %out_gptr = getelementptr inbounds float, ptr addrspace(1) %out_ptr, i32 %out_idx
    store float %acc, ptr addrspace(1) %out_gptr, align 4
    br label %exit_sync

exit_sync:
    fence syncscope("workgroup") release
    call void @llvm.amdgcn.s.barrier()
    %cmp_t0 = icmp eq i32 %tid, 0
    br i1 %cmp_t0, label %atomic_inc, label %exit

atomic_inc:
    fence syncscope("agent") acquire
    %counter_ptr = getelementptr inbounds i32, ptr addrspace(1) %completion_flag, i32 1
    %old_val = atomicrmw add ptr addrspace(1) %counter_ptr, i32 1 seq_cst
    %last_wg_val = sub i32 %num_workgroups, 1
    %is_last_wg = icmp eq i32 %old_val, %last_wg_val
    br i1 %is_last_wg, label %write_completion, label %exit

write_completion:
    store i32 1, ptr addrspace(1) %completion_flag, align 4
    br label %exit

exit:
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    call void asm sideeffect "s_nop 0", ""()
    ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nounwind readnone speculatable willreturn }
attributes #1 = { convergent nounwind willreturn }
