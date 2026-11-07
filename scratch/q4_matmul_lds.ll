; ModuleID = 'q4_matmul_lds'
target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-v32:32:32-v4:16:16-v2:16:16-v1:8:8-v512:512:512-v1024:1024:1024-v2048:2048:2048"

; ==============================================================================
; Native Q4_0 2D LDS-Tiled Cooperative Matrix Multiplication Kernel (gfx90c)
; ==============================================================================
; Workgroup: 256 threads mapped as [16 tokens (tok_row) x 16 rows (weight_row)]
; Tile per block iteration (b = 0 .. c_in/32 - 1):
;   - @lds_X: [512 x float] (16 tokens x 32 channels = 2048 bytes)
;   - @lds_W: [512 x float] (16 rows   x 32 channels = 2048 bytes, unpacked in LDS!)
; All 256 threads cooperatively unpack 2 Q4 nibbles each into @lds_W before
; s_barrier, eliminating 93.75% of bitfield extract / int-to-float instructions!
; ==============================================================================

@lds_X = local_unnamed_addr addrspace(3) global [512 x float] undef, align 16
@lds_W = local_unnamed_addr addrspace(3) global [576 x float] undef, align 16

define amdgpu_kernel void @q4_matmul(
    ptr addrspace(1) %W_packed,
    ptr addrspace(1) %scales,
    ptr addrspace(1) %x,
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

    ; 2D thread coordinates inside [16 x 16] workgroup
    %tok_row = udiv i32 %tid, 16
    %weight_row = urem i32 %tid, 16

    %c_in_blocks = udiv i32 %c_in, 32
    %c_in_i32s = udiv i32 %c_in, 8

    ; Decode 1D workgroup ID to 2D tile coordinates (ceiling division for c_out)
    %c_out_plus_15 = add i32 %c_out, 15
    %num_row_tiles = udiv i32 %c_out_plus_15, 16
    %row_tile_idx = urem i32 %wg_id, %num_row_tiles
    %token_tile_idx = udiv i32 %wg_id, %num_row_tiles

    %row_base = mul i32 %row_tile_idx, 16
    %token_base = mul i32 %token_tile_idx, 16

    %global_row = add i32 %row_base, %weight_row
    %global_token = add i32 %token_base, %tok_row

    ; Out-of-bounds workgroup early exit
    %wg_oob_row = icmp uge i32 %row_base, %c_out
    %wg_oob_token = icmp uge i32 %token_base, %batch_size
    %wg_oob = or i1 %wg_oob_row, %wg_oob_token
    br i1 %wg_oob, label %exit, label %dispatch_mode

dispatch_mode:
    %is_gemv = icmp eq i32 %batch_size, 1
    br i1 %is_gemv, label %gemv_init, label %loop_init

gemv_init:
    ; In batch_size == 1 mode, tok_row (0..15) is the row index within the 16-row tile,
    ; and weight_row (0..15) is the K-block lane index!
    ; All 16 lanes in each row process 16 blocks of 32 weights in parallel per iteration!
    %gemv_row = add i32 %row_base, %tok_row
    %gemv_row_in_bounds = icmp ult i32 %gemv_row, %c_out
    %gemv_row_safe = select i1 %gemv_row_in_bounds, i32 %gemv_row, i32 0
    %gemv_scale_base = mul i32 %gemv_row_safe, %c_in_blocks
    %gemv_w_i32_base = mul i32 %gemv_row_safe, %c_in_i32s
    br label %gemv_loop

gemv_loop:
    %gb = phi i32 [ %weight_row, %gemv_init ], [ %gb_next, %gemv_loop_body ]
    %gacc = phi float [ 0.0, %gemv_init ], [ %gacc_next, %gemv_loop_body ]
    %g_cmp = icmp ult i32 %gb, %c_in_blocks
    br i1 %g_cmp, label %gemv_loop_body, label %gemv_reduce

gemv_loop_body:
    ; Load scale for (gemv_row, gb)
    %g_scale_idx = add i32 %gemv_scale_base, %gb
    %g_scale_ptr = getelementptr inbounds float, ptr addrspace(1) %scales, i32 %g_scale_idx
    %g_scale_raw = load float, ptr addrspace(1) %g_scale_ptr, align 4
    %g_scale = select i1 %gemv_row_in_bounds, float %g_scale_raw, float 0.0

    ; Load 16 bytes (4 x i32) of packed Q4 weights for (gemv_row, gb)
    %gb_mul_4 = shl i32 %gb, 2
    %g_w_off = add i32 %gemv_w_i32_base, %gb_mul_4
    %g_w_ptr = getelementptr inbounds i32, ptr addrspace(1) %W_packed, i32 %g_w_off
    %g_w_vec = load <4 x i32>, ptr addrspace(1) %g_w_ptr, align 16
    %gw0 = extractelement <4 x i32> %g_w_vec, i32 0
    %gw1 = extractelement <4 x i32> %g_w_vec, i32 1
    %gw2 = extractelement <4 x i32> %g_w_vec, i32 2
    %gw3 = extractelement <4 x i32> %g_w_vec, i32 3

    ; Load 32 floats of x[gb * 32 .. gb * 32 + 31]
    %gb_mul_32 = shl i32 %gb, 5
    %gx_ptr_0 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gb_mul_32
    %gx_v0 = load <4 x float>, ptr addrspace(1) %gx_ptr_0, align 16
    %gx_idx_1 = add i32 %gb_mul_32, 4
    %gx_ptr_1 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gx_idx_1
    %gx_v1 = load <4 x float>, ptr addrspace(1) %gx_ptr_1, align 16
    %gx_idx_2 = add i32 %gb_mul_32, 8
    %gx_ptr_2 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gx_idx_2
    %gx_v2 = load <4 x float>, ptr addrspace(1) %gx_ptr_2, align 16
    %gx_idx_3 = add i32 %gb_mul_32, 12
    %gx_ptr_3 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gx_idx_3
    %gx_v3 = load <4 x float>, ptr addrspace(1) %gx_ptr_3, align 16
    %gx_idx_4 = add i32 %gb_mul_32, 16
    %gx_ptr_4 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gx_idx_4
    %gx_v4 = load <4 x float>, ptr addrspace(1) %gx_ptr_4, align 16
    %gx_idx_5 = add i32 %gb_mul_32, 20
    %gx_ptr_5 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gx_idx_5
    %gx_v5 = load <4 x float>, ptr addrspace(1) %gx_ptr_5, align 16
    %gx_idx_6 = add i32 %gb_mul_32, 24
    %gx_ptr_6 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gx_idx_6
    %gx_v6 = load <4 x float>, ptr addrspace(1) %gx_ptr_6, align 16
    %gx_idx_7 = add i32 %gb_mul_32, 28
    %gx_ptr_7 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %gx_idx_7
    %gx_v7 = load <4 x float>, ptr addrspace(1) %gx_ptr_7, align 16

    ; Word 0 (weights 0..7)
    %gn0_u = and i32 %gw0, 15
    %gn0_s = sub nsw i32 %gn0_u, 8
    %gwf0 = sitofp i32 %gn0_s to float
    %gn1_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw0, i32 4, i32 4)
    %gn1_s = sub nsw i32 %gn1_u, 8
    %gwf1 = sitofp i32 %gn1_s to float
    %gn2_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw0, i32 8, i32 4)
    %gn2_s = sub nsw i32 %gn2_u, 8
    %gwf2 = sitofp i32 %gn2_s to float
    %gn3_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw0, i32 12, i32 4)
    %gn3_s = sub nsw i32 %gn3_u, 8
    %gwf3 = sitofp i32 %gn3_s to float
    %gn4_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw0, i32 16, i32 4)
    %gn4_s = sub nsw i32 %gn4_u, 8
    %gwf4 = sitofp i32 %gn4_s to float
    %gn5_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw0, i32 20, i32 4)
    %gn5_s = sub nsw i32 %gn5_u, 8
    %gwf5 = sitofp i32 %gn5_s to float
    %gn6_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw0, i32 24, i32 4)
    %gn6_s = sub nsw i32 %gn6_u, 8
    %gwf6 = sitofp i32 %gn6_s to float
    %gn7_u = lshr i32 %gw0, 28
    %gn7_s = sub nsw i32 %gn7_u, 8
    %gwf7 = sitofp i32 %gn7_s to float

    ; Word 1 (weights 8..15)
    %gn8_u = and i32 %gw1, 15
    %gn8_s = sub nsw i32 %gn8_u, 8
    %gwf8 = sitofp i32 %gn8_s to float
    %gn9_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw1, i32 4, i32 4)
    %gn9_s = sub nsw i32 %gn9_u, 8
    %gwf9 = sitofp i32 %gn9_s to float
    %gn10_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw1, i32 8, i32 4)
    %gn10_s = sub nsw i32 %gn10_u, 8
    %gwf10 = sitofp i32 %gn10_s to float
    %gn11_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw1, i32 12, i32 4)
    %gn11_s = sub nsw i32 %gn11_u, 8
    %gwf11 = sitofp i32 %gn11_s to float
    %gn12_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw1, i32 16, i32 4)
    %gn12_s = sub nsw i32 %gn12_u, 8
    %gwf12 = sitofp i32 %gn12_s to float
    %gn13_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw1, i32 20, i32 4)
    %gn13_s = sub nsw i32 %gn13_u, 8
    %gwf13 = sitofp i32 %gn13_s to float
    %gn14_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw1, i32 24, i32 4)
    %gn14_s = sub nsw i32 %gn14_u, 8
    %gwf14 = sitofp i32 %gn14_s to float
    %gn15_u = lshr i32 %gw1, 28
    %gn15_s = sub nsw i32 %gn15_u, 8
    %gwf15 = sitofp i32 %gn15_s to float

    ; Word 2 (weights 16..23)
    %gn16_u = and i32 %gw2, 15
    %gn16_s = sub nsw i32 %gn16_u, 8
    %gwf16 = sitofp i32 %gn16_s to float
    %gn17_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw2, i32 4, i32 4)
    %gn17_s = sub nsw i32 %gn17_u, 8
    %gwf17 = sitofp i32 %gn17_s to float
    %gn18_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw2, i32 8, i32 4)
    %gn18_s = sub nsw i32 %gn18_u, 8
    %gwf18 = sitofp i32 %gn18_s to float
    %gn19_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw2, i32 12, i32 4)
    %gn19_s = sub nsw i32 %gn19_u, 8
    %gwf19 = sitofp i32 %gn19_s to float
    %gn20_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw2, i32 16, i32 4)
    %gn20_s = sub nsw i32 %gn20_u, 8
    %gwf20 = sitofp i32 %gn20_s to float
    %gn21_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw2, i32 20, i32 4)
    %gn21_s = sub nsw i32 %gn21_u, 8
    %gwf21 = sitofp i32 %gn21_s to float
    %gn22_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw2, i32 24, i32 4)
    %gn22_s = sub nsw i32 %gn22_u, 8
    %gwf22 = sitofp i32 %gn22_s to float
    %gn23_u = lshr i32 %gw2, 28
    %gn23_s = sub nsw i32 %gn23_u, 8
    %gwf23 = sitofp i32 %gn23_s to float

    ; Word 3 (weights 24..31)
    %gn24_u = and i32 %gw3, 15
    %gn24_s = sub nsw i32 %gn24_u, 8
    %gwf24 = sitofp i32 %gn24_s to float
    %gn25_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw3, i32 4, i32 4)
    %gn25_s = sub nsw i32 %gn25_u, 8
    %gwf25 = sitofp i32 %gn25_s to float
    %gn26_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw3, i32 8, i32 4)
    %gn26_s = sub nsw i32 %gn26_u, 8
    %gwf26 = sitofp i32 %gn26_s to float
    %gn27_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw3, i32 12, i32 4)
    %gn27_s = sub nsw i32 %gn27_u, 8
    %gwf27 = sitofp i32 %gn27_s to float
    %gn28_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw3, i32 16, i32 4)
    %gn28_s = sub nsw i32 %gn28_u, 8
    %gwf28 = sitofp i32 %gn28_s to float
    %gn29_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw3, i32 20, i32 4)
    %gn29_s = sub nsw i32 %gn29_u, 8
    %gwf29 = sitofp i32 %gn29_s to float
    %gn30_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %gw3, i32 24, i32 4)
    %gn30_s = sub nsw i32 %gn30_u, 8
    %gwf30 = sitofp i32 %gn30_s to float
    %gn31_u = lshr i32 %gw3, 28
    %gn31_s = sub nsw i32 %gn31_u, 8
    %gwf31 = sitofp i32 %gn31_s to float

    %gx0 = extractelement <4 x float> %gx_v0, i32 0
    %gx1 = extractelement <4 x float> %gx_v0, i32 1
    %gx2 = extractelement <4 x float> %gx_v0, i32 2
    %gx3 = extractelement <4 x float> %gx_v0, i32 3
    %gx4 = extractelement <4 x float> %gx_v1, i32 0
    %gx5 = extractelement <4 x float> %gx_v1, i32 1
    %gx6 = extractelement <4 x float> %gx_v1, i32 2
    %gx7 = extractelement <4 x float> %gx_v1, i32 3
    %gx8 = extractelement <4 x float> %gx_v2, i32 0
    %gx9 = extractelement <4 x float> %gx_v2, i32 1
    %gx10 = extractelement <4 x float> %gx_v2, i32 2
    %gx11 = extractelement <4 x float> %gx_v2, i32 3
    %gx12 = extractelement <4 x float> %gx_v3, i32 0
    %gx13 = extractelement <4 x float> %gx_v3, i32 1
    %gx14 = extractelement <4 x float> %gx_v3, i32 2
    %gx15 = extractelement <4 x float> %gx_v3, i32 3
    %gx16 = extractelement <4 x float> %gx_v4, i32 0
    %gx17 = extractelement <4 x float> %gx_v4, i32 1
    %gx18 = extractelement <4 x float> %gx_v4, i32 2
    %gx19 = extractelement <4 x float> %gx_v4, i32 3
    %gx20 = extractelement <4 x float> %gx_v5, i32 0
    %gx21 = extractelement <4 x float> %gx_v5, i32 1
    %gx22 = extractelement <4 x float> %gx_v5, i32 2
    %gx23 = extractelement <4 x float> %gx_v5, i32 3
    %gx24 = extractelement <4 x float> %gx_v6, i32 0
    %gx25 = extractelement <4 x float> %gx_v6, i32 1
    %gx26 = extractelement <4 x float> %gx_v6, i32 2
    %gx27 = extractelement <4 x float> %gx_v6, i32 3
    %gx28 = extractelement <4 x float> %gx_v7, i32 0
    %gx29 = extractelement <4 x float> %gx_v7, i32 1
    %gx30 = extractelement <4 x float> %gx_v7, i32 2
    %gx31 = extractelement <4 x float> %gx_v7, i32 3

    %ga0 = fmul float %gwf0, %gx0
    %gb0 = fmul float %gwf1, %gx1
    %gc0 = fmul float %gwf2, %gx2
    %gd0 = fmul float %gwf3, %gx3
    %ga1 = call float @llvm.fma.f32(float %gwf4, float %gx4, float %ga0)
    %gb1 = call float @llvm.fma.f32(float %gwf5, float %gx5, float %gb0)
    %gc1 = call float @llvm.fma.f32(float %gwf6, float %gx6, float %gc0)
    %gd1 = call float @llvm.fma.f32(float %gwf7, float %gx7, float %gd0)
    %ga2 = call float @llvm.fma.f32(float %gwf8, float %gx8, float %ga1)
    %gb2 = call float @llvm.fma.f32(float %gwf9, float %gx9, float %gb1)
    %gc2 = call float @llvm.fma.f32(float %gwf10, float %gx10, float %gc1)
    %gd2 = call float @llvm.fma.f32(float %gwf11, float %gx11, float %gd1)
    %ga3 = call float @llvm.fma.f32(float %gwf12, float %gx12, float %ga2)
    %gb3 = call float @llvm.fma.f32(float %gwf13, float %gx13, float %gb2)
    %gc3 = call float @llvm.fma.f32(float %gwf14, float %gx14, float %gc2)
    %gd3 = call float @llvm.fma.f32(float %gwf15, float %gx15, float %gd2)
    %ga4 = call float @llvm.fma.f32(float %gwf16, float %gx16, float %ga3)
    %gb4 = call float @llvm.fma.f32(float %gwf17, float %gx17, float %gb3)
    %gc4 = call float @llvm.fma.f32(float %gwf18, float %gx18, float %gc3)
    %gd4 = call float @llvm.fma.f32(float %gwf19, float %gx19, float %gd3)
    %ga5 = call float @llvm.fma.f32(float %gwf20, float %gx20, float %ga4)
    %gb5 = call float @llvm.fma.f32(float %gwf21, float %gx21, float %gb4)
    %gc5 = call float @llvm.fma.f32(float %gwf22, float %gx22, float %gc4)
    %gd5 = call float @llvm.fma.f32(float %gwf23, float %gx23, float %gd4)
    %ga6 = call float @llvm.fma.f32(float %gwf24, float %gx24, float %ga5)
    %gb6 = call float @llvm.fma.f32(float %gwf25, float %gx25, float %gb5)
    %gc6 = call float @llvm.fma.f32(float %gwf26, float %gx26, float %gc5)
    %gd6 = call float @llvm.fma.f32(float %gwf27, float %gx27, float %gd5)
    %ga7 = call float @llvm.fma.f32(float %gwf28, float %gx28, float %ga6)
    %gb7 = call float @llvm.fma.f32(float %gwf29, float %gx29, float %gb6)
    %gc7 = call float @llvm.fma.f32(float %gwf30, float %gx30, float %gc6)
    %gd7 = call float @llvm.fma.f32(float %gwf31, float %gx31, float %gd6)

    %gsum_ab = fadd float %ga7, %gb7
    %gsum_cd = fadd float %gc7, %gd7
    %gblock_sum = fadd float %gsum_ab, %gsum_cd
    %gacc_next = call float @llvm.fma.f32(float %gblock_sum, float %g_scale, float %gacc)
    %gb_next = add nuw nsw i32 %gb, 16
    br label %gemv_loop

gemv_reduce:
    %g_lds_ptr = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %tid
    store float %gacc, ptr addrspace(3) %g_lds_ptr, align 4
    call void @llvm.amdgcn.s.barrier()

    %is_lane0 = icmp eq i32 %weight_row, 0
    %do_gemv_store = and i1 %is_lane0, %gemv_row_in_bounds
    br i1 %do_gemv_store, label %gemv_write, label %exit_sync

gemv_write:
    %g_row_lds_base = shl i32 %tok_row, 4
    %gr_ptr_0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %g_row_lds_base
    %gr_v0 = load <4 x float>, ptr addrspace(3) %gr_ptr_0, align 16
    %gr_idx_1 = add i32 %g_row_lds_base, 4
    %gr_ptr_1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %gr_idx_1
    %gr_v1 = load <4 x float>, ptr addrspace(3) %gr_ptr_1, align 16
    %gr_idx_2 = add i32 %g_row_lds_base, 8
    %gr_ptr_2 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %gr_idx_2
    %gr_v2 = load <4 x float>, ptr addrspace(3) %gr_ptr_2, align 16
    %gr_idx_3 = add i32 %g_row_lds_base, 12
    %gr_ptr_3 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %gr_idx_3
    %gr_v3 = load <4 x float>, ptr addrspace(3) %gr_ptr_3, align 16

    %gr_v01 = fadd <4 x float> %gr_v0, %gr_v1
    %gr_v23 = fadd <4 x float> %gr_v2, %gr_v3
    %gr_vsum = fadd <4 x float> %gr_v01, %gr_v23
    %gr_s0 = extractelement <4 x float> %gr_vsum, i32 0
    %gr_s1 = extractelement <4 x float> %gr_vsum, i32 1
    %gr_s2 = extractelement <4 x float> %gr_vsum, i32 2
    %gr_s3 = extractelement <4 x float> %gr_vsum, i32 3
    %gr_s01 = fadd float %gr_s0, %gr_s1
    %gr_s23 = fadd float %gr_s2, %gr_s3
    %gr_total = fadd float %gr_s01, %gr_s23

    %g_out_ptr = getelementptr inbounds float, ptr addrspace(1) %out_ptr, i32 %gemv_row
    store float %gr_total, ptr addrspace(1) %g_out_ptr, align 4
    br label %exit_sync

loop_init:
    %r_offset_scales = mul i32 %global_row, %c_in_blocks
    %is_math_row_in_bounds = icmp ult i32 %global_row, %c_out
    %r_offset_scales_safe = select i1 %is_math_row_in_bounds, i32 %r_offset_scales, i32 0

    ; Precompute cooperative W loader invariants for this thread:
    ; Thread (tok_row, weight_row) loads byte `weight_row` (0..15) of row `row_base + tok_row`
    %w_coop_global_row = add i32 %row_base, %tok_row
    %is_w_row_in_bounds = icmp ult i32 %w_coop_global_row, %c_out
    %w_row_i32_base = mul i32 %w_coop_global_row, %c_in_i32s
    %w_col_i32 = udiv i32 %weight_row, 4
    %w_byte_in_i32 = urem i32 %weight_row, 4
    %w_bit_lo = mul i32 %w_byte_in_i32, 8
    %w_bit_hi = add i32 %w_bit_lo, 4
    %lds_w_store_base = mul i32 %tok_row, 36
    %w_elem_col_0 = mul i32 %weight_row, 2
    %w_elem_col_1 = add i32 %w_elem_col_0, 1
    %lds_w_store_idx_0 = add i32 %lds_w_store_base, %w_elem_col_0
    %lds_w_store_idx_1 = add i32 %lds_w_store_base, %w_elem_col_1

    br label %outer_loop

outer_loop:
    %b = phi i32 [ 0, %loop_init ], [ %b_next, %outer_loop_end ]
    %acc = phi float [ 0.0, %loop_init ], [ %acc_next, %outer_loop_end ]

    %cmp_b = icmp ult i32 %b, %c_in_blocks
    br i1 %cmp_b, label %cooperative_load, label %outer_loop_exit

cooperative_load:
    %b_mul_32 = mul i32 %b, 32

    ; --- 1. Cooperative load X (16 tokens x 32 floats = 512 floats) ---
    %x_load_idx_0 = add i32 %weight_row, %b_mul_32
    %x_global_idx_0_0 = mul i32 %global_token, %c_in
    %x_global_idx_0 = add i32 %x_global_idx_0_0, %x_load_idx_0

    %is_token_in_bounds = icmp ult i32 %global_token, %batch_size
    %x_global_idx_safe_0 = select i1 %is_token_in_bounds, i32 %x_global_idx_0, i32 0
    %x_global_ptr_0 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %x_global_idx_safe_0
    %val_x_0_raw = load float, ptr addrspace(1) %x_global_ptr_0, align 4
    %val_x_0 = select i1 %is_token_in_bounds, float %val_x_0_raw, float 0.0

    %lds_x_row_offset = mul i32 %tok_row, 32
    %lds_x_idx_0 = add i32 %lds_x_row_offset, %weight_row
    %lds_x_ptr_0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_idx_0
    store float %val_x_0, ptr addrspace(3) %lds_x_ptr_0, align 4
    br label %coop_x_second

coop_x_second:
    %weight_row_plus_16 = add i32 %weight_row, 16
    %x_load_idx_1 = add i32 %weight_row_plus_16, %b_mul_32
    %x_global_idx_1 = add i32 %x_global_idx_0_0, %x_load_idx_1
    %x_global_idx_safe_1 = select i1 %is_token_in_bounds, i32 %x_global_idx_1, i32 0
    %x_global_ptr_1 = getelementptr inbounds float, ptr addrspace(1) %x, i32 %x_global_idx_safe_1
    %val_x_1_raw = load float, ptr addrspace(1) %x_global_ptr_1, align 4
    %val_x_1 = select i1 %is_token_in_bounds, float %val_x_1_raw, float 0.0

    %lds_x_idx_1 = add i32 %lds_x_row_offset, %weight_row_plus_16
    %lds_x_ptr_1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_idx_1
    store float %val_x_1, ptr addrspace(3) %lds_x_ptr_1, align 4

    ; --- 2. Cooperative load & unpack W (16 rows x 32 floats = 512 floats) ---
    ; Each block has 4 i32s (16 bytes).
    %b_mul_4 = mul i32 %b, 4
    %w_i32_off_0 = add i32 %w_row_i32_base, %b_mul_4
    %w_i32_off = add i32 %w_i32_off_0, %w_col_i32
    %w_i32_off_safe = select i1 %is_w_row_in_bounds, i32 %w_i32_off, i32 0
    %w_global_ptr = getelementptr inbounds i32, ptr addrspace(1) %W_packed, i32 %w_i32_off_safe
    %val_w_raw = load i32, ptr addrspace(1) %w_global_ptr, align 4
    %val_w = select i1 %is_w_row_in_bounds, i32 %val_w_raw, i32 0

    %nib_lo_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %val_w, i32 %w_bit_lo, i32 4)
    %nib_lo_s = sub nsw i32 %nib_lo_u, 8
    %w_f_lo_raw = sitofp i32 %nib_lo_s to float
    %w_f_lo = select i1 %is_w_row_in_bounds, float %w_f_lo_raw, float 0.0

    %nib_hi_u = call i32 @llvm.amdgcn.ubfe.i32(i32 %val_w, i32 %w_bit_hi, i32 4)
    %nib_hi_s = sub nsw i32 %nib_hi_u, 8
    %w_f_hi_raw = sitofp i32 %nib_hi_s to float
    %w_f_hi = select i1 %is_w_row_in_bounds, float %w_f_hi_raw, float 0.0

    %lds_w_ptr_0 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_store_idx_0
    store float %w_f_lo, ptr addrspace(3) %lds_w_ptr_0, align 4
    %lds_w_ptr_1 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_store_idx_1
    store float %w_f_hi, ptr addrspace(3) %lds_w_ptr_1, align 4

    ; Synchronize workgroup after cooperative LDS writes
    call void @llvm.amdgcn.s.barrier()

    ; --- 3. Math Block Calculation ---
    %scale_idx = add i32 %r_offset_scales_safe, %b
    %scale_ptr = getelementptr inbounds float, ptr addrspace(1) %scales, i32 %scale_idx
    %scale_raw = load float, ptr addrspace(1) %scale_ptr, align 4
    %scale = select i1 %is_math_row_in_bounds, float %scale_raw, float 0.0

    ; Load 32 unpacked weights for `weight_row` from @lds_W (stride 36 avoids bank conflicts) using 8 x ds_read_b128
    %lds_w_math_base = mul i32 %weight_row, 36
    %lds_w_m_ptr_0 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_math_base
    %w_vec_0 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_0, align 16

    %lds_w_m_off_1 = add i32 %lds_w_math_base, 4
    %lds_w_m_ptr_1 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_m_off_1
    %w_vec_1 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_1, align 16

    %lds_w_m_off_2 = add i32 %lds_w_math_base, 8
    %lds_w_m_ptr_2 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_m_off_2
    %w_vec_2 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_2, align 16

    %lds_w_m_off_3 = add i32 %lds_w_math_base, 12
    %lds_w_m_ptr_3 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_m_off_3
    %w_vec_3 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_3, align 16

    %lds_w_m_off_4 = add i32 %lds_w_math_base, 16
    %lds_w_m_ptr_4 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_m_off_4
    %w_vec_4 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_4, align 16

    %lds_w_m_off_5 = add i32 %lds_w_math_base, 20
    %lds_w_m_ptr_5 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_m_off_5
    %w_vec_5 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_5, align 16

    %lds_w_m_off_6 = add i32 %lds_w_math_base, 24
    %lds_w_m_ptr_6 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_m_off_6
    %w_vec_6 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_6, align 16

    %lds_w_m_off_7 = add i32 %lds_w_math_base, 28
    %lds_w_m_ptr_7 = getelementptr inbounds [576 x float], ptr addrspace(3) @lds_W, i32 0, i32 %lds_w_m_off_7
    %w_vec_7 = load <4 x float>, ptr addrspace(3) %lds_w_m_ptr_7, align 16

    ; Load 32 activations for `tok_row` from @lds_X using 8 x ds_read_b128
    %lds_x_math_base = mul i32 %tok_row, 32
    %lds_x_m_ptr_0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_math_base
    %x_vec_0 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_0, align 16

    %lds_x_m_off_1 = add i32 %lds_x_math_base, 4
    %lds_x_m_ptr_1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_m_off_1
    %x_vec_1 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_1, align 16

    %lds_x_m_off_2 = add i32 %lds_x_math_base, 8
    %lds_x_m_ptr_2 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_m_off_2
    %x_vec_2 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_2, align 16

    %lds_x_m_off_3 = add i32 %lds_x_math_base, 12
    %lds_x_m_ptr_3 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_m_off_3
    %x_vec_3 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_3, align 16

    %lds_x_m_off_4 = add i32 %lds_x_math_base, 16
    %lds_x_m_ptr_4 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_m_off_4
    %x_vec_4 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_4, align 16

    %lds_x_m_off_5 = add i32 %lds_x_math_base, 20
    %lds_x_m_ptr_5 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_m_off_5
    %x_vec_5 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_5, align 16

    %lds_x_m_off_6 = add i32 %lds_x_math_base, 24
    %lds_x_m_ptr_6 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_m_off_6
    %x_vec_6 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_6, align 16

    %lds_x_m_off_7 = add i32 %lds_x_math_base, 28
    %lds_x_m_ptr_7 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds_X, i32 0, i32 %lds_x_m_off_7
    %x_vec_7 = load <4 x float>, ptr addrspace(3) %lds_x_m_ptr_7, align 16

    ; Extract weights
    %w0_0 = extractelement <4 x float> %w_vec_0, i32 0
    %w1_0 = extractelement <4 x float> %w_vec_0, i32 1
    %w2_0 = extractelement <4 x float> %w_vec_0, i32 2
    %w3_0 = extractelement <4 x float> %w_vec_0, i32 3

    %w0_1 = extractelement <4 x float> %w_vec_1, i32 0
    %w1_1 = extractelement <4 x float> %w_vec_1, i32 1
    %w2_1 = extractelement <4 x float> %w_vec_1, i32 2
    %w3_1 = extractelement <4 x float> %w_vec_1, i32 3

    %w0_2 = extractelement <4 x float> %w_vec_2, i32 0
    %w1_2 = extractelement <4 x float> %w_vec_2, i32 1
    %w2_2 = extractelement <4 x float> %w_vec_2, i32 2
    %w3_2 = extractelement <4 x float> %w_vec_2, i32 3

    %w0_3 = extractelement <4 x float> %w_vec_3, i32 0
    %w1_3 = extractelement <4 x float> %w_vec_3, i32 1
    %w2_3 = extractelement <4 x float> %w_vec_3, i32 2
    %w3_3 = extractelement <4 x float> %w_vec_3, i32 3

    %w0_4 = extractelement <4 x float> %w_vec_4, i32 0
    %w1_4 = extractelement <4 x float> %w_vec_4, i32 1
    %w2_4 = extractelement <4 x float> %w_vec_4, i32 2
    %w3_4 = extractelement <4 x float> %w_vec_4, i32 3

    %w0_5 = extractelement <4 x float> %w_vec_5, i32 0
    %w1_5 = extractelement <4 x float> %w_vec_5, i32 1
    %w2_5 = extractelement <4 x float> %w_vec_5, i32 2
    %w3_5 = extractelement <4 x float> %w_vec_5, i32 3

    %w0_6 = extractelement <4 x float> %w_vec_6, i32 0
    %w1_6 = extractelement <4 x float> %w_vec_6, i32 1
    %w2_6 = extractelement <4 x float> %w_vec_6, i32 2
    %w3_6 = extractelement <4 x float> %w_vec_6, i32 3

    %w0_7 = extractelement <4 x float> %w_vec_7, i32 0
    %w1_7 = extractelement <4 x float> %w_vec_7, i32 1
    %w2_7 = extractelement <4 x float> %w_vec_7, i32 2
    %w3_7 = extractelement <4 x float> %w_vec_7, i32 3

    ; Extract activations
    %x0_0 = extractelement <4 x float> %x_vec_0, i32 0
    %x1_0 = extractelement <4 x float> %x_vec_0, i32 1
    %x2_0 = extractelement <4 x float> %x_vec_0, i32 2
    %x3_0 = extractelement <4 x float> %x_vec_0, i32 3

    %x0_1 = extractelement <4 x float> %x_vec_1, i32 0
    %x1_1 = extractelement <4 x float> %x_vec_1, i32 1
    %x2_1 = extractelement <4 x float> %x_vec_1, i32 2
    %x3_1 = extractelement <4 x float> %x_vec_1, i32 3

    %x0_2 = extractelement <4 x float> %x_vec_2, i32 0
    %x1_2 = extractelement <4 x float> %x_vec_2, i32 1
    %x2_2 = extractelement <4 x float> %x_vec_2, i32 2
    %x3_2 = extractelement <4 x float> %x_vec_2, i32 3

    %x0_3 = extractelement <4 x float> %x_vec_3, i32 0
    %x1_3 = extractelement <4 x float> %x_vec_3, i32 1
    %x2_3 = extractelement <4 x float> %x_vec_3, i32 2
    %x3_3 = extractelement <4 x float> %x_vec_3, i32 3

    %x0_4 = extractelement <4 x float> %x_vec_4, i32 0
    %x1_4 = extractelement <4 x float> %x_vec_4, i32 1
    %x2_4 = extractelement <4 x float> %x_vec_4, i32 2
    %x3_4 = extractelement <4 x float> %x_vec_4, i32 3

    %x0_5 = extractelement <4 x float> %x_vec_5, i32 0
    %x1_5 = extractelement <4 x float> %x_vec_5, i32 1
    %x2_5 = extractelement <4 x float> %x_vec_5, i32 2
    %x3_5 = extractelement <4 x float> %x_vec_5, i32 3

    %x0_6 = extractelement <4 x float> %x_vec_6, i32 0
    %x1_6 = extractelement <4 x float> %x_vec_6, i32 1
    %x2_6 = extractelement <4 x float> %x_vec_6, i32 2
    %x3_6 = extractelement <4 x float> %x_vec_6, i32 3

    %x0_7 = extractelement <4 x float> %x_vec_7, i32 0
    %x1_7 = extractelement <4 x float> %x_vec_7, i32 1
    %x2_7 = extractelement <4 x float> %x_vec_7, i32 2
    %x3_7 = extractelement <4 x float> %x_vec_7, i32 3

    ; 4-lane FMA chains
    %ch_a_0 = call float @llvm.fma.f32(float %w0_0, float %x0_0, float 0.0)
    %ch_b_0 = call float @llvm.fma.f32(float %w1_0, float %x1_0, float 0.0)
    %ch_c_0 = call float @llvm.fma.f32(float %w2_0, float %x2_0, float 0.0)
    %ch_d_0 = call float @llvm.fma.f32(float %w3_0, float %x3_0, float 0.0)

    %ch_a_1 = call float @llvm.fma.f32(float %w0_1, float %x0_1, float %ch_a_0)
    %ch_b_1 = call float @llvm.fma.f32(float %w1_1, float %x1_1, float %ch_b_0)
    %ch_c_1 = call float @llvm.fma.f32(float %w2_1, float %x2_1, float %ch_c_0)
    %ch_d_1 = call float @llvm.fma.f32(float %w3_1, float %x3_1, float %ch_d_0)

    %ch_a_2 = call float @llvm.fma.f32(float %w0_2, float %x0_2, float %ch_a_1)
    %ch_b_2 = call float @llvm.fma.f32(float %w1_2, float %x1_2, float %ch_b_1)
    %ch_c_2 = call float @llvm.fma.f32(float %w2_2, float %x2_2, float %ch_c_1)
    %ch_d_2 = call float @llvm.fma.f32(float %w3_2, float %x3_2, float %ch_d_1)

    %ch_a_3 = call float @llvm.fma.f32(float %w0_3, float %x0_3, float %ch_a_2)
    %ch_b_3 = call float @llvm.fma.f32(float %w1_3, float %x1_3, float %ch_b_2)
    %ch_c_3 = call float @llvm.fma.f32(float %w2_3, float %x2_3, float %ch_c_2)
    %ch_d_3 = call float @llvm.fma.f32(float %w3_3, float %x3_3, float %ch_d_2)

    %ch_a_4 = call float @llvm.fma.f32(float %w0_4, float %x0_4, float %ch_a_3)
    %ch_b_4 = call float @llvm.fma.f32(float %w1_4, float %x1_4, float %ch_b_3)
    %ch_c_4 = call float @llvm.fma.f32(float %w2_4, float %x2_4, float %ch_c_3)
    %ch_d_4 = call float @llvm.fma.f32(float %w3_4, float %x3_4, float %ch_d_3)

    %ch_a_5 = call float @llvm.fma.f32(float %w0_5, float %x0_5, float %ch_a_4)
    %ch_b_5 = call float @llvm.fma.f32(float %w1_5, float %x1_5, float %ch_b_4)
    %ch_c_5 = call float @llvm.fma.f32(float %w2_5, float %x2_5, float %ch_c_4)
    %ch_d_5 = call float @llvm.fma.f32(float %w3_5, float %x3_5, float %ch_d_4)

    %ch_a_6 = call float @llvm.fma.f32(float %w0_6, float %x0_6, float %ch_a_5)
    %ch_b_6 = call float @llvm.fma.f32(float %w1_6, float %x1_6, float %ch_b_5)
    %ch_c_6 = call float @llvm.fma.f32(float %w2_6, float %x2_6, float %ch_c_5)
    %ch_d_6 = call float @llvm.fma.f32(float %w3_6, float %x3_6, float %ch_d_5)

    %ch_a_7 = call float @llvm.fma.f32(float %w0_7, float %x0_7, float %ch_a_6)
    %ch_b_7 = call float @llvm.fma.f32(float %w1_7, float %x1_7, float %ch_b_6)
    %ch_c_7 = call float @llvm.fma.f32(float %w2_7, float %x2_7, float %ch_c_6)
    %ch_d_7 = call float @llvm.fma.f32(float %w3_7, float %x3_7, float %ch_d_6)

    %sum_ab = fadd float %ch_a_7, %ch_b_7
    %sum_cd = fadd float %ch_c_7, %ch_d_7
    %block_sum = fadd float %sum_ab, %sum_cd

    %acc_curr = call float @llvm.fma.f32(float %block_sum, float %scale, float %acc)
    br label %outer_loop_end

outer_loop_end:
    %acc_next = phi float [ %acc_curr, %coop_x_second ]
    %b_next = add nuw nsw i32 %b, 1
    call void @llvm.amdgcn.s.barrier()
    br label %outer_loop

outer_loop_exit:
    ; Use out_stride if > 0, else c_out
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
declare i32 @llvm.amdgcn.ubfe.i32(i32, i32, i32) #0
declare float @llvm.fma.f32(float, float, float) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { convergent nounwind }
