#pragma OPENCL EXTENSION cl_intel_channels : enable

#define TILE_C 8
#define MAX_WEIGHT_ITERS 1152 
#define MAX_ROW_VECS 2560  
#define MAX_RING_SIZE 7680 

__attribute__((max_global_work_dim(0))) 
__kernel void cnn_engine(
    __global const uchar4* restrict mem_in,     
    __global const uchar4* restrict weights_vec,
    __global const int* restrict bias,          
    __global uchar* restrict mem_out,           
    int in_offset_bytes, int out_offset_bytes, int w_off_bytes, int b_off_ints,
    int w, int h, int in_c, int out_c,
    int stride, int pad, uchar x_zero, uchar w_zero, int y_zero,
    int M_mult, int M_shift)
{
    int in_offset_vec = in_offset_bytes >> 2; 
    int vec_c = in_c >> 2; 
    int w_vec_c = w * vec_c; 
    int base_w_off_vec = w_off_bytes >> 2; 

    int tmp_w = w + 2 * pad - 3;
    int tmp_h = h + 2 * pad - 3;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;
    int total_inner_iters = 9 * vec_c;

    short xz = (short)x_zero;
    short wz = (short)w_zero;
    uchar4 w_zero_vec = (uchar4)(w_zero, w_zero, w_zero, w_zero);

    long long round_term = (M_shift > 0) ? (1LL << (M_shift - 1)) : 0;

    __local short4 ring_buf[MAX_RING_SIZE] __attribute__((memory, numbanks(1), singlepump));
    
    __local short4 lw0[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));
    __local short4 lw1[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));
    __local short4 lw2[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));
    __local short4 lw3[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));
    __local short4 lw4[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));
    __local short4 lw5[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));
    __local short4 lw6[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));
    __local short4 lw7[MAX_WEIGHT_ITERS] __attribute__((memory, numbanks(1), singlepump));

    for (int t = 0; t < out_c; t += TILE_C) {
        
        int tile_channels = out_c - t;
        if (tile_channels > TILE_C) tile_channels = TILE_C;

        #pragma unroll 1 
        for (int i = 0; i < total_inner_iters; i++) {
            uchar4 val0 = (0 < tile_channels) ? weights_vec[base_w_off_vec + (t + 0)*total_inner_iters + i] : w_zero_vec;
            uchar4 val1 = (1 < tile_channels) ? weights_vec[base_w_off_vec + (t + 1)*total_inner_iters + i] : w_zero_vec;
            uchar4 val2 = (2 < tile_channels) ? weights_vec[base_w_off_vec + (t + 2)*total_inner_iters + i] : w_zero_vec;
            uchar4 val3 = (3 < tile_channels) ? weights_vec[base_w_off_vec + (t + 3)*total_inner_iters + i] : w_zero_vec;
            uchar4 val4 = (4 < tile_channels) ? weights_vec[base_w_off_vec + (t + 4)*total_inner_iters + i] : w_zero_vec;
            uchar4 val5 = (5 < tile_channels) ? weights_vec[base_w_off_vec + (t + 5)*total_inner_iters + i] : w_zero_vec;
            uchar4 val6 = (6 < tile_channels) ? weights_vec[base_w_off_vec + (t + 6)*total_inner_iters + i] : w_zero_vec;
            uchar4 val7 = (7 < tile_channels) ? weights_vec[base_w_off_vec + (t + 7)*total_inner_iters + i] : w_zero_vec;

            lw0[i] = (short4)((short)val0.s0 - wz, (short)val0.s1 - wz, (short)val0.s2 - wz, (short)val0.s3 - wz);
            lw1[i] = (short4)((short)val1.s0 - wz, (short)val1.s1 - wz, (short)val1.s2 - wz, (short)val1.s3 - wz);
            lw2[i] = (short4)((short)val2.s0 - wz, (short)val2.s1 - wz, (short)val2.s2 - wz, (short)val2.s3 - wz);
            lw3[i] = (short4)((short)val3.s0 - wz, (short)val3.s1 - wz, (short)val3.s2 - wz, (short)val3.s3 - wz);
            lw4[i] = (short4)((short)val4.s0 - wz, (short)val4.s1 - wz, (short)val4.s2 - wz, (short)val4.s3 - wz);
            lw5[i] = (short4)((short)val5.s0 - wz, (short)val5.s1 - wz, (short)val5.s2 - wz, (short)val5.s3 - wz);
            lw6[i] = (short4)((short)val6.s0 - wz, (short)val6.s1 - wz, (short)val6.s2 - wz, (short)val6.s3 - wz);
            lw7[i] = (short4)((short)val7.s0 - wz, (short)val7.s1 - wz, (short)val7.s2 - wz, (short)val7.s3 - wz);
        }

        int local_b[8];
        for(int oc=0; oc < TILE_C; oc++) {
            local_b[oc] = (oc < tile_channels) ? bias[b_off_ints + t + oc] : 0;
        }

        int highest_fetched_y = -1;
        // 🚀 Nuevo Tracker: Evita usar % 3 al inicio
        int highest_fetched_y_mod3 = 2; 

        int p = pad;
        while (p >= 3) { p -= 3; }
        int running_mod3 = (p == 0) ? 0 : (3 - p);

        for (int y = 0; y < out_h; y++) {
            int base_y = (stride == 2 ? (y << 1) : y) - pad;
            
            int base_y_mod3 = running_mod3;
            running_mod3 += stride;
            if (running_mod3 >= 3) running_mod3 -= 3;
            
            int start_y = highest_fetched_y + 1;
            if (start_y < base_y) start_y = base_y; 
            if (start_y < 0) start_y = 0; 
            
            int end_y = base_y + 2;
            if (end_y >= h) end_y = h - 1;

            int rows_to_fetch = end_y - start_y + 1;
            
            if (rows_to_fetch > 0) {
                // 🚀 ADIÓS F_Y_MOD = START_Y % 3 (Ahorro de 4.092 ALUTs)
                int f_y_mod;
                if (start_y == 0) f_y_mod = 0;
                else if (start_y == base_y) f_y_mod = base_y_mod3;
                else {
                    f_y_mod = highest_fetched_y_mod3 + 1;
                    if (f_y_mod >= 3) f_y_mod -= 3;
                }

                #pragma unroll 1
                for(int f_y_idx = 0; f_y_idx < rows_to_fetch; f_y_idx++) {
                    int current_y = start_y + f_y_idx;
                    int ring_base = f_y_mod * MAX_ROW_VECS;
                    int mem_base = in_offset_vec + current_y * w_vec_c;
                    
                    #pragma unroll 1
                    for(int i = 0; i < w_vec_c; i++) {
                        uchar4 val = mem_in[mem_base + i];
                        ring_buf[ring_base + i] = (short4)((short)val.s0 - xz, (short)val.s1 - xz, (short)val.s2 - xz, (short)val.s3 - xz);
                    }

                    f_y_mod++;
                    if (f_y_mod >= 3) f_y_mod -= 3;
                }
                
                highest_fetched_y = end_y;
                highest_fetched_y_mod3 = f_y_mod - 1;
                if (highest_fetched_y_mod3 < 0) highest_fetched_y_mod3 += 3;
            }

            for (int x = 0; x < out_w; x++) {
                int base_x = (stride == 2 ? (x << 1) : x) - pad;
                
                int a0 = local_b[0], a1 = local_b[1], a2 = local_b[2], a3 = local_b[3];
                int a4 = local_b[4], a5 = local_b[5], a6 = local_b[6], a7 = local_b[7];

                int w_idx = 0;
                int base_x_addr = base_x * vec_c;

                // 🚀 ADIÓS OPERADOR TERNARIO (Ahorro de 4.272 ALUTs)
                // Usamos un acumulador circular que suma +MAX_ROW_VECS
                int current_row_ptr = base_y_mod3 * MAX_ROW_VECS;

                #pragma unroll 1
                for (int ky = 0; ky < 3; ky++) {
                    int in_y = base_y + ky;
                    bool valid_y = (in_y >= 0 && in_y < h);

                    #pragma unroll 1
                    for (int kx = 0; kx < 3; kx++) {
                        int in_x = base_x + kx;
                        bool valid_x = (in_x >= 0 && in_x < w);
                        bool valid = valid_y && valid_x; 
                        
                        int kx_off = (kx == 0) ? 0 : ((kx == 1) ? vec_c : (vec_c << 1));
                        int base_addr = current_row_ptr + base_x_addr + kx_off;
                        int safe_base = valid ? base_addr : 0;

                        #pragma unroll 1
                        for (int c = 0; c < vec_c; c++) {
                            short4 px = valid ? ring_buf[safe_base + c] : (short4)(0,0,0,0);
                            short4 w_v;

                            w_v = lw0[w_idx];
                            a0 += (int)px.s0 * w_v.s0; a0 += (int)px.s1 * w_v.s1;
                            a0 += (int)px.s2 * w_v.s2; a0 += (int)px.s3 * w_v.s3;

                            w_v = lw1[w_idx];
                            a1 += (int)px.s0 * w_v.s0; a1 += (int)px.s1 * w_v.s1;
                            a1 += (int)px.s2 * w_v.s2; a1 += (int)px.s3 * w_v.s3;

                            w_v = lw2[w_idx];
                            a2 += (int)px.s0 * w_v.s0; a2 += (int)px.s1 * w_v.s1;
                            a2 += (int)px.s2 * w_v.s2; a2 += (int)px.s3 * w_v.s3;

                            w_v = lw3[w_idx];
                            a3 += (int)px.s0 * w_v.s0; a3 += (int)px.s1 * w_v.s1;
                            a3 += (int)px.s2 * w_v.s2; a3 += (int)px.s3 * w_v.s3;

                            w_v = lw4[w_idx];
                            a4 += (int)px.s0 * w_v.s0; a4 += (int)px.s1 * w_v.s1;
                            a4 += (int)px.s2 * w_v.s2; a4 += (int)px.s3 * w_v.s3;

                            w_v = lw5[w_idx];
                            a5 += (int)px.s0 * w_v.s0; a5 += (int)px.s1 * w_v.s1;
                            a5 += (int)px.s2 * w_v.s2; a5 += (int)px.s3 * w_v.s3;

                            w_v = lw6[w_idx];
                            a6 += (int)px.s0 * w_v.s0; a6 += (int)px.s1 * w_v.s1;
                            a6 += (int)px.s2 * w_v.s2; a6 += (int)px.s3 * w_v.s3;

                            w_v = lw7[w_idx];
                            a7 += (int)px.s0 * w_v.s0; a7 += (int)px.s1 * w_v.s1;
                            a7 += (int)px.s2 * w_v.s2; a7 += (int)px.s3 * w_v.s3;

                            w_idx++;
                        }
                    }
                    // 🚀 Actualización circular del puntero en 0 ALUTs
                    current_row_ptr += MAX_ROW_VECS;
                    if (current_row_ptr >= 3 * MAX_ROW_VECS) current_row_ptr -= 3 * MAX_ROW_VECS;
                }

                int acc_arr[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
                uchar8 out_vec;

                #pragma unroll 1 
                for (int oc = 0; oc < 8; oc++) {
                    long long res_q = (long long)(acc_arr[oc]) * M_mult;
                    int res_shifted = (int)((res_q + round_term) >> M_shift) + y_zero;
                    if (res_shifted < 0) res_shifted = 0; else if (res_shifted > 255) res_shifted = 255;
                    
                    if(oc == 0) out_vec.s0 = (uchar)res_shifted; else if(oc == 1) out_vec.s1 = (uchar)res_shifted;
                    else if(oc == 2) out_vec.s2 = (uchar)res_shifted; else if(oc == 3) out_vec.s3 = (uchar)res_shifted;
                    else if(oc == 4) out_vec.s4 = (uchar)res_shifted; else if(oc == 5) out_vec.s5 = (uchar)res_shifted;
                    else if(oc == 6) out_vec.s6 = (uchar)res_shifted; else if(oc == 7) out_vec.s7 = (uchar)res_shifted;
                }
                
                int y_off = out_offset_bytes + y * out_w * out_c;
                int base_idx_x = y_off + x * out_c + t;
                vstore8(out_vec, 0, &mem_out[base_idx_x]);
            }
        }
    }
}