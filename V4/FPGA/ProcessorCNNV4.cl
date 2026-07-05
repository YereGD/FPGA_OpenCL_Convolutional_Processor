#pragma OPENCL EXTENSION cl_intel_channels : enable

#define TILE_OUT_C 8
#define MAX_WEIGHT_ITERS 576 
#define MAX_ROW_VECS 1280 // Capacidad matemática del ancho de red

channel uchar4 ch_in __attribute__((depth(16))); 
channel uchar8 ch_out __attribute__((depth(16))); 

// =========================================================================
// KERNEL 1: RING BUFFER SMART CACHE (Mínimas Lecturas a DDR3)
// =========================================================================
__attribute__((max_global_work_dim(0))) 
__kernel void mem_read_generic(
    __global const uchar4* restrict mem_in __attribute__((aligned(64))),
    int in_offset_bytes, int w, int h, int in_c, int stride, int pad, uchar x_zero) 
{
    // El Buffer Circular de 3 filas
    __local uchar4 ring_buf[3][MAX_ROW_VECS];

    int in_offset_vec = in_offset_bytes >> 2; 
    int vec_c = in_c >> 2; 
    int w_vec_c = w * vec_c; 
    
    int tmp_w = w + 2 * pad - 3;
    int tmp_h = h + 2 * pad - 3;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;

    // Llevamos la cuenta de qué hemos traído ya de la DDR3
    int highest_fetched_y = -1;

    for (int y = 0; y < out_h; y++) {
        int base_y = (stride == 2 ? (y << 1) : y) - pad;
        
        // --- 1. LECTURA BURST: Solo traemos filas NUEVAS de la DDR3 ---
        for (int ky = 0; ky < 3; ky++) {
            int in_y = base_y + ky;
            
            // ¿Es una fila válida y NO la tenemos ya en el Ring Buffer?
            if (in_y > highest_fetched_y && in_y >= 0 && in_y < h) {
                
                int ring_idx = in_y % 3; // Magia circular: 0, 1, 2, 0, 1, 2...
                int base_idx_y = in_offset_vec + in_y * w_vec_c;
                
                // Quartus convertirá este bucle en una Ráfaga (Burst) rapidísima
                for (int i = 0; i < w_vec_c; i++) {
                    ring_buf[ring_idx][i] = mem_in[base_idx_y + i];
                }
                
                // Actualizamos nuestra marca de agua
                highest_fetched_y = in_y; 
            }
        }

        // --- 2. ALIMENTACIÓN DEL MOTOR: A velocidad de vértigo ---
        int total_feed_iters = out_w * 9 * vec_c;
        int px = 0, kx = 0, ky = 0, c = 0;
        
        #pragma unroll 1
        for(int i = 0; i < total_feed_iters; i++) {
            int in_x = (stride == 2 ? (px << 1) : px) - pad + kx;
            int in_y = base_y + ky;
            
            bool valid = (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h);
            uchar4 val;
            
            if (valid) {
                // Buscamos la fila en su posición circular
                val = ring_buf[in_y % 3][in_x * vec_c + c];
            } else {
                val = (uchar4)(x_zero, x_zero, x_zero, x_zero);
            }
            
            write_channel_intel(ch_in, val); 
            
            c++;
            if (c == vec_c) {
                c = 0; kx++;
                if (kx == 3) { 
                    kx = 0; ky++; 
                    if (ky == 3) { ky = 0; px++; }
                }
            }
        }
    }
}

// =========================================================================
// KERNEL 2: MOTOR MATEMÁTICO (Intacto)
// =========================================================================
__attribute__((max_global_work_dim(0))) 
__kernel void conv_generic(
    __global const uchar4* restrict weights_vec __attribute__((aligned(64))),
    __global const int* restrict bias __attribute__((aligned(64))),
    int w_off_bytes, int b_off_ints, int w, int h, int in_c, int tile_channels,
    int stride, int pad, uchar x_zero, uchar w_zero, int y_zero,
    int M_mult, int M_shift)
{
    int vec_c = in_c >> 2; 
    int w_off_vec = w_off_bytes >> 2; 
    
    int tmp_w = w + 2 * pad - 3;
    int tmp_h = h + 2 * pad - 3;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;
    int total_inner_iters = 9 * vec_c;

    __local uchar4 local_w_0[MAX_WEIGHT_ITERS]; __local uchar4 local_w_1[MAX_WEIGHT_ITERS];
    __local uchar4 local_w_2[MAX_WEIGHT_ITERS]; __local uchar4 local_w_3[MAX_WEIGHT_ITERS];
    __local uchar4 local_w_4[MAX_WEIGHT_ITERS]; __local uchar4 local_w_5[MAX_WEIGHT_ITERS];
    __local uchar4 local_w_6[MAX_WEIGHT_ITERS]; __local uchar4 local_w_7[MAX_WEIGHT_ITERS];

    int local_b_0 = (0 < tile_channels) ? bias[b_off_ints + 0] : 0;
    int local_b_1 = (1 < tile_channels) ? bias[b_off_ints + 1] : 0;
    int local_b_2 = (2 < tile_channels) ? bias[b_off_ints + 2] : 0;
    int local_b_3 = (3 < tile_channels) ? bias[b_off_ints + 3] : 0;
    int local_b_4 = (4 < tile_channels) ? bias[b_off_ints + 4] : 0;
    int local_b_5 = (5 < tile_channels) ? bias[b_off_ints + 5] : 0;
    int local_b_6 = (6 < tile_channels) ? bias[b_off_ints + 6] : 0;
    int local_b_7 = (7 < tile_channels) ? bias[b_off_ints + 7] : 0;

    for (int oc = 0; oc < TILE_OUT_C; oc++) {
        int base = w_off_vec + (oc * total_inner_iters);
        for (int i = 0; i < total_inner_iters; i++) {
            uchar4 w_val = (oc < tile_channels) ? weights_vec[base + i] : (uchar4)(w_zero, w_zero, w_zero, w_zero);
            if (oc == 0) local_w_0[i] = w_val;      else if (oc == 1) local_w_1[i] = w_val;
            else if (oc == 2) local_w_2[i] = w_val; else if (oc == 3) local_w_3[i] = w_val;
            else if (oc == 4) local_w_4[i] = w_val; else if (oc == 5) local_w_5[i] = w_val;
            else if (oc == 6) local_w_6[i] = w_val; else if (oc == 7) local_w_7[i] = w_val;
        }
    }

    short xz = (short)x_zero; short wz = (short)w_zero;

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            int acc0 = local_b_0, acc1 = local_b_1, acc2 = local_b_2, acc3 = local_b_3;
            int acc4 = local_b_4, acc5 = local_b_5, acc6 = local_b_6, acc7 = local_b_7;

            #pragma unroll 1
            for (int i = 0; i < total_inner_iters; i++) { 
                uchar4 px = read_channel_intel(ch_in);
                short p0 = (short)px.s0 - xz; short p1 = (short)px.s1 - xz;
                short p2 = (short)px.s2 - xz; short p3 = (short)px.s3 - xz;
                uchar4 w;
                w = local_w_0[i]; acc0 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
                w = local_w_1[i]; acc1 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
                w = local_w_2[i]; acc2 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
                w = local_w_3[i]; acc3 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
                w = local_w_4[i]; acc4 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
                w = local_w_5[i]; acc5 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
                w = local_w_6[i]; acc6 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
                w = local_w_7[i]; acc7 += (int)(p0 * ((short)w.s0 - wz)) + (int)(p1 * ((short)w.s1 - wz)) + (int)(p2 * ((short)w.s2 - wz)) + (int)(p3 * ((short)w.s3 - wz));
            }

            int acc_array[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
            uchar q_out[8];

            #pragma unroll 1 
            for (int oc = 0; oc < 8; oc++) {
                long long res_q = (long long)(acc_array[oc]) * M_mult;
                int res_shifted = (int)((res_q + (1LL << (M_shift - 1))) >> M_shift) + y_zero;
                if (res_shifted < 0) res_shifted = 0; else if (res_shifted > 255) res_shifted = 255;
                q_out[oc] = (uchar)res_shifted;
            }

            uchar8 out_vec;
            out_vec.s0 = q_out[0]; out_vec.s1 = q_out[1]; out_vec.s2 = q_out[2]; out_vec.s3 = q_out[3];
            out_vec.s4 = q_out[4]; out_vec.s5 = q_out[5]; out_vec.s6 = q_out[6]; out_vec.s7 = q_out[7];
            
            write_channel_intel(ch_out, out_vec);
        }
    }
}

// =========================================================================
// KERNEL 3: ESCRITOR EN MEMORIA EXTERNA (Vectorizado)
// =========================================================================
__attribute__((max_global_work_dim(0))) 
__kernel void mem_write_generic(
    __global uchar* restrict mem_out __attribute__((aligned(64))),
    int out_offset, int w, int h, int out_c, int tile_channels, int t_offset,
    int stride, int pad)
{
    int tmp_w = w + 2 * pad - 3;
    int tmp_h = h + 2 * pad - 3;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;
    
    int out_row_stride = out_w * out_c;

    for (int y = 0; y < out_h; y++) {
        int y_off = out_offset + y * out_row_stride + t_offset; 
        for (int x = 0; x < out_w; x++) {
            uchar8 val_vec = read_channel_intel(ch_out);
            int base_idx = y_off + x * out_c;
            vstore8(val_vec, 0, &mem_out[base_idx]);
        }
    }
}