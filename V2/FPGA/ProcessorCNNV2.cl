// ============================================================================
// ACELERADOR DE REDES NEURONALES INT8 - EXTREME PERFORMANCE EDITION (ROUTING FIXED)
// ============================================================================
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel uchar ch_in __attribute__((depth(64)));
channel uchar ch_out __attribute__((depth(64)));

#define TILE_OUT_C 16
#define MAX_W_PER_OC 4608

// ============================================================================
// KERNEL 1: LECTURA DE MEMORIA (CANAL APLANADO)
// ============================================================================
__attribute__((max_global_work_dim(0))) 
__kernel void mem_read_generic(
    __global const uchar* restrict mem_in,
    int in_offset, int w, int h, int in_c, int stride, int pad, uchar x_zero)
{
    int tmp_h = h + 2 * pad - 3;
    int tmp_w = w + 2 * pad - 3;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;
    int w_in_c = w * in_c;

    #pragma unroll 1
    for (int oy = 0; oy < out_h; oy++) {
        #pragma unroll 1
        for (int ox = 0; ox < out_w; ox++) {
            #pragma unroll 1
            for (int ky = 0; ky < 3; ky++) {
                int iy = oy * stride - pad + ky;
                
                #pragma unroll 1
                for (int kx = 0; kx < 3; kx++) {
                    int ix = ox * stride - pad + kx;
                    
                    // Condición limpia
                    bool out_of_bounds = (iy < 0 || iy >= h || ix < 0 || ix >= w);
                    int base_idx = in_offset + iy * w_in_c + ix * in_c;
                    
                    // ARREGLO KERNEL 1: Un solo canal de escritura. Cero condicionales complejos.
                    #pragma unroll 1
                    for (int c = 0; c < in_c; c++) {
                        uchar val_to_write = x_zero;
                        if (!out_of_bounds) {
                            val_to_write = mem_in[base_idx + c];
                        }
                        write_channel_intel(ch_in, val_to_write);
                    }
                }
            }
        }
    }
}

// ============================================================================
// KERNEL 2: MOTOR MATEMÁTICO (SHIFT REGISTERS)
// ============================================================================
__attribute__((max_global_work_dim(0)))
__kernel void conv_generic(
    __global const uchar* restrict weights,
    __global const int* restrict bias,
    int w_off, int b_off, int w, int h, int in_c, int tile_channels,
    int stride, int pad, uchar x_zero, uchar w_zero, int y_zero,
    int M_mult, int M_shift)
{
    int tmp_h = h + 2 * pad - 3;
    int tmp_w = w + 2 * pad - 3;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;

    int local_bias[TILE_OUT_C];
    #pragma unroll 1
    for(int oc = 0; oc < TILE_OUT_C; oc++) {
        if (oc < tile_channels) local_bias[oc] = bias[(b_off >> 2) + oc]; 
    }

    __attribute__((numbanks(TILE_OUT_C), bankwidth(1)))
    __local uchar local_W[MAX_W_PER_OC][TILE_OUT_C];
    
    int limit = 3 * 3 * in_c;
    #pragma unroll 1
    for (int oc = 0; oc < TILE_OUT_C; oc++) {
        if (oc < tile_channels) {
            int base_w = w_off + oc * limit;
            #pragma unroll 1
            for (int i = 0; i < limit; i++) {
                local_W[i][oc] = weights[base_w + i];
            }
        }
    }

    #pragma unroll 1
    for (int oy = 0; oy < out_h; oy++) {
        #pragma unroll 1
        for (int ox = 0; ox < out_w; ox++) {

            // ARREGLO KERNEL 2: Shift Register puro. Cero Multiplexores (MUXes).
            __attribute__((register)) int shift_reg[TILE_OUT_C][8];
            
            #pragma unroll
            for (int oc = 0; oc < TILE_OUT_C; oc++) {
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    shift_reg[oc][b] = 0;
                }
            }

            #pragma unroll 1
            for (int i = 0; i < limit; i++) {
                uchar px = read_channel_intel(ch_in);
                int px_val = (int)px - (int)x_zero;

                #pragma unroll
                for (int oc = 0; oc < TILE_OUT_C; oc++) {
                    int w_val = 0;
                    if (oc < tile_channels) {
                        w_val = (int)local_W[i][oc] - (int)w_zero;
                    }
                    
                    int mult_res = px_val * w_val;
                    int oldest_val = shift_reg[oc][7];
                    
                    // Empujamos los datos físicamente por el cable (cero lógica)
                    #pragma unroll
                    for (int b = 7; b > 0; b--) {
                        shift_reg[oc][b] = shift_reg[oc][b-1];
                    }
                    
                    // Sumamos al más antiguo (garantiza el II=1)
                    shift_reg[oc][0] = oldest_val + mult_res; 
                }
            }

            // Suma final de la cascada
            #pragma unroll 1
            for (int oc = 0; oc < tile_channels; oc++) {
                int res = local_bias[oc];
                
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    res += shift_reg[oc][b];
                }

                long long res_mult = (long long)res * (long long)M_mult;
                int q_res = 0;

                if (M_shift > 0) {
                    long long nudge = 1LL << (M_shift - 1); 
                    q_res = (int)((res_mult + nudge) >> M_shift);
                } else {
                    q_res = (int)res_mult;
                }

                q_res += y_zero;
                if (q_res < 0) q_res = 0;
                if (q_res > 255) q_res = 255;

                write_channel_intel(ch_out, (uchar)q_res);
            }
        }
    }
}

// ============================================================================
// KERNEL 3: ESCRITURA EN MEMORIA EXTERNA
// ============================================================================
__attribute__((max_global_work_dim(0)))
__kernel void mem_write_generic(
    __global uchar* restrict mem_out,
    int out_offset, int w, int h, int out_c, int tile_channels, int t_offset,
    int stride, int pad)
{
    int tmp_h = h + 2 * pad - 3;
    int tmp_w = w + 2 * pad - 3;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;
    int current_out_idx = out_offset + t_offset; 

    #pragma unroll 1
    for (int oy = 0; oy < out_h; oy++) {
        #pragma unroll 1
        for (int ox = 0; ox < out_w; ox++) {
            #pragma unroll 1
            for (int oc = 0; oc < tile_channels; oc++) {
                uchar val = read_channel_intel(ch_out);
                mem_out[current_out_idx + oc] = val;
            }
            current_out_idx += out_c; 
        }
    }
}