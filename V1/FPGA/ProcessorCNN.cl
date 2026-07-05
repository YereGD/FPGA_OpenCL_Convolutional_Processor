// ============================================================================
// ACELERADOR DE REDES NEURONALES INT8 - INTEL FPGA (CYCLONE V)
// ============================================================================
#pragma OPENCL EXTENSION cl_intel_channels : enable

// Canales de comunicación interna en hardware (FIFOs)
channel uchar ch_in __attribute__((depth(32)));
channel uchar ch_out __attribute__((depth(32)));

// 🔥 TILING AJUSTADO A 16 (Garantiza velocidad y éxito en el enrutamiento)
#define TILE_OUT_C 16
#define MAX_W_PER_OC 9216 // 3*3*1024 (Máximo posible en nuestra arquitectura)

// ============================================================================
// KERNEL 1: LECTURA DE MEMORIA EXTERNA (DDR3)
// ============================================================================
__attribute__((max_global_work_dim(0))) 
__kernel void mem_read_generic(
    __global const uchar* restrict mem_in,
    int in_offset, int w, int h, int in_c, int stride, int pad, uchar x_zero)
{
    // OPTIMIZACIÓN 1: Cero Divisiones Hardware (Uso de Bit Shift para stride 2)
    int tmp_h = h + 2 * pad - 3;
    int tmp_w = w + 2 * pad - 3;
    int out_h = (stride == 2 ? (tmp_h >> 1) : tmp_h) + 1;
    int out_w = (stride == 2 ? (tmp_w >> 1) : tmp_w) + 1;

    // OPTIMIZACIÓN 2: Precalcular constantes fuera de los bucles para ahorrar ALUTs
    int w_in_c = w * in_c;

    #pragma unroll 1
    for (int oy = 0; oy < out_h; oy++) {
        #pragma unroll 1
        for (int ox = 0; ox < out_w; ox++) {
            #pragma unroll 1
            for (int ky = 0; ky < 3; ky++) {
                #pragma unroll 1
                for (int kx = 0; kx < 3; kx++) {
                    
                    int iy = oy * stride - pad + ky;
                    int ix = ox * stride - pad + kx;
                    bool out_of_bounds = (iy < 0 || iy >= h || ix < 0 || ix >= w);

                    // Índice base calculado con sumas baratas
                    int base_idx = in_offset + iy * w_in_c + ix * in_c;

                    #pragma unroll 1
                    for (int c = 0; c < in_c; c++) {
                        uchar val = x_zero;
                        if (!out_of_bounds) {
                            val = mem_in[base_idx + c];
                        }
                        write_channel_intel(ch_in, val);
                    }
                }
            }
        }
    }
}

// ============================================================================
// KERNEL 2: MOTOR MATEMÁTICO (CONVOLUCIÓN INT8)
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
        if (oc < tile_channels) local_bias[oc] = bias[(b_off / 4) + oc]; 
    }

    // 🔥 SOLUCIÓN A LOS PUNTOS VERDES Y LA PANTALLA NEGRA:
    // Al usar una matriz 2D pura sin atributos de banco, la FPGA construirá 16 "taquillas" físicas.
    // Esto permite a los 16 trabajadores leer sus pesos simultáneamente sin colapsar.
    __local uchar local_W[TILE_OUT_C][MAX_W_PER_OC];
    
    int limit = 3 * 3 * in_c;
    #pragma unroll 1
    for (int oc = 0; oc < TILE_OUT_C; oc++) {
        if (oc < tile_channels) {
            int base_w = w_off + oc * limit;
            #pragma unroll 1
            for (int i = 0; i < limit; i++) {
                local_W[oc][i] = weights[base_w + i];
            }
        }
    }

    #pragma unroll 1
    for (int oy = 0; oy < out_h; oy++) {
        #pragma unroll 1
        for (int ox = 0; ox < out_w; ox++) {

            int acc[TILE_OUT_C];
            #pragma unroll
            for (int oc = 0; oc < TILE_OUT_C; oc++) {
                acc[oc] = 0;
            }

            int w_idx = 0; 
            
            #pragma unroll 1
            for (int ky = 0; ky < 3; ky++) {
                #pragma unroll 1
                for (int kx = 0; kx < 3; kx++) {
                    #pragma unroll 1
                    for (int c = 0; c < in_c; c++) {
                        
                        uchar px = read_channel_intel(ch_in);
                        int px_val = (int)px - (int)x_zero;

                        // MATEMÁTICAS EN PARALELO (16 DSPs trabajando a la vez)
                        #pragma unroll
                        for (int oc = 0; oc < TILE_OUT_C; oc++) {
                            if (oc < tile_channels) {
                                int w_val = (int)local_W[oc][w_idx] - (int)w_zero;
                                acc[oc] += px_val * w_val;
                            }
                        }
                        w_idx++; // Incremento barato (cero multiplicaciones complejas)
                    }
                }
            }

            #pragma unroll 1
            for (int oc = 0; oc < tile_channels; oc++) {
                int res = acc[oc] + local_bias[oc];
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