//  MOTOR MONOLÍTICO: TILE_C 8 + Ejecución Autónoma de 42 Capas
#define TILE_C 8
#define MAX_WEIGHT_ITERS 576  
#define MAX_ROW_VECS 1280     
#define MAX_RING_SIZE 3840    

__attribute__((max_global_work_dim(0))) 
__kernel void cnn_engine(
    __global uchar* restrict ram_pool,          // (32MB)
    __global const uchar4* restrict weights_vec,
    __global const int* restrict bias,          
    __global const int* restrict schedule_data, //  El Manual de Instrucciones
    int total_layers)                           // 42 capas
{
    // 1. Reservamos la memoria RAM interna de la FPGA UNA SOLA VEZ
    __local short4 ring_buf[MAX_RING_SIZE] __attribute__((memory, numbanks(1)));
    __local short4 lw[MAX_WEIGHT_ITERS][TILE_C] __attribute__((memory, numbanks(8)));

    // 2. EL SÚPER-BUCLE: La FPGA se auto-configura y ejecuta cada capa de la CNN
    for (int layer = 0; layer < total_layers; layer++) {
        
        // 3. La FPGA se auto-configura leyendo su manual
        int in_offset_bytes  = schedule_data[layer * 16 + 0];
        int out_offset_bytes = schedule_data[layer * 16 + 1];
        int w_off_bytes      = schedule_data[layer * 16 + 2];
        int b_off_ints       = schedule_data[layer * 16 + 3];
        int w                = schedule_data[layer * 16 + 4];
        int h                = schedule_data[layer * 16 + 5];
        int in_c             = schedule_data[layer * 16 + 6];
        int out_c            = schedule_data[layer * 16 + 7];
        int stride           = schedule_data[layer * 16 + 8];
        int pad              = schedule_data[layer * 16 + 9];
        uchar x_zero         = (uchar)schedule_data[layer * 16 + 10];
        uchar w_zero         = (uchar)schedule_data[layer * 16 + 11];
        int y_zero           = schedule_data[layer * 16 + 12];
        int M_mult           = schedule_data[layer * 16 + 13];
        int M_shift          = schedule_data[layer * 16 + 14];

        // Mapeamos los punteros a la piscina gigante
        __global const uchar4* mem_in = (__global const uchar4*)ram_pool;
        __global uchar* mem_out = ram_pool;

        // 4. Cálculos previos de la capa
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

        //  EL NÚCLEO MATEMÁTICO (Tu V6 Optimizada y Vectorizada)
        for (int t = 0; t < out_c; t += TILE_C) {
            
            int tile_channels = out_c - t;
            if (tile_channels > TILE_C) tile_channels = TILE_C;

            int local_b[TILE_C];

            #pragma unroll
            for(int oc=0; oc < TILE_C; oc++) {
                local_b[oc] = (oc < tile_channels) ? bias[b_off_ints + t + oc] : 0;
            }
            // INTERCAMBIO DE BUCLES + COALESCE=
            #pragma loop_coalesce 2
            for (int oc = 0; oc < TILE_C; oc++) {
                for (int i = 0; i < total_inner_iters; i++) {
                    uchar4 val = (oc < tile_channels) ? weights_vec[base_w_off_vec + (t + oc)*total_inner_iters + i] : w_zero_vec;
                    lw[i][oc] = (short4)((short)val.s0 - wz, (short)val.s1 - wz, (short)val.s2 - wz, (short)val.s3 - wz);
                }
            }

            int highest_fetched_y = -1;
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
                    int f_y_mod;
                    if (start_y == 0) f_y_mod = 0;
                    else if (start_y == base_y) f_y_mod = base_y_mod3;
                    else {
                        f_y_mod = highest_fetched_y_mod3 + 1;
                        if (f_y_mod >= 3) f_y_mod -= 3;
                    }

                    #pragma unroll 1
                    #pragma loop_coalesce 2
                    for(int f_y_idx = 0; f_y_idx < rows_to_fetch; f_y_idx++) {
                        int current_y = start_y + f_y_idx;
                        int ring_base = f_y_mod * MAX_ROW_VECS;
                        int mem_base = in_offset_vec + current_y * w_vec_c;

                        #pragma unroll 1
                        for(int i = 0; i < w_vec_c; i += 4) { 
                            // Traemos 16 bytes (128 bits) en un solo ciclo de reloj
                            uchar16 v16 = vload16(0, (__global const uchar*)&mem_in[mem_base + i]);
                            
                            // Desempaquetamos en los 4 espacios correspondientes del anillo
                            ring_buf[ring_base + i]     = (short4)((short)v16.s0 - xz, (short)v16.s1 - xz, (short)v16.s2 - xz, (short)v16.s3 - xz);
                            ring_buf[ring_base + i + 1] = (short4)((short)v16.s4 - xz, (short)v16.s5 - xz, (short)v16.s6 - xz, (short)v16.s7 - xz);
                            // Nota: OpenCL nombra los elementos hexadecimales: 8, 9, a, b, c, d, e, f
                            ring_buf[ring_base + i + 2] = (short4)((short)v16.s8 - xz, (short)v16.s9 - xz, (short)v16.sa - xz, (short)v16.sb - xz);
                            ring_buf[ring_base + i + 3] = (short4)((short)v16.sc - xz, (short)v16.sd - xz, (short)v16.se - xz, (short)v16.sf - xz);
                        }

                        f_y_mod++;
                        if (f_y_mod >= 3) f_y_mod -= 3;
                    }
                    
                    highest_fetched_y = end_y;
                    highest_fetched_y_mod3 = f_y_mod - 1;
                    if (highest_fetched_y_mod3 < 0) highest_fetched_y_mod3 += 3;
                }

                int p0_mod = base_y_mod3;
                int p1_mod = base_y_mod3 + 1; if (p1_mod >= 3) p1_mod -= 3;
                int p2_mod = base_y_mod3 + 2; if (p2_mod >= 3) p2_mod -= 3;

                int row_ptrs[3];
                row_ptrs[0] = p0_mod * MAX_ROW_VECS;
                row_ptrs[1] = p1_mod * MAX_ROW_VECS;
                row_ptrs[2] = p2_mod * MAX_ROW_VECS;
                // 4. Cálculo de la convolución para cada posición de salida
                for (int x = 0; x < out_w; x++) {
                    int base_x = (stride == 2 ? (x << 1) : x) - pad;
                    
                    int a0 = local_b[0], a1 = local_b[1], a2 = local_b[2], a3 = local_b[3];
                    int a4 = local_b[4], a5 = local_b[5], a6 = local_b[6], a7 = local_b[7];

                    int w_idx = 0;
                    int base_x_addr = base_x * vec_c;

                    #pragma loop_coalesce 2
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int in_y = base_y + ky;
                            int in_x = base_x + kx;
                            bool valid = (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w);
                            
                            int current_row_ptr = row_ptrs[ky];
                            int kx_off = (kx == 0) ? 0 : ((kx == 1) ? vec_c : (vec_c << 1));
                            int base_addr = current_row_ptr + base_x_addr + kx_off;
                            int safe_base = valid ? base_addr : 0;

                            #pragma unroll 2
                            for (int c = 0; c < vec_c; c++) {
                                short4 px = valid ? ring_buf[safe_base + c] : (short4)(0,0,0,0);
                                
                                #pragma unroll
                                for(int oc=0; oc<TILE_C; oc++){
                                    short4 w_v = lw[w_idx][oc];
                                    short4 prod_vec = px * w_v; 
                                    int prod = (int)prod_vec.s0 + (int)prod_vec.s1 + (int)prod_vec.s2 + (int)prod_vec.s3;
                                    
                                    if(oc==0) a0+=prod; else if(oc==1) a1+=prod; else if(oc==2) a2+=prod; else if(oc==3) a3+=prod;
                                    else if(oc==4) a4+=prod; else if(oc==5) a5+=prod; else if(oc==6) a6+=prod; else if(oc==7) a7+=prod;
                                }
                                w_idx++;
                            }
                        }
                    }

                    // 5. Post-procesamiento: Requantización y saturación
                    int acc_arr[8];
                    acc_arr[0]=a0; acc_arr[1]=a1; acc_arr[2]=a2; acc_arr[3]=a3;
                    acc_arr[4]=a4; acc_arr[5]=a5; acc_arr[6]=a6; acc_arr[7]=a7;

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
}