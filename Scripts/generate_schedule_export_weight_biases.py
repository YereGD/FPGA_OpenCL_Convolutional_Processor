import os
import math
import numpy as np
import onnx
from onnx import numpy_helper

def extract_scalar(tensor):
    return numpy_helper.to_array(tensor).item()

def quantize_multiplier(real_multiplier):
    if real_multiplier == 0: return 0, 0
    s, m = math.frexp(real_multiplier)
    q_fixed = round(s * (1 << 31))
    if q_fixed == (1 << 31):
        q_fixed /= 2; m += 1
    return int(q_fixed), int(31 - m)

def align_to_16(val): 
    return (val + 15) // 16 * 16

def export_uint8_tiled_architecture(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    
    w_file = open("weights.bin", "wb")
    b_file = open("bias.bin", "wb")
    
    nodes = [n for n in graph.node if n.op_type == 'QLinearConv']
    conv_inputs = set()
    for node in nodes: conv_inputs.add(node.input[0])
    
    weights_dict = {init.name: init for init in graph.initializer}
    
    header = "#ifndef NETWORK_SCHEDULE_H\n#define NETWORK_SCHEDULE_H\n\n#include <stdint.h>\n\n"
    header += "typedef struct {\n"
    header += "    int buf_in, in_offset, buf_out, out_offset;\n"
    header += "    int w_offset, b_offset, w, h, in_c, out_c, out_c_orig, stride, pad;\n"
    header += "    unsigned char x_zero, w_zero, y_zero;\n"
    header += "    int M_multiplier, M_shift;\n    float y_scale;\n} ConvLayerDesc;\n\n"
    layers_str = "ConvLayerDesc network_schedule[] = {\n"
    
    current_w_offset = 0; current_b_offset = 0
    tensor_shapes = {graph.input[0].name: (align_to_16(3), 240, 320)}
    buffer_map = {graph.input[0].name: (0, 0)} 
    
    # 🚨 CORRECCIÓN FINAL: Añadido el buffer 2 al free_offset
    free_offset = {0: align_to_16(3) * 240 * 320, 1: 0, 2: 0}
    layer_count = 0
    
    for i, node in enumerate(nodes):
        in_name = node.input[0]
        out_name = node.output[0]
        
        x_scale = extract_scalar(weights_dict[node.input[1]])
        x_zp = extract_scalar(weights_dict[node.input[2]])
        w_scale = extract_scalar(weights_dict[node.input[4]])
        w_zp = extract_scalar(weights_dict[node.input[5]])
        y_scale = extract_scalar(weights_dict[node.input[6]])
        y_zp = extract_scalar(weights_dict[node.input[7]])
        
        w_tensor = numpy_helper.to_array(weights_dict[node.input[3]])
        out_c, in_c, kh, kw = w_tensor.shape
        out_c_aligned = align_to_16(out_c)
        
        attrs = {attr.name: attr for attr in node.attribute}
        hw_pad = attrs['pads'].ints[0] if 'pads' in attrs else 0
        strides = attrs['strides'].ints[0] if 'strides' in attrs else 1
        
        # 🚨 PARCHE DE ORO 1x1: Evitamos que la FPGA se coma los bordes
        if kh == 1 and kw == 1:
            hw_pad = 1

        # 🚨 DETECCIÓN DEPTHWISE: Colocamos pesos en diagonal
        group = attrs['group'].i if 'group' in attrs else 1
        is_depthwise = (group == out_c and out_c > 1)

        if is_depthwise:
            in_c_aligned = align_to_16(out_c)
            w_aligned = np.full((out_c_aligned, 3, 3, in_c_aligned), w_zp, dtype=np.uint8)
            for o in range(out_c):
                if kh == 1:
                    w_aligned[o, 1:2, 1:2, o] = w_tensor[o, 0, :, :]
                else:
                    w_aligned[o, :kh, :kw, o] = w_tensor[o, 0, :, :]
        else:
            in_c_aligned = align_to_16(in_c)
            w_aligned = np.full((out_c_aligned, 3, 3, in_c_aligned), w_zp, dtype=np.uint8)
            w_reshaped = w_tensor.transpose(0, 2, 3, 1) 
            if kh == 1:
                w_aligned[:out_c, 1:2, 1:2, :in_c] = w_reshaped
            else:
                w_aligned[:out_c, :kh, :kw, :in_c] = w_reshaped
            
        w_file.write(w_aligned.tobytes())
        
        b_aligned = np.zeros(out_c_aligned, dtype=np.int32)
        if len(node.input) > 8:
            b_aligned[:out_c] = numpy_helper.to_array(weights_dict[node.input[8]]).flatten()
        b_file.write(b_aligned.tobytes())
        
        M_mult, M_shift = quantize_multiplier((x_scale * w_scale) / y_scale)
        curr_c, curr_h, curr_w = tensor_shapes[in_name]
        
        out_h = (curr_h + 2 * hw_pad - 3) // strides + 1
        out_w = (curr_w + 2 * hw_pad - 3) // strides + 1
        
        is_output = out_name not in conv_inputs
        buf_in_id, in_offset = buffer_map[in_name]
        
        # 🚨 GESTIÓN DE MEMORIA FINAL: Para que las cajas no se machaquen
        if is_output:
            buf_out = 2
        else:
            buf_out = 1 - buf_in_id
            
        out_offset = free_offset[buf_out]
        
        if not is_output:
            buffer_map[out_name] = (buf_out, out_offset)
            
        free_offset[buf_out] += (out_c_aligned * out_h * out_w)
        
        tensor_shapes[out_name] = (out_c_aligned, out_h, out_w)
        
        layers_str += f"    {{ {buf_in_id}, {in_offset}, {buf_out}, {out_offset}, {current_w_offset}, {current_b_offset}, {curr_w}, {curr_h}, {in_c_aligned}, {out_c_aligned}, {out_c}, {strides}, {hw_pad}, {int(x_zp)}, {int(w_zp)}, {int(y_zp)}, {M_mult}, {M_shift}, {float(y_scale)}f }},\n"
        
        current_w_offset += (out_c_aligned * 3 * 3 * in_c_aligned)
        current_b_offset += (out_c_aligned * 4)
        layer_count += 1

    header += f"#define TOTAL_LAYERS {layer_count}\n\n" + layers_str + "};\n\n#endif\n"
    with open("network_schedule.h", 'w') as f: f.write(header)
    w_file.close(); b_file.close()
    print("Schedule Tiled generado (Soporte Nativo DEPTHWISE + Parche Geometría 1x1 + Memoria Final Secuencial).")

if __name__ == "__main__":
    export_uint8_tiled_architecture("modelo_int8_cut.onnx")