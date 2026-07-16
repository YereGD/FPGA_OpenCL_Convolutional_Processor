# FPGA_OpenCL_Convolutional_Processor




## Sobre el Proyecto


---

## Estructura del Repositorio

```
.
├── BenchmarkSoloARM
│   └── mainONNXCPUBenchmark.cpp
├── Modelos
│   ├── modelo_cola_cpu.onnx
│   └── modelo_int8_cut.onnx
├── Scripts
│   ├── cutgraph.py
│   └── generate_schedule_export_weight_biases.py
├── V1
│   ├── FPGA
│   │   ├── ProcessorCNN.aoco
│   │   ├── ProcessorCNN.aocx
│   │   └── ProcessorCNN.cl
│   └── Host
│       └── mainV2.cpp
├── V2
│   ├── FPGA
│   │   ├── ProcessorCNNV2.aoco
│   │   ├── ProcessorCNNV2.aocx
│   │   └── ProcessorCNNV2.cl
│   └── Host
│       └── mainV2.cpp
├── V3
│   ├── FPGA
│   │   ├── ProcessorCNNV3.aoco
│   │   ├── ProcessorCNNV3.aocx
│   │   └── ProcessorCNNV3.cl
│   └── Host
│       └── mainV3Cam.cpp
├── V4
│   ├── FPGA
│   │   ├── ProcessorCNNV4.aoco
│   │   ├── ProcessorCNNV4.aocx
│   │   └── ProcessorCNNV4.cl
│   └── Host
│       └── mainV4Cam.cpp
├── V5
│   ├── FPGA
│   │   ├── ProcessorCNNV5.aoco
│   │   ├── ProcessorCNNV5.aocx
│   │   └── ProcessorCNNV5.cl
│   └── Host
│       └── mainV5cam.cpp
├── V6
│   ├── FPGA
│   │   ├── ProcessorCNNV6.aoco
│   │   ├── ProcessorCNNV6.aocx
│   │   └── ProcessorCNNV6.cl
│   └── Host
│       └── mainV6cam.cpp
├── V7
│   ├── FPGA
│   │   ├── ProcessorCNNV7.aoco
│   │   ├── ProcessorCNNV7.aocx
│   │   └── ProcessorCNNV7.cl
│   └── Host
│       └── mainV7cam.cpp
├── V8
│   ├── FPGA
│   │   ├── ProcessorCNNV8.aoco
│   │   ├── ProcessorCNNV8.aocx
│   │   └── ProcessorCNNV8.cl
│   └── Host
│       └── mainV8cam.cpp
├── bias.bin
├── build.sh
├── build_emu.sh
├── build_onnx.sh
├── mainDefinitivo.cpp
├── modelo_cola_cpu.onnx
├── network_schedule.h
└── weights.bin
```


---

## Tecnologías Utilizadas

* **Lenguajes**: C, C++, OpenCL
* **Framework**: OpenCL, onnx
* **Hardware (Target)**: SoC DE10-Nano
* **Herramientas**: aoc y aocl linux, onnxruntime, opencv

---

## Ejecución

Para compilar y ejecutar este proyecto, necesitarás tener el SDK OpenCL para FPGA del fabricante.

### Prerrequisitos

* SDK de OpenCL del fabricante (En este proyecto con la version 18.1 de intel)
* Drivers de la FPGA
* Onnxrutime
* Opencv

### Compilación y Ejecución

1.  **Compilar el Host:**
    ```bash
    ./build.sh
    ```
    ```bash
    ./build_emu.sh
    ```

2.  **Compilar el Kernel (FPGA):**


    ```bash
    aoc ProcessorCNNV8.cl
    ```

3.  **Ejecutar el Pipeline (Se ha de tener en el mismo directorio los ficheros weights.bin, bias.bin, network_schedule.h y modelo_cola_cpu.onnx):**
    ```bash
    ./aplicacion
    ```

---

## Resultados

Latencias


Estas son las latencias comparadas entre varias soluciones:


### Base Float32


### Opencl Int8 primeras 3 convoluciones
