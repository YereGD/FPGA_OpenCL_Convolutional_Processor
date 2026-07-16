# FPGA_OpenCL_Convolutional_Processor




## Sobre el Proyecto


---

## Estructura del Repositorio


---

## Tecnologías Utilizadas

* **Lenguajes**: C, C++, OpenCL
* **Framework**: OpenCL, onnx
* **Hardware (Target)**: SoC DE10-Nano
* **Herramientas**: aoc windows, aocl linux, onnxruntime, opencv

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

3.  **Ejecutar el Pipeline:**
    ```bash
    ./aplicacion
    ```

---

## Resultados

Latencias


Estas son las latencias comparadas entre varias soluciones:


### Base Float32


### Opencl Int8 primeras 3 convoluciones
