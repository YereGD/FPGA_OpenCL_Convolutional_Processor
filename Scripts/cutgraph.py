import onnx
from onnx import version_converter, helper
MODELO_ORIGINAL = "modelo_int8.onnx"
MODELO_MODIFICADO = "modelo_int8_cut.onnx"

NOMBRE_TENSOR_NUEVA_ENTRADA = "input_quantized"

def modify_model_input_to_uint8(input_path, output_path, new_input_name):
    """
    Crea un nuevo modelo ONNX cuyo input es un tensor intermedio del modelo original.
    """
    try:
        print(f"Cargando modelo original: {input_path}")
        modelo_original = onnx.load(input_path)
        
        # Mantenemos las salidas originales del modelo
        nombres_salida_original = [out.name for out in modelo_original.graph.output]
        
        print(f"Extrayendo subgrafo. Nueva entrada: '{new_input_name}'")
        
        # Usamos la herramienta de extracción para cortar el grafo en el punto deseado
        onnx.utils.extract_model(
            input_path, #Modelo original
            output_path, #Modelo de salida
            [new_input_name], # El nuevo tensor de entrada
            nombres_salida_original #Las salidas originales
        )
        
        print(f"\n¡Modelo modificado guardado en: {output_path}! ✅")
        
    except Exception as e:
        print(f"\nError: No se pudo procesar el modelo. Revisa los nombres.")
        print(e)
        return

def verify_new_model(model_path):
    """
    Carga el nuevo modelo y verifica las propiedades de su entrada.
    """
    try:
        import onnxruntime as ort
        
        print("\n--- Verificando el nuevo modelo ---")
        session = ort.InferenceSession(model_path)
        new_input = session.get_inputs()[0]
        
        # Mapeo de tipos de ONNX a strings legibles
        dtype_map = {
            'tensor(float)': 'float32',
            'tensor(uint8)': 'uint8',
            'tensor(int8)': 'int8',
        }
        
        print(f"Nombre de la nueva entrada: {new_input.name}")
        print(f"Forma (Shape) de la entrada: {new_input.shape}")
        print(f"Tipo de dato de la entrada: {dtype_map.get(new_input.type, new_input.type)}")
        
        if new_input.type == 'tensor(uint8)':
            print("\n¡Éxito! La entrada del modelo es ahora uint8.")
        else:
            print("\nAviso: La entrada del modelo NO es uint8. Revisa el nombre del tensor.")
            
    except ImportError:
        print("\nVerificación omitida: onnxruntime no está instalado.")
    except Exception as e:
        print(f"\nError al verificar el modelo: {e}")

if __name__ == "__main__":
    modify_model_input_to_uint8(MODELO_ORIGINAL, MODELO_MODIFICADO, NOMBRE_TENSOR_NUEVA_ENTRADA)
    verify_new_model(MODELO_MODIFICADO)