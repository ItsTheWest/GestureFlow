import shutil
from pathlib import Path

def delete_bad_folders(base_path: Path, min_sequences: int = 1) -> None:
    if not base_path.exists(): # Se evalua si la carpeta existe
        print(f"Error: Base directory '{base_path}' not found.")
        return

    for target_path in base_path.iterdir():
        if target_path.is_dir(): # Se evalua si la carpeta es un directorio
            count = len(list(target_path.glob("*.npy")))
            if count < min_sequences: # Se evalua si la carpeta tiene menos de 1 secuencia
                print(f"Eliminando '{target_path}' ({count} secuencias)...")
                shutil.rmtree(target_path) # Elimina la carpeta con menos de 1 secuencia

if __name__ == "__main__":
    gestos_dir = Path("gestos")
    
    print("Iniciando limpieza automática...") # Se inicia la limpieza automática
    # Elimina carpetas con menos de 1 secuencia
    delete_bad_folders(gestos_dir, min_sequences=1)
    
    print("Limpieza completada.") # Se indica que la limpieza ha finalizado
