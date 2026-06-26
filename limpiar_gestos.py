import shutil
from pathlib import Path

def delete_bad_folders(base_path: Path, min_sequences: int = 1) -> None:
    if not base_path.exists(): # Check if the directory exists
        print(f"Error: Base directory '{base_path}' not found.")
        return

    for target_path in base_path.iterdir():
        if target_path.is_dir(): # Check if the path is a directory
            count = len(list(target_path.glob("*.npy")))
            if count < min_sequences: # Check if the folder has fewer than the minimum required sequences
                print(f"Deleting '{target_path}' ({count} sequences)...")
                shutil.rmtree(target_path) # Delete the folder with fewer than the minimum sequences

if __name__ == "__main__":
    gestos_dir = Path("gestos")
    
    print("Starting automatic cleanup...") # Start the automatic cleanup
    # Delete folders with fewer than 1 sequence
    delete_bad_folders(gestos_dir, min_sequences=1)
    
    print("Cleanup complete.") # Indicate that the cleanup has finished
