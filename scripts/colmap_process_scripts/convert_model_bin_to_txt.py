import argparse
import os
import subprocess
import sys
#python scripts/colmap_process_scripts/convert_model_bin_to_txt.py --input_path data1.15/process/colmap_process_datas/mutlecamers0/distorted/sparse/0
def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP binary model to text model.")
    parser.add_argument("--input_path", required=True, help="Path to the directory containing binary model files (cameras.bin, images.bin, points3D.bin).")
    parser.add_argument("--output_path", help="Path to the output directory for text model files. Defaults to input_path.")
    parser.add_argument("--colmap_executable", default="colmap", help="Path to the colmap executable.")
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path) if args.output_path else input_path
    
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    print(f"Converting model from '{input_path}' to '{output_path}'...")
    
    command = [
        args.colmap_executable, "model_converter",
        "--input_path", input_path,
        "--output_path", output_path,
        "--output_type", "TXT"
    ]
    
    try:
        subprocess.check_call(command)
        print("Conversion successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error running colmap model_converter: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: '{args.colmap_executable}' executable not found. Please ensure COLMAP is installed and in your PATH, or provide the path using --colmap_executable.")
        sys.exit(1)

if __name__ == "__main__":
    main()
