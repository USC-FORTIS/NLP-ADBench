import os
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor

def run_python_script(filepath):
    output_dir = "out" 
    os.makedirs(output_dir, exist_ok=True)  
    output_file = os.path.join(output_dir, f"{os.path.basename(filepath)}_output.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        subprocess.run(["python", filepath], stdout=f, stderr=subprocess.STDOUT)
    
    print(f"Completed execution of {filepath}. Output redirected to {output_file}.")

def run_benchmark_scripts(directory):
    benchmark_files = glob.glob(f'{directory}/*benchmark*.py')
    
    with ProcessPoolExecutor() as executor:
        executor.map(run_python_script, benchmark_files)

if __name__ == '__main__':
    run_benchmark_scripts('.')
