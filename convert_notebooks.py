#!/usr/bin/env python3
"""
Script to convert Jupyter notebooks to Python scripts.
"""
import json
import os
import sys

def convert_notebook_to_script(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to a Python script.
    
    Args:
        notebook_path (str): Path to the Jupyter notebook
        output_path (str, optional): Path to save the Python script. If None,
                                    will use the same name with .py extension.
    
    Returns:
        str: Path to the created Python script
    """
    if output_path is None:
        output_path = os.path.splitext(notebook_path)[0] + '.py'
    
    print(f"Converting {notebook_path} to {output_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract code cells
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    
    # Write to Python script
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write(f'# Converted from {os.path.basename(notebook_path)}\n\n')
        
        for i, cell in enumerate(code_cells):
            # Skip empty cells
            if not cell['source']:
                continue
                
            # Add a comment to separate cells
            f.write(f'# Cell {i+1}\n')
            
            # Write the cell content
            cell_content = ''.join(cell['source'])
            f.write(cell_content)
            
            # Add newlines between cells
            if not cell_content.endswith('\n'):
                f.write('\n')
            f.write('\n')
    
    print(f"Successfully converted {notebook_path} to {output_path}")
    return output_path

def main():
    """
    Main function to convert all notebooks in a directory.
    """
    # Directory containing notebooks
    notebook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl')
    
    # Find all notebook files
    notebook_files = [
        os.path.join(notebook_dir, f) 
        for f in os.listdir(notebook_dir) 
        if f.endswith('.ipynb') and not f.endswith('.ipynb:Zone.Identifier')
    ]
    
    # Convert each notebook
    for notebook_path in notebook_files:
        convert_notebook_to_script(notebook_path)

if __name__ == '__main__':
    main()
