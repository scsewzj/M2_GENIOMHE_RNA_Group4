# rnaid, chain_id1, chain_id2,...
import os
import argparse
"""
Generate a configuration template for RNA chain selection from PDB/CIF files in a given directory, can't specify. Default in the same dir as your inputdir.
"""
def generate_config_template(input_dir):
    valid_files = []
    files = os.listdir(input_dir)
    for file in files:
        if file.endswith('.pdb') or file.endswith('.cif'):
            if file not in valid_files:
                valid_files.append(file) # not allowed to put duplicate files, even same id but in different formats
    with open(os.path.join(input_dir, 'config_template.txt'), 'w') as f:
        for file in valid_files:
            rna_ids = file.split('.')[:-1]
            rna_id = ''
            for i in rna_ids:
                rna_id += i
            f.write(f"{rna_id}, \n")
    print("Configuration template generated: config_template.txt, You can fill in chain IDs accordingly.")
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate configuration template for RNA chain selection. Can't specify the target path. Default in the same dir as your inputdir.")
    argparser.add_argument('input_dir', type=str, help='Directory containing PDB/CIF files.')
    args = argparser.parse_args()
    generate_config_template(args.input_dir)
            