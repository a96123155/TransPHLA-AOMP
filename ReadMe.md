# TransMut: Automatically optimize mutated peptide program from the Transformer-derived self-attention model to predict peptide and HLA binding
## webserver: https://issubmission.sjtu.edu.cn/TransPHLA-AOMP/index.html

1. Install pytorch environment on your linux system: 
pip install -r pHLAIformer_simple_requirements.txt
pHLAIformer_simple_requirements.txt is a simplified version of the dependency package selected by the author. If the dependency package is not installed enough, you can check pHLAIformer_requirements.txt to check the version of the corresponding dependency package. The installation of pytorch needs to correspond to your system, GPU and other versions, please check https://pytorch.org/get-started/locally/, we are using GeForce RTX 3080 GPU and CUDA 11.1.

2. Enter the current directory: cd ./TransPHLA-AOMP/

3. python pHLAIformer.py --peptide_file "peptides.fasta" --HLA_file "hlas.fasta" --threshold 0.5 --cut_length 10 --cut_peptide True --output_dir './results/' --output_attention True --output_heatmap True --output_mutation True

**peptides.fasta and hlas.fasta are sample files provided by us, which contain some error debug cases.**

**Parameter:**
- peptide_file: type = str, help = the filename of the .fasta file contains peptides
- HLA_file: type = str, help = the filename of the .fasta file contains sequence
- threshold: type = float, default = 0.5, help = the threshold to define predicted binder, float from 0 - 1, the recommended value is 0.5
- cut_peptide: type = bool, default = True, help = Whether to split peptides larger than cut_length?
- cut_length: type = int, default = 9, help = if there is a peptide sequence length > 15, we will segment the peptide according the length you choose, from 8 - 15
- output_dir: type = str, help = The directory where the output results are stored.
- output_attention, type = bool, default = True, help = Output the mutual influence of peptide and HLA on the binding?
- output_heatmap: type = bool, default = True, help = Visualize the mutual influence of peptide and HLA on the binding?
- output_mutation: type = bool, default = True, help = Whether to perform mutations with better affinity for each sample?