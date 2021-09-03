# TransMut: Automatically optimize mutated peptide program from the Transformer-derived self-attention model to predict peptide and HLA binding
## webserver: https://issubmission.sjtu.edu.cn/TransPHLA-AOMP/index.html

### 1. All software dependencies and oprating system:
The dependencies is the pytorch environment on linux system, the operating system is CentOS Linux release 7.7.1908.

### 2. Any required non-standard hardware: The GPU we used GeForce RTX 3080 GPU and CUDA 11.1.

### 3. Software version: This is the 1th version, this verision has been tested on indenpendent set, external set, neoantigen data, and HPV vaccine data.

### 4. Installation guide:

(1) Instructions:

- Install pytorch environment on your linux system: pip install -r pHLAIformer_simple_requirements.txt
pHLAIformer_simple_requirements.txt is a simplified version of the dependency package selected by the author. If the dependency package is not installed enough, you can check pHLAIformer_requirements.txt to check the version of the corresponding dependency package. The installation of pytorch needs to correspond to your system, GPU and other versions, please check https://pytorch.org/get-started/locally/, we are using GeForce RTX 3080 GPU and CUDA 11.1.

- Enter the current directory: cd ./TransPHLA-AOMP/

(2) Typical install time on a "normal" destop computer: The install time for this code cannot be determinated, it depends on the network situation and computer configuration. The user-friendly webserver is no need to be installed, so there is no install time.

### 5. Demo:

(1) Instructions to run on data:

- Code: python pHLAIformer.py --peptide_file "peptides.fasta" --HLA_file "hlas.fasta" --threshold 0.5 --cut_length 10 --cut_peptide True --output_dir './results/' --output_attention True --output_heatmap True --output_mutation True

- Webserver: click the Example button on the https://issubmission.sjtu.edu.cn/TransPHLA-AOMP/index.html and click the submit button. 

(2) Excepted output: The output can be seen in https://issubmission.sjtu.edu.cn/TransPHLA-AOMP/tutorials.html

(3) Expected run time for demo on a "normal" desktop computer: 

- Code: The prediction of 170000 peptide-HLA binders only needs 2 minutes on a CPU.

- Webserver: For one sample, if we choose to generate all optional output, we need approximately 30 

### 6. Instructions for use:

(1) How to run the software on your data: https://issubmission.sjtu.edu.cn/TransPHLA-AOMP/tutorials.html

(2) Reproduction instructions: You can input the same data many times on the webserver to obtain the reproduction results.

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
