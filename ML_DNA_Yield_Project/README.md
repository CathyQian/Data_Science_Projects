### Problem Description:
A small startup named X is focused on synthesizing genes based on customized oligonucleotides. Theoretically, each input oligonucleotide sequence would output one gene. In practice, due to the complexity of the synthetic procedure, they usually get a syntheses failure rate, defined as 1 - practical yield of genes/theoretical yield of genes*100%, of around 10% for any batches of input oligonucleotide sequences. However, for their recent batch of input oligonucleotide sequences, they noticed a siginificant jump in the failure rate to around 20%. X reached out to me for help. They wanted to figure out the key features that are responsible for the doubled failure rate.

### Data
The data provided (base_data.csv) contained around 2000 distinct oligonucleotide sequences they inputted for syntheses, of which around 20% failed in producing the desired genes. Each oligonucleotide sequence contains the following features:

- rec: ID number for each input oligonucleotide
- length: number of nucleotides (smallest unit of oligonucleotide sequence) in the input oligonucleotide
- mfe:minimum free energy of the input oligonucleotide. The general rule is the smaller the minimum free energy, the more stable the DNA sequence.
- unpaired:number of unpaired nucleotide. Each oligonucleotide is made of a chain of four nucleotides (A, T, G, C) linked in a specific order. Here A and T can be paired up while G and C can be paired up at certain geometry and energy state.
- paired: number of paried AT or GC nucleotides
- unpairedGC: number of unpaired G and C nucleotides
- pairedGC:number of paired G and C nucleotides
- stack: number of stacks in the input oligonucleotides
- hairpin: number of hairpins(loop intramolecular base pairing) in the input oligonucleotides
- interior_loop: number of interior loop in the input oligonucleotides
- bulge: number of bulge in the input oligonucleotides
- multi_loop: number of loops in the input oligonucleotides
- external:number of external terminals in the input oligonucleotides
- success: binary value 0 or 1 indicating either the syntheses failed or succeeded.

### Methods and Results
Below I used statistical methods to analyze the difference between different features of failed and succeeded oligonucleotides. Next, I used machine learning method to figure out the key features that is related to the failure of the syntheses. 

All above analysis are done in Jupyter Notebook.

Given that this data is from one of my clients, it is not shared in public. If you would like to know more about this project and the data, please feel free to contact me via zhaoxiaq@gmail.com.
