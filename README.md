## Surface Tension Prediction with RoBERTa Model

This repository utilizes the RoBERTa model to predict surface tension based on a SMILES string, which serves as a string representation for a molecular structure. The computation of the SMILES string is facilitated using the RDKit tool.

### Pretraining Details
The RoBERTa model has been pretrained with a custom vocabulary specifically designed for SMILES strings. The training dataset comprises a substantial 77 million SMILES strings. The pretraining process employed masked language modeling techniques to enhance the model's understanding of molecular structures.

### Fine-Tuning on Experimental Data
Following pretraining, the model underwent fine-tuning using experimental surface tension data. This dataset includes SMILES strings paired with their corresponding surface tension values.

Feel free to explore the code and experiment with surface tension predictions using the RoBERTa model!
