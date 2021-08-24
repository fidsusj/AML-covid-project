## Features to be implemented

Documentation:
- Add motivation section to the README
- Provide Docker image parallel to setup guide
- Keep conda requirements up to date

Create Dataset:
- Improve logic how parent-child sequences are selected (Phylogenetic analysis/tree, Levenshtein distance, read papers (like https://science.sciencemag.org/content/371/6526/284)) -> We need temporal relations
- Bring more variety into the data (currently the sequences only vary at the beginning and the end)
- Split the data into training, test and validation dataset on .csv file level

Preprocessing:
- Numericalize only once

Quality:
- Delete legacy and template code

Training:
- Do the training
- Why is there no output shift during evaluation?
- Does model.eval() set the dropouts off in our custom nn.Module?
- How to select the hyperparameters?
- Where to train for several hours?
- Teacher forcing during training? (Use the true output instead of the predicted one)
- Kullback-Leibler divergence instead of cross entropy loss or both?
- Wasserstein loss for GAN
- Beam search instead of greedy decoding?
- Use diff-match package from Google
- Use a replay buffer?
- DNA2Vec instead of pytorchs nn-Embedding?
- 