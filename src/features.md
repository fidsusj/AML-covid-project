## Features to be implemented

Documentation:
- Add motivation section to the README
- Provide Docker image parallel to setup guide
- Keep conda requirements up to date
- Add [PyTorch Tutorial](https://github.com/aladdinpersson/Machine-Learning-Collection) to references

Create Dataset:
- Improve logic how parent-child sequences are selected (Phylogenetic analysis/tree, Levenshtein distance, read papers (like https://science.sciencemag.org/content/371/6526/284)) -> We need temporal relations
- Bring more variety into the data (currently the sequences only vary at the beginning and the end)
- Split the data into training, test and validation dataset on .csv file level
- After training, create pairs of:
  - True parent and child pairs
  - True Parent and predicted child pairs
  - True but unrelated parent child pairs

Preprocessing:
- Minimal frequency to be added to the vocabulary?

Discriminator:
- Linear layer should use the same weight as the embedding layer of the generator
- Flatten MLP input? -> Shape (batch_size, sequence_length * embedding_size) or (batch_size, sequence_length, embedding_size)

Generator:
- DNA2Vec instead of PyTorchs nn.Embedding?
- Beam search instead of greedy decoding?

Quality:
- Delete legacy and template code

Training:
- Do the training
- Does model.eval() set the dropouts off in our custom nn.Module?
- How to select the hyperparameters? -> Read MutaGAN paper
- Where to train for several hours? Paperspace?
- Use diff-match package from Google
- Use a replay buffer (see MutaGAN paper)?
- Avoid mode collapse
- Kullback-Leibler divergence instead of cross entropy loss or both?
- Scheduler for GAN?

Evaluation:
- Define evaluation criteria (MutaGAN paper, BLEU score only?, ...)