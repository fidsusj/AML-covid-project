## Features to be implemented

Documentation:
- Provide Docker image parallel to setup guide
- Keep conda requirements up to date

Create Dataset:
- Split the data into training, test and validation dataset on .csv file level

Preprocessing:
- Minimal frequency to be added to the vocabulary?

Discriminator:
- Flatten MLP input? -> Shape (batch_size, sequence_length * embedding_size) or (batch_size, sequence_length, embedding_size)

Generator:
- Beam search instead of greedy decoding?

Training:
- Loss of generator for GAN: Wasserstein + sparse categorical cross entropy

## Work to do - Report

General:
- Harmonize bibliography

Related Work:
- Other techniques section?

Nils:
- Example record: genome sequence and metadata
- Read over Felix's chapters regarding domain specific terminology
- Why not prefilter completely equal parent-child genome pairs?

Handover:
- Upload the models
- Add [PyTorch Tutorial](https://github.com/aladdinpersson/Machine-Learning-Collection) to references
- Delete legacy and template code
- Code comments
- Upload code to Mampf?
