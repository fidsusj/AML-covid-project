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

## Work to do - Coding

Training: 
- Increase patience for LR schedule
- GAN training

Evaluation:
- Calculate Levenshtein distance, BLEU score, sequence true positive rate using Googles diff-match package
- Name concrete nucleotide mutations: manual evaluation for the concrete mutation positions, taking one/two examples
- Mutations count per amino acid chart

## Work to do - Report

General:
- Read the report from top to bottom
- Check comments in LaTeX and harmonize
- Shorten
- Update contributions with the chapters written
- Don't write "we"
- Smartphone picture: either replace or explain in footnote why this happened
- Delete "own representation"
- Harmonize bibliography
- Grammarly
- "MutaGAN paper" or just [6]?

Related Work:
- Other techniques section?

Evaluation:
- Mode collapse reason: dataset, wrong loss function, not trained enough, LR schedule, ...?
- The generated sequences are very close, but not identical to the parent sequences, indicating that the model is augmenting its input to account for the learned model of protein evolution.
- The BLEU score for MutaGAN model is 97.46%, which indicates a high level of precision in sequence generation

Conclusion:
- State open features as future improvements

Handover:
- Picture of project structure showing the git ignored folders
- Upload the models
- Add [PyTorch Tutorial](https://github.com/aladdinpersson/Machine-Learning-Collection) to references
- Delete legacy and template code
- Code comments
- Upload code to Mampf?

## Notes

MutaGAN:
- Pretraining: 72 epochs, but converged far before that
- Training: 350 epochs
