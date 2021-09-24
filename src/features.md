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
- genome or strang
- sequennce or genome

Related Work:
- Other techniques section?

Evaluation:
- Mode collapse reason: dataset, wrong loss function, not trained enough, LR schedule, ...?

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
