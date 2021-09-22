## Features to be implemented

Documentation:
- Provide Docker image parallel to setup guide
- Keep conda requirements up to date
- Picture of project structure showing the git ignored folders
- Upload the models
- Add [PyTorch Tutorial](https://github.com/aladdinpersson/Machine-Learning-Collection) to references

Create Dataset:
- Split the data into training, test and validation dataset on .csv file level

Preprocessing:
- Minimal frequency to be added to the vocabulary?

Discriminator:
- Linear layer should use the same weight as the embedding layer of the generator
- Flatten MLP input? -> Shape (batch_size, sequence_length * embedding_size) or (batch_size, sequence_length, embedding_size)

Generator:
- Beam search instead of greedy decoding?

Quality:
- Delete legacy and template code

Training:
- Does model.eval() set the dropouts off in our custom nn.Module?
- Use a replay buffer (see MutaGAN paper)?
- Avoid mode collapse
- Kullback-Leibler divergence instead of cross entropy loss or both?
- Loss of generator for GAN: binary + sparse categorical cross entropy

Evaluation:
- Define evaluation criteria (MutaGAN paper, BLEU score only?, ...)

For the report:
- Work to do:
  - Reverse the sequences?
  - Hyperparameter tuning -> Read MutaGAN paper and play around (e.g. smaller embedding size)
  - Other evaluation methods -> MutaGAN Generator Evaluation, read its evaluation chapter in general
  - Increase patience for LR schedule
  - Wasserstein loss
  - => Still leads to mode collapse, explain why: dataset, wrong loss function, not trained enough, LR schedule, ...?
  - Use diff-match package from Google
  - Compare performance with those of others in general
  - Does GAN training improve the performance?
  - \# encoder/decoder layers
- Content:
  - Other techniques section?
  - We do not use beam search
  - We don't have true parent, but unrelated child sequences like MutaGAN. Why did MutaGAN use this?
  - Dataset dividable by 3 (length of sequence: 29904)
  - Loss curve does not converge as fast because of the loss curve
  - How many batches are necessary according to MutaGAN and how many epochs, how long does training normally take?
  - Get inspired by motivation from MutaGAN
  - We use teacher forcing and early stopping
  - See comments in LaTeX and harmonize
  - Show correct mutation prediction at the specific position (BLEU score not only high because the parts that never change are correctly "predicted"")
  - Shorten
  - Explain train test mode
- Formal:
  - Update contributions with the chapters written
  - Don't write "we"
  - Smartphone picture: either replace or explain in footnote why this happened
  - Delete "own representation"

Wednesday:
- Other evaluation methods -> MutaGAN Generator Evaluation, read its evaluation chapter in general
=> Write methods of evaluation subchapter in report
- Hyperparameter tuning -> Read MutaGAN paper and play around (e.g. smaller embedding size)
- Reverse the sequences?
- What happens if I restart the training? Also big initial drop?
- Increase patience for LR schedule
- Wasserstein loss feasible?
=> Create BLEU Score table
- => Still leads to mode collapse, explain why: dataset, wrong loss function, not trained enough, LR schedule, ...?
- Use diff-match package from Google
- Compare performance with those of others in general

Thursday:
- Harmonize the report
- And finish off points left open => Into conclusion

Friday:
- Harmonize the report
- Meeting with Nils
