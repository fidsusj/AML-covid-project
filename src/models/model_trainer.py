def train_models(run_from_scratch, df_preprocessed):
    parent_sequences = df_preprocessed["parent"].tolist()
    child_sequences = df_preprocessed["child"].tolist()
    chars = range(216)  # TODO: durchreichen vocab size
