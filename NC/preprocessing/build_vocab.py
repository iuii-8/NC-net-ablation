__author__ = "Yuyu Luo / compatibility patch"

import csv
from collections import Counter
from pathlib import Path

import torch

try:
    from torchtext.data import Field, TabularDataset, BucketIterator
    TORCHTEXT_AVAILABLE = True
except Exception:
    Field = None
    TabularDataset = None
    BucketIterator = None
    TORCHTEXT_AVAILABLE = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleVocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))


class SimpleField:
    def __init__(self, init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>', lower=True):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.lower = lower
        self.vocab = None

    def preprocess(self, text):
        text = str(text)
        if self.lower:
            text = text.lower()
        return text.split(' ')

    def build_vocab_from_token_lists(self, token_lists, min_freq=2):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        specials = [self.unk_token, self.pad_token, self.init_token, self.eos_token]
        vocab_tokens = list(specials)
        for token, freq in counter.items():
            if freq >= min_freq and token not in vocab_tokens:
                vocab_tokens.append(token)
        self.vocab = SimpleVocab(vocab_tokens)

    def numericalize(self, tokens):
        if self.vocab is None:
            raise ValueError('Vocabulary has not been built.')
        return [self.vocab.stoi.get(tok, self.vocab.stoi[self.unk_token]) for tok in tokens]


class SimpleExample:
    def __init__(self, src, trg, tok_types):
        self.src = src
        self.trg = trg
        self.tok_types = tok_types


class SimpleBatch:
    def __init__(self, examples, src_field, tok_field, device):
        src_sequences = []
        trg_sequences = []
        tok_sequences = []

        for ex in examples:
            src_tokens = [src_field.init_token] + src_field.preprocess(ex.src) + [src_field.eos_token]
            trg_tokens = [src_field.init_token] + src_field.preprocess(ex.trg) + [src_field.eos_token]
            tok_tokens = [tok_field.init_token] + tok_field.preprocess(ex.tok_types) + [tok_field.eos_token]

            src_sequences.append(src_field.numericalize(src_tokens))
            trg_sequences.append(src_field.numericalize(trg_tokens))
            tok_sequences.append(tok_field.numericalize(tok_tokens))

        self.src = pad_sequences(src_sequences, src_field.vocab.stoi[src_field.pad_token], device)
        self.trg = pad_sequences(trg_sequences, src_field.vocab.stoi[src_field.pad_token], device)
        self.tok_types = pad_sequences(tok_sequences, tok_field.vocab.stoi[tok_field.pad_token], device)


class SimpleIterator:
    def __init__(self, examples, batch_size, src_field, tok_field, device):
        self.examples = examples
        self.batch_size = batch_size
        self.src_field = src_field
        self.tok_field = tok_field
        self.device = device

    def __iter__(self):
        for i in range(0, len(self.examples), self.batch_size):
            yield SimpleBatch(
                self.examples[i:i + self.batch_size],
                self.src_field,
                self.tok_field,
                self.device,
            )

    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size


def pad_sequences(sequences, pad_idx, target_device):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long, device=target_device)


def read_ncnet_csv(path):
    examples = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(
                SimpleExample(
                    src=row['source'],
                    trg=row['labels'],
                    tok_types=row['token_types'],
                )
            )
    return examples


def read_db_information(path):
    token_lists = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_lists.append(str(row.get('table', '')).lower().split(' '))
            token_lists.append(str(row.get('column', '')).lower().split(' '))
            token_lists.append(str(row.get('value', '')).lower().split(' '))
    return token_lists


def build_vocab_without_torchtext(data_dir, db_info, batch_size, max_input_length):
    data_dir = Path(data_dir)
    train_data = read_ncnet_csv(data_dir / 'train.csv')
    valid_data = read_ncnet_csv(data_dir / 'dev.csv')
    test_data = read_ncnet_csv(data_dir / 'test.csv')

    SRC = SimpleField()
    TOK_TYPES = SimpleField()

    src_token_lists = []
    tok_type_lists = []

    for dataset in [train_data, valid_data, test_data]:
        for ex in dataset:
            src_token_lists.append(SRC.preprocess(ex.src))
            src_token_lists.append(SRC.preprocess(ex.trg))
            tok_type_lists.append(TOK_TYPES.preprocess(ex.tok_types))

    src_token_lists.extend(read_db_information(db_info))

    SRC.build_vocab_from_token_lists(src_token_lists, min_freq=2)
    TOK_TYPES.build_vocab_from_token_lists(tok_type_lists, min_freq=2)
    TRG = SRC

    train_iterator = SimpleIterator(train_data, batch_size, SRC, TOK_TYPES, device)
    valid_iterator = SimpleIterator(valid_data, batch_size, SRC, TOK_TYPES, device)
    test_iterator = SimpleIterator(test_data, batch_size, SRC, TOK_TYPES, device)

    return SRC, TRG, TOK_TYPES, batch_size, train_iterator, valid_iterator, test_iterator, max_input_length


def build_vocab_with_torchtext(data_dir, db_info, batch_size, max_input_length):
    def tokenizer(text):
        return text.split(' ')

    SRC = Field(tokenize=tokenizer,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TOK_TYPES = Field(tokenize=tokenizer,
                      init_token='<sos>',
                      eos_token='<eos>',
                      lower=True,
                      batch_first=True)

    train_data, valid_data, test_data = TabularDataset.splits(
        path=data_dir, format='csv', skip_header=True,
        train='train.csv', validation='dev.csv', test='test.csv',
        fields=[
            ('tvBench_id', None),
            ('db_id', None),
            ('chart', None),
            ('hardness', None),
            ('query', None),
            ('question', None),
            ('vega_zero', None),
            ('mentioned_columns', None),
            ('mentioned_values', None),
            ('query_template', None),
            ('src', SRC),
            ('trg', SRC),
            ('tok_types', TOK_TYPES)
        ])

    db_information = TabularDataset(
        path=db_info,
        format='csv',
        skip_header=True,
        fields=[
            ('table', SRC),
            ('column', SRC),
            ('value', SRC)
        ]
    )

    SRC.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)
    TRG = SRC
    TOK_TYPES.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), sort=False,
        batch_size=batch_size,
        device=device)

    return SRC, TRG, TOK_TYPES, batch_size, train_iterator, valid_iterator, test_iterator, max_input_length


def build_vocab(data_dir, db_info, batch_size, max_input_length):
    if TORCHTEXT_AVAILABLE:
        return build_vocab_with_torchtext(data_dir, db_info, batch_size, max_input_length)

    print('[INFO] torchtext.data is unavailable; using built-in compatibility vocabulary loader.')
    return build_vocab_without_torchtext(data_dir, db_info, batch_size, max_input_length)
