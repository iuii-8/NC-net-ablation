__author__ = "Yuyu Luo / AutoTemplate extension"

"""Evaluate ncNet with AutoTemplate-predicted chart templates.

This script is based on the official test.py, but it separates evaluation by
using the original dataset_final/test.csv to identify whether each sample was
originally a w/o-template or with-template row.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.VisAwareTranslation import (
    get_all_table_columns,
    postprocessing,
    translate_sentence_with_guidance,
)
from model.Model import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from preprocessing.build_vocab import build_vocab


def normalize_query(query):
    return " ".join(str(query).replace('"', "'").split()).lower()


def is_original_without_template(source):
    return "[t]" in str(source).lower()


def build_ncnet_model(opt, SRC, TRG, TOK_TYPES, my_max_length, device):
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    hid_dim = 256
    enc_layers = 3
    dec_layers = 3
    enc_heads = 8
    dec_heads = 8
    enc_pf_dim = 512
    dec_pf_dim = 512
    enc_dropout = 0.1
    dec_dropout = 0.1

    enc = Encoder(
        input_dim,
        hid_dim,
        enc_layers,
        enc_heads,
        enc_pf_dim,
        enc_dropout,
        device,
        TOK_TYPES,
        my_max_length,
    )
    dec = Decoder(
        output_dim,
        hid_dim,
        dec_layers,
        dec_heads,
        dec_pf_dim,
        dec_dropout,
        device,
        my_max_length,
    )

    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]
    model = Seq2Seq(enc, dec, SRC, src_pad_idx, trg_pad_idx, device).to(device)
    model.load_state_dict(torch.load(opt.model, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="test_autotemplate.py")
    parser.add_argument("-model", default="./save_models/trained_model.pt")
    parser.add_argument("-data_dir", default="./dataset/dataset_final/")
    parser.add_argument("-db_info", default="./dataset/database_information.csv")
    parser.add_argument("-original_test_data", default="./dataset/dataset_final/test.csv")
    parser.add_argument("-autotemplate_test_data", default="./dataset/dataset_autotemplate/test.csv")
    parser.add_argument("-db_schema", default="./dataset/db_tables_columns.json")
    parser.add_argument("-db_tables_columns_types", default="./dataset/db_tables_columns_types.json")
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-max_input_length", type=int, default=128)
    parser.add_argument("-show_progress", default=False)
    parser.add_argument("-limit", type=int, default=None, help="Only evaluate the first N rows for a smoke test.")
    parser.add_argument(
        "-output_errors",
        default="./dataset/dataset_autotemplate/autotemplate_errors.csv",
        help="Path to save failed AutoTemplate-origin samples for error analysis.",
    )
    opt = parser.parse_args()
    print("the input parameters: ", opt)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, _, _, _, _, my_max_length = build_vocab(
        data_dir=opt.data_dir,
        db_info=opt.db_info,
        batch_size=opt.batch_size,
        max_input_length=opt.max_input_length,
    )
    print("------------------------------\n| Build vocab end ... | \n------------------------------")

    print("------------------------------\n| Build and load ncNet ... | \n------------------------------")
    ncNet = build_ncnet_model(opt, SRC, TRG, TOK_TYPES, my_max_length, device)

    print("------------------------------\n| Testing AutoTemplate ... | \n------------------------------")
    db_tables_columns = get_all_table_columns(opt.db_schema)
    db_tables_columns_types = get_all_table_columns(opt.db_tables_columns_types)

    original_df = pd.read_csv(opt.original_test_data)
    autotemplate_df = pd.read_csv(opt.autotemplate_test_data)

    if len(original_df) != len(autotemplate_df):
        raise ValueError(
            f"Row count mismatch: original={len(original_df)}, autotemplate={len(autotemplate_df)}"
        )

    if opt.limit is not None:
        original_df = original_df.head(opt.limit).reset_index(drop=True)
        autotemplate_df = autotemplate_df.head(opt.limit).reset_index(drop=True)
        print(f"Limit enabled: evaluating first {len(autotemplate_df)} rows only.")

    autotemplate_origin_cnt = 0
    autotemplate_origin_match = 0
    original_with_template_cnt = 0
    original_with_template_match = 0
    overall_cnt = 0
    overall_match = 0
    error_rows = []

    for index, row in tqdm(autotemplate_df.iterrows(), total=len(autotemplate_df)):
        original_row = original_df.iloc[index]
        originally_without_template = is_original_without_template(original_row["source"])

        try:
            gold_query = str(row["labels"]).lower()
            src = str(row["source"]).lower()
            tok_types = row["token_types"]
            gold_tokens = gold_query.split(" ")
            table_name = gold_tokens[gold_tokens.index("data") + 1]

            translation, _, _ = translate_sentence_with_guidance(
                row["db_id"],
                table_name,
                src,
                SRC,
                TRG,
                TOK_TYPES,
                tok_types,
                SRC,
                ncNet,
                db_tables_columns,
                db_tables_columns_types,
                device,
                my_max_length,
                show_progress=opt.show_progress,
            )

            pred_query = " ".join(translation).replace(" <eos>", "").lower()
            pred_query = postprocessing(gold_query, pred_query, True, src)
            matched = normalize_query(gold_query) == normalize_query(pred_query)

            overall_cnt += 1
            overall_match += int(matched)

            if originally_without_template:
                autotemplate_origin_cnt += 1
                autotemplate_origin_match += int(matched)
                if not matched:
                    error_rows.append(
                        {
                            "index": index,
                            "tvBench_id": row.get("tvBench_id", ""),
                            "db_id": row.get("db_id", ""),
                            "question": row.get("question", ""),
                            "source": row.get("source", ""),
                            "gold": gold_query,
                            "pred": pred_query,
                        }
                    )
            else:
                original_with_template_cnt += 1
                original_with_template_match += int(matched)

        except Exception as ex:
            print(f"error at row {index}: {ex}")

    autotemplate_acc = autotemplate_origin_match / autotemplate_origin_cnt if autotemplate_origin_cnt else 0
    original_template_acc = original_with_template_match / original_with_template_cnt if original_with_template_cnt else 0
    overall_acc = overall_match / overall_cnt if overall_cnt else 0

    print("========================================================")
    print("AutoTemplate-origin samples:", autotemplate_origin_cnt)
    print("Original with-template samples:", original_with_template_cnt)
    print("Overall evaluated samples:", overall_cnt)
    print("--------------------------------------------------------")
    print("ncNet + AutoTemplate on original w/o-template rows:", autotemplate_acc)
    print("ncNet with original template rows:", original_template_acc)
    print("ncNet AutoTemplate overall:", overall_acc)
    print("========================================================")

    output_errors = Path(opt.output_errors)
    output_errors.parent.mkdir(parents=True, exist_ok=True)
    with open(output_errors, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["index", "tvBench_id", "db_id", "question", "source", "gold", "pred"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)
    print(f"Saved error analysis rows to {output_errors}")


if __name__ == "__main__":
    main()
