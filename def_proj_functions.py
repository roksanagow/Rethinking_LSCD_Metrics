from typing import List, Optional, Sequence, Tuple, Union
import pandas as pd
import torch
import re
import numpy as np

from transformers import AutoTokenizer, AutoModel

import os
from pathlib import Path

def _get_data_root() -> Path:
    root = os.getenv("SEMEVAL_DATA_ROOT")
    if not root:
        raise RuntimeError("SEMEVAL_DATA_ROOT is not set. Put it in .env or your environment.")
    return Path(root)


def create_df_from_graded_changes(path_to_stats):
    df = pd.read_csv(path_to_stats, delimiter='\t')
    keep_df = df[['lemma','change_graded','change_binary']]
    keep_df = keep_df.rename(columns={'lemma':'words', 'change_graded':'graded_scores', 'change_binary':'binary_scores'})
    return keep_df

def load_semeval_df(lang: str):

    SEMEVAL_DATA_ROOT = _get_data_root()
    """Return the correct dataframe for the given semeval language."""
    
    # direct CSV paths
    csv_paths = {
        'english': f'{SEMEVAL_DATA_ROOT}/semeval2020_ulscd_eng/en_definitions.csv',
        'german': f'{SEMEVAL_DATA_ROOT}/semeval2020_ulscd_ger/de_definitions.csv',
        'swedish': f'{SEMEVAL_DATA_ROOT}/semeval2020_ulscd_swe/swe_definitions.csv',
        'latin': f'{SEMEVAL_DATA_ROOT}/semeval2020_ulscd_lat/lat_definitions.csv',
        'spanish': f'{SEMEVAL_DATA_ROOT}/dwug_es/es_definitions.csv',
    }
    
    # special handlers that require custom logic
    special_handlers = {
        'chinese': lambda: create_df_from_graded_changes(
            f'{SEMEVAL_DATA_ROOT}/chiwug/stats/opt/stats_groupings.csv'
        ),
        
        'norwegian_1': lambda: create_df_from_graded_changes(
            f'{SEMEVAL_DATA_ROOT}/nor_dia_change/subset1/stats/stats_groupings.tsv'
        ),
        
        'norwegian_2': lambda: create_df_from_graded_changes(
            f'{SEMEVAL_DATA_ROOT}/nor_dia_change/subset2/stats/stats_groupings.tsv'
        ),
    }
    
    # case 1: simple CSV load
    if lang in csv_paths:
        return pd.read_csv(csv_paths[lang])
    
    # case 2: special handler
    if lang in special_handlers:
        return special_handlers[lang]()
    
    # unknown language
    raise ValueError(f"Unknown semeval language: {lang}")

CharSpan = Union[Tuple[int, int], List[int], str]  # (start,end) OR [start,end] OR "start:end"

name_map = {
        "xll": "pierluigic/xl-lexeme",
        "xlmr": "FacebookAI/xlm-roberta-large",
        "multilingual-e5": "intfloat/multilingual-e5-large",
        "roberta": "FacebookAI/roberta-large",
        "rembert": "google/rembert",
        "mmbert": "jhu-clsp/mmBERT-base",
        # Monolingual models for each language:
        "german_GBERT": "deepset/gbert-large",
        "latin_BammanBurns": "ashleygong03/bamman-burns-latin-bert",
        "spanish_BERT_WWM": "dccuchile/bert-base-spanish-wwm-cased",
        "chinese_RoBERTa_WWM": "hfl/chinese-roberta-wwm-ext-large",
        "norwegian_NB-BERT": "NbAiLab/nb-bert-large",
        "swedish_MegatronBERT": "KBLab/megatron-bert-large-swedish-cased-165k",
    }

def get_wordtransformer_model(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Returns a HF (tokenizer, model) tuple for the given model name.

    Supported names (from your file):
      - "xll"             -> pierluigic/xl-lexeme
      - "xlmr"            -> FacebookAI/xlm-roberta-large
      - "multilingual-e5" -> intfloat/multilingual-e5-large
      - "roberta"         -> FacebookAI/roberta-large

    You can also pass a HuggingFace repo id or a local path directly.
    """
    resolved = name_map.get(model_name, model_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device)
    print("Resolved model name/path:", resolved, "on device:", device)


    tok = AutoTokenizer.from_pretrained(resolved, use_fast=True)
    mdl = AutoModel.from_pretrained(resolved)
    mdl.to(device)
    mdl.eval()
    return tok, mdl

def _normalize_span(span: CharSpan) -> Tuple[int, int]:
    if isinstance(span, str):
        a, b = span.split(":")
        return int(a), int(b)
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return int(span[0]), int(span[1])
    raise ValueError(f"Unsupported span format: {span!r}. Expected (start,end) or 'start:end'.")


def _extract_hf_parts(model) -> Tuple[AutoTokenizer, torch.nn.Module, torch.device]:
    """
    Supports:
      - A (tokenizer, model) tuple
    """
    if isinstance(model, tuple) and len(model) == 2:
        tok, mdl = model
        dev = next(mdl.parameters()).device
        return tok, mdl, dev

    # Try common wrapper attribute names
    tok = getattr(model, "tokenizer", None) or getattr(model, "tok", None)
    mdl = getattr(model, "model", None) or getattr(model, "encoder", None) or getattr(model, "transformer", None)

    if tok is None or mdl is None:
        raise TypeError(
            "Couldn't find tokenizer/model inside the provided `model`. "
            "Pass (tokenizer, hf_model)."
        )

    dev = next(mdl.parameters()).device
    return tok, mdl, dev


@torch.no_grad()
def wordtransformer_embeddings_from_char_positions(
    model,
    sentences: Sequence[str],
    char_positions: Sequence[CharSpan],
    *,
    batch_size: int = 32,
    pooling: str = "mean",   # "mean" or "first"
    max_length: Optional[int] = 256,
) -> torch.Tensor:
    """
    Given:
      - model: (tokenizer, hf_model)
      - sentences: list[str]
      - char_positions: list of (start,end) character spans (or "start:end")

    Returns:
      - Tensor of shape [len(sentences), hidden_dim] with the *span* embedding.

    Span-to-token alignment uses the fast tokenizer offsets mapping and selects tokens
    whose character offsets overlap the [start,end) span.
    """
    if len(sentences) != len(char_positions):
        raise ValueError(f"len(sentences)={len(sentences)} != len(char_positions)={len(char_positions)}")

    tok, mdl, dev = _extract_hf_parts(model)
    spans = [_normalize_span(s) for s in char_positions]

    out: List[torch.Tensor] = []

    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i : i + batch_size]
        batch_spans = spans[i : i + batch_size]

        enc = tok(
            batch_sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        offsets = enc.pop("offset_mapping")  # [B, T, 2] on CPU
        enc = {k: v.to(dev) for k, v in enc.items()}

        hs = mdl(**enc).last_hidden_state  # [B, T, H]
        attn = enc.get("attention_mask", None)

        for b in range(hs.size(0)):
            start, end = batch_spans[b]
            off = offsets[b].tolist()  # [[s,e], ...]

            # pick tokens overlapping the span
            token_ids: List[int] = []
            for t, (s_off, e_off) in enumerate(off):
                # skip special tokens which often have (0,0) offsets
                if s_off == 0 and e_off == 0:
                    continue
                if e_off <= start:
                    continue
                if s_off >= end:
                    continue
                token_ids.append(t)

            if not token_ids:
                # Fallback: try to choose the closest token by start position
                best_t = None
                best_dist = 10**18
                for t, (s_off, e_off) in enumerate(off):
                    if s_off == 0 and e_off == 0:
                        continue
                    d = min(abs(s_off - start), abs(e_off - start))
                    if d < best_dist:
                        best_dist = d
                        best_t = t
                token_ids = [best_t] if best_t is not None else [0]

            token_ids_t = torch.tensor(token_ids, device=dev, dtype=torch.long)

            vecs = hs[b].index_select(0, token_ids_t)  # [K, H]

            if attn is not None:
                # ensure we don't pool padded tokens by mistake
                mask = attn[b].index_select(0, token_ids_t).unsqueeze(-1).float()
                vecs = vecs * mask
                denom = mask.sum().clamp_min(1.0)
            else:
                denom = torch.tensor(vecs.size(0), device=dev, dtype=torch.float)

            if pooling == "first":
                emb = vecs[0]
            elif pooling == "mean":
                emb = vecs.sum(dim=0) / denom
            else:
                raise ValueError("pooling must be 'mean' or 'first'")

            out.append(emb.detach().cpu())

    return torch.stack(out, dim=0)

def get_wordtransformer_embeddings(
    model,
    sentences: Sequence[str],
    char_positions: Sequence[CharSpan],
    **kwargs,
) -> torch.Tensor:
    """
    Alias: returns span embeddings (one per sentence) from character spans.
    """
    return wordtransformer_embeddings_from_char_positions(model, sentences, char_positions, **kwargs)



def compute_spearman_correlation(df, results, metrics=['amd_sym','amd_opt_sym','amd','apd','prt'], spaces=['full','def','pca','rand']):
    df = df[[col for col in df.columns if 'pca' not in col]] # Drop PCA columns from the gold-standard df because some had older results - this is not the results!
    df['words'] = df['words'].apply(lambda x: x.split('_')[0])
    merged_df = pd.merge(df, results, on='words')

    # define metric_cols based on metrics and spaces
    metric_cols = []
    for metric in metrics:
        for space in spaces:
            col_name = f"{metric}_{space}"
            if col_name in merged_df.columns:
                metric_cols.append(col_name)
    
    scores = {}
    for col in metric_cols:
        # find the metric 
        metric = next((m for m in metrics if col.startswith(m)), None)
        space = col[len(metric)+1:]  # get the rest of the string after metric

        spearman_corr = merged_df['graded_scores'].corr(merged_df[col], method='spearman')
        value = float(np.round(spearman_corr, 3))

        # print(f"{metric} ({space}): {value}")
        scores[(metric, space)] = value

    return scores

######## Find word positions with edit distance if not provided ########

def edit_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def find_position_of_similar_word(word, sentence):
    """
    Find the start and end positions of the word in the sentence that has the smallest Levenshtein distance
    to the target word. If multiple words have the same smallest distance, the first one is returned.

    Args:
        word (str): The target word to find a similar match for.
        sentence (str): The sentence to search within.

    Returns:
        tuple: A tuple containing the start and end positions (start_pos, end_pos) of the similar word.
               If no match is found, returns None.
    """
    # Find all whole words in the sentence
    matches = list(re.finditer(r'\b\w+\'?\w*\b', sentence))
    min_distance = float('inf')
    best_position = None

    # Iterate through each word in the sentence
    for match in matches:
        sentence_word = match.group(0)
        distance = edit_distance(word, sentence_word)
        
        # Update the best position if a smaller distance is found
        if distance < min_distance:
            min_distance = distance
            best_position = (match.start(), match.end())
        # If the same minimum distance is found, retain the first occurrence
        elif distance == min_distance:
            pass

    return best_position

def get_positions(sentence, word):
    """
    Finds the start and end character positions of a word in a sentence. If the word is found as a complete word,
    returns its start and end positions. If not found, returns the start and end positions of the word with the
    smallest Levenshtein edit distance from the target word.

    Args:
        sentence (str): The sentence to search within.
        word (str): The word to find the positions of.

    Returns:
        list: A list containing the start and end positions [start_pos, end_pos] of the word or the closest match.
    """
    match = re.search(rf'\b{re.escape(word)}\b', sentence)
    if match:
        start_pos = match.start()
        end_pos = match.end()
    else:
        start_pos, end_pos = find_position_of_similar_word(word, sentence)
    return [start_pos, end_pos]



lang_specific_models = {
    "german": ["deepset/gbert-large", ],
    "latin": ["ashleygong03/bamman-burns-latin-bert",],
    "spanish": ["dccuchile/bert-base-spanish-wwm-cased",],
    "chinese": ["hfl/chinese-roberta-wwm-ext-large",],
    "norwegian_1": ["NbAiLab/nb-bert-large",],    
    "norwegian_2": ["NbAiLab/nb-bert-large",],
    "swedish": ["KBLab/megatron-bert-large-swedish-cased-165k",],
    "english": ["FacebookAI/roberta-large",],
}

lang_specific_model_names = {
    "TUM/GottBERT_large": "german_GottBERT",
    "deepset/gbert-large": "german_GBERT",
    "ashleygong03/bamman-burns-latin-bert": "latin_BammanBurns",
    "PlanTL-GOB-ES/roberta-large-bne": "spanish_RoBERTa_BNE",
    "dccuchile/bert-base-spanish-wwm-cased": "spanish_BERT_WWM",
    "hfl/chinese-roberta-wwm-ext-large": "chinese_RoBERTa_WWM",
    "NbAiLab/nb-bert-large": "norwegian_NB-BERT",
    "KBLab/megatron-bert-large-swedish-cased-165k": "swedish_MegatronBERT",
    "FacebookAI/roberta-large": "roberta",
}

models = ['xll','xlmr','rembert','multilingual-e5','mmbert'] # + monolingual model per language

definition_models = ['gemini25pro']
languages = ['english','german','swedish','latin','spanish','chinese','norwegian_1','norwegian_2']

