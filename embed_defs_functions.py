import json
from def_proj_functions import load_semeval_df, get_wordtransformer_model, get_wordtransformer_embeddings, get_positions
import pandas as pd
import numpy as np



def get_definitions(language, model_name):
    path_to_defs = '/Users/acw747/Projects/definition_projection/definitions'
    with open(f'{path_to_defs}/{language}_definitions_by_{model_name}.json', 'r') as f:
        definitions = json.load(f)

    # process the defs - remove any duplicates
    for word in definitions:
        definitions[word] = list(set([definition.strip() for definition in definitions[word]]))
        # add word: to the start of each definition if not already present
        for i, definition in enumerate(definitions[word]):
            if not definition.lower().startswith(word.split('_')[0].lower() + ':'):
                definitions[word][i] = f"{word.split('_')[0]}: {definition.strip()}"
        
    return definitions

def embed_definitions(tok, model, definitions):
    """
    definitions: dict of {word: [definition1, definition2, ...]}
    """
    lang_def_embeddings = {}
    for word, def_list in definitions.items():
        word = word.split('_')[0]  # take the first part before underscore
        def_embs_list = []
        for i, definition in enumerate(def_list):
            # find the position of the word in the definition (case insensitive)
            char_pos = get_positions(definition, word)        
            def_lower = definition.lower()  
            # emb = get_XLL_focused_sentence_embedding_from_pos(def_lower, model, char_pos)
            emb = get_wordtransformer_embeddings((tok, model), [definition], [char_pos], batch_size=1, max_length=128)

            def_embs_list.append(emb) 
        lang_def_embeddings[word] = def_embs_list
    return lang_def_embeddings

def compute_spearman_correlation(df, results, metric_cols=None, metrics=None):
    df = df[[col for col in df.columns if 'pca' not in col]]
    df['words'] = df['words'].apply(lambda x: x.split('_')[0])
    merged_df = pd.merge(df, results, on='words')

    if metric_cols is None:
        metric_cols = [
            'apd_full','prt_full','amd_full',
            'apd_def','prt_def','amd_def',
            'apd_pca','prt_pca','amd_pca'
        ]

    potential_metrics = ['amd1','amd2', 'amd_mean', 'amd_min', 'amd_max', 'amd_diff', 'amd_sym','amd_opt_sym','amd_hmean', 'amd_cos','amd','apd','prt', 'amd_sym']
    if metrics is not None:
        potential_metrics = metrics
    
    scores = {}
    for col in metric_cols:
        # metric, space = col.split('_', 1)  # e.g. "apd_full" -> ("apd", "full")
        # find the metric 
        metric = next((m for m in potential_metrics if col.startswith(m)), None)
        space = col[len(metric)+1:]  # get the rest of the string after metric

        spearman_corr = merged_df['graded_scores'].corr(merged_df[col], method='spearman')
        value = float(np.round(spearman_corr, 3))

        # print(f"{metric} ({space}): {value}")
        scores[(metric, space)] = value

    return scores