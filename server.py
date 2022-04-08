import os
import csv
import math
import json
import argparse
import numpy as np
from os.path import join as pjoin


#### only needed for cross-origin requests:
# from flask_cors import CORS
import torch
import collections
from tqdm import tqdm
from sklearn.manifold import TSNE
from flask import Flask, send_from_directory, request, Response, redirect, url_for, jsonify

import pdb



app = Flask(__name__)
#### only needed for cross-origin requests:
# CORS(app)

# Variables
data_dir = None
hidden_states = None
interpretation_results = None
flattened_tokens = []
flattened_interactions = []

def load_globals(args):
    """
    Initialize global variables to be used by later function calls
    """
    global hidden_states, interpretation_results, data_dir, flattened_tokens, flattened_interactions
    data_dir = args.data_dir
    hidden_states_path = pjoin(args.data_dir, 'cls_hidden_states.pt')
    hidden_states = torch.load(hidden_states_path)

    # Aggregate partitions
    interpretation_results_path = pjoin(args.data_dir, 'interaction_results.json')
    if not os.path.isfile(interpretation_results_path):
        prediction_root_dir = pjoin('results', 'debertav3-large', 'predictions')
        partition_dirs = [d for d in os.listdir(prediction_root_dir) if d.startswith('dev_')]
        partition_dirs = sorted(partition_dirs, key=lambda x: int(x.split('_')[1]))
        interpretation_results = collections.defaultdict(list)
        for i, d in enumerate(partition_dirs):
            partition_dir = pjoin(prediction_root_dir, d)
            result_file = pjoin(partition_dir, 'interaction_results.json')
            with open(result_file, 'r') as f:
                results = json.load(f)

            for k, v in results.items():
                interpretation_results[k].extend(v)


        with open(interpretation_results_path, 'w+', encoding='utf-8') as f:
            json.dump(interpretation_results, f)

    else:
        with open(interpretation_results_path) as f:
            interpretation_results = json.load(f)

    prediction_root_dir = pjoin('results', 'debertav3-large', 'predictions')
    partition_dirs = [d for d in os.listdir(prediction_root_dir) if d.startswith('dev_')]
    partition_dirs = sorted(partition_dirs, key=lambda x: int(x.split('_')[1]))
    partition_indices = [idx for idx in map(lambda x: int(x.split('_')[1]), partition_dirs)]
    data_indices = []
    for idx in partition_indices:
        data_indices.extend(list(range(idx * 123, idx * 123 + 123)))
    hidden_states = hidden_states[[idx for idx in data_indices if idx < len(hidden_states)]]
    print(hidden_states.shape)
    # results_path = pjoin(args.data_dir, 'interaction_results.json')
    # with open(results_path) as f:
    #     interpretation_results = json.load(f)

    tokens, interactions = interpretation_results['tokens'], interpretation_results['attritions']
    assert len(tokens) == len(interactions)

    for idx in range(len(tokens)):
        flattened_tokens.extend(tokens[idx])
        flattened_interactions.extend(interactions[idx])



####################################### Helper Functions #######################################
def pairwise_cosine_similarity(a, b, eps=1e-8):
    """
    Compute pairwise cosine similarity between two sets of vectors
    (Added eps for numerical stability)
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def normalize(matrix, axis=None):
    """
    Normalize NumPy matrix, across all dimensions by default
    """
    normalized = (matrix - matrix.min(axis=axis)) /\
                 (matrix.max(axis=axis) - matrix.min(axis=axis))
    return normalized

def save_hidden_state_projection(hidden_states, label_mask):

    flattened_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    flattened_label_mask = label_mask.view(-1)

    assert len(flattened_hidden_states) == len(flattened_label_mask)
    n_projections = len(flattened_hidden_states)

    # Create TSNE projections using cosine similarity
    dist = pairwise_cosine_similarity(flattened_hidden_states, flattened_hidden_states).numpy()
    dist = (dist - dist.min()) / (dist.max() - dist.min())

    tsne_model = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000, metric='precomputed', random_state=0)
    tsne_vectors = tsne_model.fit_transform(dist).round(decimals=5)

    # Save formatted output to projection file
    example_ids = np.expand_dims(np.arange(n_projections), axis=1).astype(int)
    label_column = np.expand_dims(flattened_label_mask.numpy(), axis=1).astype(int)
    save_data = np.hstack((example_ids, tsne_vectors, label_column))
    return save_data

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    List, int, int, set -> Tuple[set, "torch.LongTensor"]
    """

    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0

    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[~mask].long()
    return heads, index

def estimate_importance(model, dataloader, measures=['taylor']):

    assert set(measures).issubset(set(['oracle', 'taylor', 'sensitivity', 'lrp']))

    encoder = model.bert if hasattr(model, 'bert') else model.encoder

    n_layers = encoder.config.num_hidden_layers
    n_heads = encoder.config.num_attention_heads
    head_size = int(encoder.config.hidden_size / n_heads)

    importance_scores = {}
    for measure in measures:
        importance_scores[measure] = np.zeros((n_layers, n_heads))

    device = next(model.parameters()).device
    model.train()

    total_loss = 0.

    if 'taylor' in measures or 'sensitivity' in measures or 'lrp' in measures:

        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            logits, loss = output['logits'], output['loss']
            loss.backward(retain_graph=True)

            total_loss += loss.item()

            if 'taylor' in measures :
                for i in range(n_layers):
                    attention = encoder.encoder.layer[i].attention
                    num_attention_heads = attention.self.num_attention_heads

                    pruned_heads = attention.pruned_heads
                    leftover_heads = set(list(range(n_heads))) - pruned_heads

                    for head_idx in leftover_heads:
                        heads, index = find_pruneable_heads_and_indices([head_idx], num_attention_heads, head_size, pruned_heads)
                        index = index.to(device)

                        query_b_grad = (attention.self.query.bias.grad[index] *\
                                        attention.self.query.bias[index]) ** 2
                        query_W_grad = (attention.self.query.weight.grad.index_select(0, index) *\
                                        attention.self.query.weight.index_select(0, index)) ** 2

                        key_b_grad = (attention.self.key.bias.grad[index] *\
                                      attention.self.key.bias[index]) ** 2
                        key_W_grad = (attention.self.key.weight.grad.index_select(0, index) *\
                                      attention.self.key.weight.index_select(0, index)) ** 2

                        value_b_grad = (attention.self.value.bias.grad[index] *\
                                        attention.self.value.bias[index]) ** 2
                        value_W_grad = (attention.self.value.weight.grad.index_select(0, index) *\
                                        attention.self.value.weight.index_select(0, index)) ** 2

                        abs_grad_magnitude = query_b_grad.sum() + query_W_grad.sum() + key_b_grad.sum() + \
                            key_W_grad.sum() + value_b_grad.sum() + value_W_grad.sum() 

                        score = abs_grad_magnitude.item()
                        importance_scores['taylor'][i, head_idx] += score

    return importance_scores

################################################################################################



# redirect requests from root to index.html
@app.route('/')
def index():
    return redirect('client/index.html')


@app.route('/api/projections', methods=['POST'])
def encoder_embedding():
    """
    Load files containing processed data for hidden states projection and 
    attention head importance scores before returning the data to front-end  
    """
    projection_file = pjoin(data_dir, 'projection_data.csv')


    if not os.path.isfile(projection_file):
        labels = interpretation_results['labels']
        label_mask = torch.zeros(hidden_states.shape[:-1])
        for i, index in enumerate(labels):
            label_mask[i, index] = 1

        header = ','.join(['ID', 'tsne_1', 'tsne_2', 'label'])
        save_data = save_hidden_state_projection(hidden_states, label_mask)
        np.savetxt(projection_file, save_data, delimiter=',', header=header, comments='', fmt='%s')
        print(f"Successfully saved projection data at \"{projection_file}\"")

    with open(projection_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        projection_data = [row for row in csv_reader]

    results = {}
    results['projection'] = projection_data
    return jsonify(results)


@app.route('/api/interpretation', methods=['POST'])
def get_interaction_map():
    index = int(request.json['ID'])
    print(index)
    # index = -2

    tokens = [t.replace('\u2581', '') for t in flattened_tokens[index]]
    interactions = flattened_interactions[index]
    results = {}
    results['tokens'] = tokens
    results['interactions'] = normalize(interactions)
    return jsonify(results)


def normalize(array):
    normalized = []
    for row in np.array(array):
        pos_mask = row > 0
        neg_mask = row < 0
        row[pos_mask] = (row[pos_mask] - row[pos_mask].min()) \
                      / (row[pos_mask].max() - row[pos_mask].min())

        row[neg_mask] = (row[neg_mask] - row[neg_mask].min()) \
                      / (row[neg_mask].max() - row[neg_mask].min()) - 1

        normalized.append(row.tolist())
    return normalized


# send everything from client as static content
@app.route('/client/<path:path>')
def send_static(path):
    """ serves all files from ./client/ to ``/client/<path:path>``
    :param path: path from api call
    """
    return send_from_directory('client/', path)


if __name__ == '__main__':
    CWD = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-port', type=int, default='8888')
    parser.add_argument('-host', default=None)
    parser.add_argument('-data_dir', default=pjoin('results', 'debertav3-large', 'predictions', 'dev'), type=str)

    args = parser.parse_args()

    
    load_globals(args)
    app.run(host=args.host, port=int(args.port), debug=args.debug)