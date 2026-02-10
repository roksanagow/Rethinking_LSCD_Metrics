import torch
from sklearn import decomposition as pca    

def return_full_space(t1, t2, ctx):
    return t1, t2

def def_space(t1, t2, ctx):
    # project the word usage embeddings onto the definition embeddings by calculating cosine similarities between each word usage embedding and each definition embedding
    def_emb = ctx["def_emb"]
    projected_usage_embeddings_t1 = torch.cdist(t1, def_emb)
    projected_usage_embeddings_t2 = torch.cdist(t2, def_emb)
    return projected_usage_embeddings_t1, projected_usage_embeddings_t2

def pca_space(t1, t2, ctx):
    num_defs = ctx["num_defs"]
    n_components = min(num_defs, t1.size(0), t1.size(1))
    pca_reducer = pca.PCA(n_components=n_components)

    joint_embs = torch.cat([t1, t2], dim=0)

    pca_joint_embs = torch.tensor(pca_reducer.fit_transform(joint_embs.numpy()))
    halfway = pca_joint_embs.shape[0] // 2
    pca_usage_embs_t1 = pca_joint_embs[:halfway]
    pca_usage_embs_t2 = pca_joint_embs[halfway:]

    return pca_usage_embs_t1, pca_usage_embs_t2


def pca_space_num(t1, t2, ctx, num_components=1024):
    num_components = min(num_components, t1.size(0), t1.size(1))
    pca_reducer = pca.PCA(n_components=num_components)
    joint_embs = torch.cat([t1, t2], dim=0)

    pca_joint_embs = torch.tensor(pca_reducer.fit_transform(joint_embs.numpy()))
    halfway = pca_joint_embs.shape[0] // 2
    pca_usage_embs_t1 = pca_joint_embs[:halfway]
    pca_usage_embs_t2 = pca_joint_embs[halfway:]

    return pca_usage_embs_t1, pca_usage_embs_t2

def random_dim_selection(t1, t2, ctx):
    """Randomly remove noise_level fraction of dimensions from the embeddings."""
    t_all = torch.cat([t1, t2], dim=0)
    n, d = t_all.shape
    num_keep = ctx["num_defs"]
    if num_keep < 1:
        num_keep = 1

    perm = torch.randperm(d, device=t_all.device)
    keep_idx = perm[:num_keep]

    t_pruned = t_all[:, keep_idx]
    # print how many dimensions were kept
    halfway = t_pruned.shape[0] // 2
    return t_pruned[:halfway], t_pruned[halfway:]

def random_dim_selection_num(t1, t2, ctx, num_dims=1024):
    """Randomly remove noise_level fraction of dimensions from the embeddings."""
    t_all = torch.cat([t1, t2], dim=0)
    n, d = t_all.shape
    num_keep = num_dims
    if num_keep < 1:
        num_keep = 1

    perm = torch.randperm(d, device=t_all.device)
    keep_idx = perm[:num_keep]

    t_pruned = t_all[:, keep_idx]
    # print how many dimensions were kept
    halfway = t_pruned.shape[0] // 2
    return t_pruned[:halfway], t_pruned[halfway:]