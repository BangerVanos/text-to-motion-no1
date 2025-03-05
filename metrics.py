import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from os.path import join as pjoin
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics.pairwise import cosine_similarity

# ==================== Функции вычисления метрик ====================

def compute_r_precision(motion_embeddings, text_embeddings, topk=(1, 2, 3)):
    """
    Вычисляет R-Precision для соответствия движения и текста.
    """    
    similarities = cosine_similarity(motion_embeddings, text_embeddings)
    N = similarities.shape[0]
    results = {}
    for k in topk:
        correct = 0
        for i in range(N):
            top_indices = np.argsort(similarities[i])[::-1][:k]
            if i in top_indices:
                correct += 1
        results[f"R-Precision@{k}"] = correct / N
    return results

def compute_fid(real_features, generated_features):
    """
    Вычисляет FID между реальными и сгенерированными данными.
    """
    from scipy.linalg import sqrtm
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    diff = mu_real - mu_gen
    diff_sq = np.sum(diff**2)
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff_sq + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

def compute_diversity(generated_features):
    """
    Среднее евклидово расстояние между всеми парами эмбеддингов – мера разнообразия.
    """
    from scipy.spatial.distance import pdist
    distances = pdist(generated_features, metric='euclidean')
    return np.mean(distances)

def compute_multimodality(generated_features, texts):
    """
    Для каждого уникального текстового описания вычисляет среднее парное расстояние между
    сгенерированными эмбеддингами движений и усредняет их.
    """
    from scipy.spatial.distance import pdist
    unique_texts = set(texts)
    modality_scores = []
    for txt in unique_texts:
        indices = [i for i, t in enumerate(texts) if t == txt]
        if len(indices) > 1:
            group_features = generated_features[indices]
            distances = pdist(group_features, metric='euclidean')
            modality_scores.append(np.mean(distances))
    return np.mean(modality_scores) if modality_scores else 0.0

def compute_mm_distance(motion_features, text_features):
    """
    Среднее евклидово расстояние между соответствующими эмбеддингами движения и текста.
    """
    distances = np.linalg.norm(motion_features - text_features, axis=1)
    return np.mean(distances)

def evaluate_metrics(real_data, generated_data, texts):
    """
    Объединяет вычисление всех оценочных метрик.
    Здесь real_data и generated_data могут быть torch.Tensor или numpy-массивами.
    """
    # Приводим данные к numpy, если это тензоры
    real_np = real_data if isinstance(real_data, np.ndarray) else real_data.cpu().numpy()
    gen_np = generated_data if isinstance(generated_data, np.ndarray) else generated_data.cpu().numpy()
    
    # Для R-Precision и MM Distance в данном примере считаем, что эмбеддинги текста равны эмбеддингам реальных данных.
    metrics = {}
    metrics.update(compute_r_precision(gen_np, real_np))
    metrics["FID"] = compute_fid(real_np, gen_np)
    metrics["Diversity"] = compute_diversity(gen_np)
    metrics["Multimodality"] = compute_multimodality(gen_np, texts)
    metrics["MM Distance"] = compute_mm_distance(real_np, real_np)  # Здесь замените на реальные текстовые эмбеддинги, если они есть
    return metrics

def plot_eval_metrics(metrics):
    """
    Создаёт график (бар-чарт) по оценочным метрикам.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(metrics.keys())
    values = list(metrics.values())
    ax.bar(names, values, color='skyblue')
    ax.set_title("Evaluation Metrics")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
