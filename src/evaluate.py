"""
Evaluation module for deepfake detection models.

This module contains functions for model evaluation, including both sequence-level
and video-level evaluation metrics and confusion matrix analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from .models import load_model


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device,
                   threshold: float = 0.4, return_predictions: bool = False) -> Dict[str, float]:
    """
    Evaluate model on test data and return comprehensive metrics.
    
    Args:
        model: The trained model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        return_predictions: Whether to return predictions and targets
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    preds, targets, probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if len(batch) == 4:  # Video Swin Transformer format
                video_tensor, label, video_id, face_idx = batch
                video_tensor = video_tensor.to(device)
            else:  # XceptionNet+LSTM format
                frames, label, video_id, face_idx = batch
                video_tensor = frames.to(device)
            
            label = label.to(device)
            
            out = model(video_tensor)
            prob = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            pred_label = (prob > threshold).astype(int)
            
            preds.extend(pred_label)
            targets.extend(label.cpu().numpy())
            probs.extend(prob)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'f1_score': f1_score(targets, preds),
        'auc': roc_auc_score(targets, probs),
        'precision': precision_score(targets, preds),
        'recall': recall_score(targets, preds),
        'mcc': matthews_corrcoef(targets, preds),
        'pr_auc': average_precision_score(targets, probs)
    }
    
    if return_predictions:
        metrics['predictions'] = preds
        metrics['targets'] = targets
        metrics['probabilities'] = probs
    
    return metrics


def evaluate_video_level(model: nn.Module, test_loader: DataLoader, device: torch.device,
                        threshold: float = 0.4) -> Dict[str, float]:
    """
    Evaluate model at video level for multi-face videos.
    
    For videos with multiple faces, if any face is predicted as manipulated,
    the entire video is considered manipulated.
    
    Args:
        model: The trained model to evaluate
        test_loader: Test data loader (should contain multi-face videos)
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing video-level evaluation metrics
    """
    model.eval()
    video_pred_dict = defaultdict(list)
    video_target_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Video-level Evaluation"):
            if len(batch) == 4:  # Video Swin Transformer format
                video_tensor, label, video_id, face_idx = batch
                video_tensor = video_tensor.to(device)
            else:  # XceptionNet+LSTM format
                frames, label, video_id, face_idx = batch
                video_tensor = frames.to(device)
            
            out = model(video_tensor)
            prob = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            pred_label = (prob > threshold).astype(int)
            
            for vid, pl, tar in zip(video_id, pred_label, label.cpu().numpy()):
                video_pred_dict[vid].append(pl)
                video_target_dict[vid] = tar
    
    # Aggregate predictions at video level
    video_preds = []
    video_targets = []
    
    for vid in video_pred_dict:
        # If any face is predicted as manipulated, video is manipulated
        is_manipulated = (np.array(video_pred_dict[vid]) > 0).any()
        video_preds.append(int(is_manipulated))
        video_targets.append(video_target_dict[vid])
    
    # Calculate video-level metrics
    metrics = {
        'video_accuracy': accuracy_score(video_targets, video_preds),
        'video_f1_score': f1_score(video_targets, video_preds),
        'video_precision': precision_score(video_targets, video_preds, zero_division=0),
        'video_recall': recall_score(video_targets, video_preds, zero_division=0),
        'video_mcc': matthews_corrcoef(video_targets, video_preds)
    }
    
    return metrics


def get_confusion_matrix(model: nn.Module, test_loader: DataLoader, device: torch.device,
                        threshold: float = 0.4) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Generate confusion matrix for model evaluation.
    
    Args:
        model: The trained model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        
    Returns:
        Tuple of (confusion_matrix, true_labels, predicted_labels)
    """
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Confusion Matrix"):
            if len(batch) == 4:  # Video Swin Transformer format
                video_tensor, label, video_id, face_idx = batch
                video_tensor = video_tensor.to(device)
            else:  # XceptionNet+LSTM format
                frames, label, video_id, face_idx = batch
                video_tensor = frames.to(device)
            
            out = model(video_tensor)
            prob = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            pred_label = (prob > threshold).astype(int)
            
            preds.extend(pred_label)
            targets.extend(label.cpu().numpy())
    
    cm = confusion_matrix(targets, preds)
    class_labels = ['Original', 'Manipulated']
    
    return cm, class_labels, class_labels


def find_misclassified_samples(model: nn.Module, test_loader: DataLoader, device: torch.device,
                              threshold: float = 0.4, max_samples: int = 20) -> List[Tuple]:
    """
    Find misclassified samples for analysis.
    
    Args:
        model: The trained model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        max_samples: Maximum number of misclassified samples to return
        
    Returns:
        List of tuples containing (video_id, face_idx, true_label, predicted_label, probability)
    """
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Finding Misclassified Samples"):
            if len(batch) == 4:  # Video Swin Transformer format
                video_tensor, label, video_id, face_idx = batch
                video_tensor = video_tensor.to(device)
            else:  # XceptionNet+LSTM format
                frames, label, video_id, face_idx = batch
                video_tensor = frames.to(device)
            
            out = model(video_tensor)
            prob = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            pred_label = (prob > threshold).astype(int)
            
            for vid, fidx, true_lab, pred_lab, prob_val in zip(
                video_id, face_idx, label.cpu().numpy(), pred_label, prob
            ):
                if true_lab != pred_lab:
                    misclassified.append((vid, fidx, true_lab, pred_lab, prob_val))
                    
                    if len(misclassified) >= max_samples:
                        return misclassified
    
    return misclassified


def evaluate_model_comprehensive(model: nn.Module, test_loader: DataLoader, device: torch.device,
                                threshold: float = 0.4) -> Dict[str, any]:
    """
    Comprehensive evaluation of the model including all metrics and analysis.
    
    Args:
        model: The trained model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing comprehensive evaluation results
    """
    print("Starting comprehensive model evaluation...")
    
    # Sequence-level evaluation
    print("\n1. Sequence-level Evaluation:")
    seq_metrics = evaluate_model(model, test_loader, device, threshold, return_predictions=True)
    
    for metric, value in seq_metrics.items():
        if metric not in ['predictions', 'targets', 'probabilities']:
            print(f"  {metric}: {value:.4f}")
    
    # Video-level evaluation (if applicable)
    print("\n2. Video-level Evaluation:")
    video_metrics = evaluate_video_level(model, test_loader, device, threshold)
    
    for metric, value in video_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Confusion matrix
    print("\n3. Confusion Matrix:")
    cm, true_labels, pred_labels = get_confusion_matrix(model, test_loader, device, threshold)
    print(f"  Confusion Matrix:\n{cm}")
    
    # Misclassified samples
    print("\n4. Misclassified Samples:")
    misclassified = find_misclassified_samples(model, test_loader, device, threshold)
    print(f"  Found {len(misclassified)} misclassified samples")
    
    for i, (vid, fidx, true_lab, pred_lab, prob) in enumerate(misclassified[:10]):
        print(f"    {i+1}. Video: {vid}, Face: {fidx}, True: {true_lab}, Pred: {pred_lab}, Prob: {prob:.3f}")
    
    # Combine all results
    results = {
        'sequence_metrics': seq_metrics,
        'video_metrics': video_metrics,
        'confusion_matrix': cm,
        'misclassified_samples': misclassified
    }
    
    return results


def print_evaluation_summary(results: Dict[str, any]) -> None:
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Results dictionary from evaluate_model_comprehensive
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Sequence-level metrics
    print("\nSequence-level Performance:")
    seq_metrics = results['sequence_metrics']
    for metric, value in seq_metrics.items():
        if metric not in ['predictions', 'targets', 'probabilities']:
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Video-level metrics
    if 'video_metrics' in results:
        print("\nVideo-level Performance:")
        video_metrics = results['video_metrics']
        for metric, value in video_metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  Original -> Original: {cm[0,0]}")
    print(f"  Original -> Manipulated: {cm[0,1]}")
    print(f"  Manipulated -> Original: {cm[1,0]}")
    print(f"  Manipulated -> Manipulated: {cm[1,1]}")
    
    print("="*60)
