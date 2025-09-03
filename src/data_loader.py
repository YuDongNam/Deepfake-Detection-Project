"""
Data loading and preprocessing module for deepfake detection.

This module contains the dataset classes and data loading utilities for the deepfake detection project.
It handles face extraction, frame sampling, and data splitting based on face count.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path


class DeepfakeVideoDataset(Dataset):
    """
    Dataset class for deepfake video detection using Video Swin Transformer.
    
    This dataset loads video frames and applies face count filtering for data splitting strategy:
    - Training/Validation: Videos with single face
    - Test: Videos with multiple faces
    
    Args:
        root_dir (str): Root directory containing 'original' and 'manipulated' subdirectories
        frame_num (int): Number of consecutive frames to extract (default: 72)
        size (int): Frame image size for resizing (default: 224 for Swin Transformer)
        face_count_filter (Optional[int]): Filter by face count (1=single face, 2=multiple faces, None=all)
    """
    
    def __init__(self, root_dir: str, frame_num: int = 72, size: int = 224, 
                 face_count_filter: Optional[int] = None):
        self.data = []
        self.labels = []
        self.video_face_count = {}  # Track face count per video
        self.frame_num = frame_num
        self.size = size
        
        # ImageNet normalization for Video Swin Transformer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self._load_data(root_dir, face_count_filter)
    
    def _load_data(self, root_dir: str, face_count_filter: Optional[int]) -> None:
        """Load and organize video data based on face count filtering."""
        for label, subdir in enumerate(['original', 'manipulated']):
            sub_path = os.path.join(root_dir, subdir)
            video_dict = {}
            
            # Group images by video and face index
            for img_path in glob.glob(os.path.join(sub_path, "*.jpg")):
                fname = os.path.basename(img_path)
                main, ext = os.path.splitext(fname)
                parts = main.split('_')
                
                if len(parts) < 3:
                    continue
                    
                try:
                    face_idx = int(parts[-1])
                    fnum = int(parts[-2])
                except ValueError:
                    continue
                    
                video_id = '_'.join(parts[:-2])
                
                if video_id not in video_dict:
                    video_dict[video_id] = {}
                if face_idx not in video_dict[video_id]:
                    video_dict[video_id][face_idx] = {}
                    
                video_dict[video_id][face_idx][fnum] = img_path
            
            # Apply face count filtering and create dataset entries
            for video_id, people_dict in video_dict.items():
                face_count = len(people_dict)
                self.video_face_count[video_id] = face_count
                
                if face_count_filter is not None:
                    if face_count_filter == 1 and face_count != 1:
                        continue
                    elif face_count_filter == 2 and face_count < 2:
                        continue
                
                for face_idx, frame_dict in people_dict.items():
                    if len(frame_dict) == self.frame_num:
                        frames = [frame_dict[fidx] for fidx in range(self.frame_num)]
                        self.data.append((video_id, face_idx, frames))
                        self.labels.append(label)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple containing:
                - video_tensor: Tensor of shape (C, T, H, W) for Video Swin Transformer
                - label: Binary label (0=original, 1=manipulated)
                - video_id: Video identifier
                - face_idx: Face index within the video
        """
        video_id, face_idx, frames = self.data[idx]
        label = self.labels[idx]
        imgs = []
        
        for img_path in frames:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size))
            img = self.transform(img)
            imgs.append(img)
        
        # Stack frames and permute for Video Swin Transformer: (T, C, H, W) -> (C, T, H, W)
        video_tensor = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)
        
        return video_tensor, label, video_id, face_idx


class DeepfakeSeqDataset(Dataset):
    """
    Dataset class for deepfake video detection using XceptionNet + LSTM.
    
    This dataset loads video frames for the XceptionNet+LSTM architecture.
    
    Args:
        root_dir (str): Root directory containing 'original' and 'manipulated' subdirectories
        frame_num (int): Number of consecutive frames to extract (default: 72)
        size (int): Frame image size for resizing (default: 299 for XceptionNet)
        face_count_filter (Optional[int]): Filter by face count (1=single face, 2=multiple faces, None=all)
    """
    
    def __init__(self, root_dir: str, frame_num: int = 72, size: int = 299, 
                 face_count_filter: Optional[int] = None):
        self.data = []
        self.labels = []
        self.video_face_count = {}
        self.frame_num = frame_num
        self.size = size
        
        # XceptionNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self._load_data(root_dir, face_count_filter)
    
    def _load_data(self, root_dir: str, face_count_filter: Optional[int]) -> None:
        """Load and organize video data based on face count filtering."""
        for label, subdir in enumerate(['original', 'manipulated']):
            sub_path = os.path.join(root_dir, subdir)
            video_dict = {}
            
            # Group images by video and face index
            for img_path in glob.glob(os.path.join(sub_path, "*.jpg")):
                fname = os.path.basename(img_path)
                main, ext = os.path.splitext(fname)
                parts = main.split('_')
                
                if len(parts) < 3:
                    continue
                    
                try:
                    face_idx = int(parts[-1])
                    fnum = int(parts[-2])
                except ValueError:
                    continue
                    
                video_id = '_'.join(parts[:-2])
                
                if video_id not in video_dict:
                    video_dict[video_id] = {}
                if face_idx not in video_dict[video_id]:
                    video_dict[video_id][face_idx] = {}
                    
                video_dict[video_id][face_idx][fnum] = img_path
            
            # Apply face count filtering and create dataset entries
            for video_id, people_dict in video_dict.items():
                face_count = len(people_dict)
                self.video_face_count[video_id] = face_count
                
                if face_count_filter is not None:
                    if face_count_filter == 1 and face_count != 1:
                        continue
                    elif face_count_filter == 2 and face_count < 2:
                        continue
                
                for face_idx, frame_dict in people_dict.items():
                    if len(frame_dict) == self.frame_num:
                        frame_files = [frame_dict[fidx] for fidx in range(self.frame_num)]
                        self.data.append((video_id, face_idx, frame_files))
                        self.labels.append(label)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple containing:
                - frames: Tensor of shape (T, C, H, W) for XceptionNet+LSTM
                - label: Binary label (0=original, 1=manipulated)
                - video_id: Video identifier
                - face_idx: Face index within the video
        """
        video_id, face_idx, frames = self.data[idx]
        label = self.labels[idx]
        imgs = []
        
        for img_path in frames:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size))
            img = self.transform(img)
            imgs.append(img)
        
        # Stack frames for XceptionNet+LSTM: (T, C, H, W)
        frames_tensor = torch.stack(imgs, dim=0)
        
        return frames_tensor, label, video_id, face_idx


def create_data_loaders(root_dir: str, model_type: str = "swin", batch_size: int = 4, 
                       num_workers: int = 2, random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        root_dir (str): Root directory containing the dataset
        model_type (str): Model type ("swin" or "xception")
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        random_state (int): Random state for train/val split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Choose dataset class based on model type
    if model_type.lower() == "swin":
        DatasetClass = DeepfakeVideoDataset
        size = 224
    elif model_type.lower() == "xception":
        DatasetClass = DeepfakeSeqDataset
        size = 299
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'swin' or 'xception'.")
    
    # Create single-face dataset for training and validation
    dataset_1face = DatasetClass(root_dir, frame_num=72, size=size, face_count_filter=1)
    
    # Split into train and validation
    indices = np.arange(len(dataset_1face))
    labels = [dataset_1face.labels[i] for i in range(len(dataset_1face))]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=random_state
    )
    
    train_set = Subset(dataset_1face, train_idx)
    val_set = Subset(dataset_1face, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Create multi-face dataset for testing
    dataset_2face = DatasetClass(root_dir, frame_num=72, size=size, face_count_filter=2)
    test_loader = DataLoader(dataset_2face, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def extract_faces_from_videos(video_dir: str, save_dir: str, num_frames: int = 8) -> None:
    """
    Extract faces from videos using RetinaFace detection.
    
    This function processes videos to extract face crops and save them as images.
    It's used for the initial data preprocessing step.
    
    Args:
        video_dir (str): Directory containing input videos
        save_dir (str): Directory to save extracted face images
        num_frames (int): Number of frames to extract per video
    """
    try:
        from retinaface import RetinaFace
    except ImportError:
        raise ImportError("RetinaFace is required for face extraction. Install with: pip install retina-face")
    
    from tqdm import tqdm
    
    video_dir = Path(video_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    video_paths = sorted([p for p in video_dir.glob("*.mp4") if p.is_file()])
    
    tqdm.write(f"[INFO] Found {len(video_paths)} videos in {video_dir.name}")
    
    for video_path in tqdm(video_paths, desc=f"Processing {video_dir.name}"):
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < num_frames:
                tqdm.write(f"[WARN] Skipping {video_path.name} (not enough frames)")
                cap.release()
                continue
            
            interval = total_frames // num_frames
            frame_idx, saved = 0, 0
            
            while cap.isOpened() and saved < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % interval == 0:
                    faces = RetinaFace.detect_faces(frame)
                    if isinstance(faces, dict) and len(faces) > 0:
                        face = list(faces.values())[0]
                        x1, y1, x2, y2 = face['facial_area']
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # Create save path
                        vid_stem = video_path.stem
                        save_path = save_dir / f"{vid_stem}_{saved:02d}.jpg"
                        cv2.imwrite(str(save_path), face_crop)
                        saved += 1
                        
                frame_idx += 1
            cap.release()
            
        except Exception as e:
            tqdm.write(f"[ERROR] Failed processing {video_path.name}: {e}")
            continue
