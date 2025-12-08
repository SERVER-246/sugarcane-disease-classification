#!/usr/bin/env python3
"""
================================================================================
🌿 SUGARCANE DISEASE CLASSIFICATION - SCIENTIFIC GUI
================================================================================

A visually stunning, professional GUI for deploying the 15-COIN ensemble system.

Features:
- Primary: Knowledge Distillation Student Model (93.21% accuracy, fast)
- Fallback 1: Meta-Ensemble MLP (96.61% accuracy)
- Fallback 2: CustomMaxViT backbone (95.39% accuracy)
- Confidence filtering to reject unrelated images
- Real-time visualization with probability charts
- Batch processing support
- Export results to CSV/JSON

Author: SERVER-246
Version: 1.0.0
================================================================================
"""

import sys
import os
from pathlib import Path

# Set up paths BEFORE other imports
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'BASE-BACK' / 'src'))
sys.path.insert(0, str(BASE_DIR / 'ensemble_system'))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Image validator for filtering non-sugarcane images
from image_validator import ImageValidator, ValidationReport, ValidationResult
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


# =============================================================================
# CONFIGURATION
# =============================================================================

CLASS_NAMES = [
    'Black Stripe', 'Brown Spot', 'Grassy Shoot Disease', 'Healthy',
    'Leaf Flecking', 'Leaf Scorching', 'Mosaic', 'Pokkah Boeng',
    'Red Rot', 'Ring Spot', 'Smut', 'Wilt', 'Yellow Leaf Disease'
]

# Map to folder names
CLASS_FOLDER_MAP = {
    'Black Stripe': 'Black_stripe',
    'Brown Spot': 'Brown_spot',
    'Grassy Shoot Disease': 'Grassy_shoot_disease',
    'Healthy': 'Healthy',
    'Leaf Flecking': 'Leaf_flecking',
    'Leaf Scorching': 'Leaf_scorching',
    'Mosaic': 'Mosaic',
    'Pokkah Boeng': 'Pokkah_boeng',
    'Red Rot': 'Red_rot',
    'Ring Spot': 'Ring_spot',
    'Smut': 'Smut',
    'Wilt': 'Wilt',
    'Yellow Leaf Disease': 'Yellow_leaf_Disease'
}

# Disease info for display
DISEASE_INFO = {
    'Black Stripe': {'severity': 'Moderate', 'color': '#2C3E50', 'action': 'Remove infected leaves, apply fungicide'},
    'Brown Spot': {'severity': 'Low', 'color': '#8B4513', 'action': 'Improve drainage, reduce humidity'},
    'Grassy Shoot Disease': {'severity': 'High', 'color': '#228B22', 'action': 'Remove infected plants, control vectors'},
    'Healthy': {'severity': 'None', 'color': '#27AE60', 'action': 'No action needed - healthy plant!'},
    'Leaf Flecking': {'severity': 'Low', 'color': '#F39C12', 'action': 'Monitor, usually cosmetic'},
    'Leaf Scorching': {'severity': 'Moderate', 'color': '#E74C3C', 'action': 'Improve irrigation, check nutrients'},
    'Mosaic': {'severity': 'High', 'color': '#9B59B6', 'action': 'Remove infected plants, control aphids'},
    'Pokkah Boeng': {'severity': 'Moderate', 'color': '#1ABC9C', 'action': 'Apply fungicide, improve air circulation'},
    'Red Rot': {'severity': 'Critical', 'color': '#C0392B', 'action': 'Immediate removal, soil treatment'},
    'Ring Spot': {'severity': 'Moderate', 'color': '#3498DB', 'action': 'Remove infected leaves, apply copper fungicide'},
    'Smut': {'severity': 'High', 'color': '#34495E', 'action': 'Burn infected material, use resistant varieties'},
    'Wilt': {'severity': 'Critical', 'color': '#7F8C8D', 'action': 'Remove plants, soil fumigation'},
    'Yellow Leaf Disease': {'severity': 'High', 'color': '#F1C40F', 'action': 'Use virus-free seed, control whiteflies'}
}

# Model paths
STUDENT_MODEL_PATH = BASE_DIR / 'ensembles' / 'stage7_distillation' / 'student_model.pth'
META_MLP_PATH = BASE_DIR / 'ensembles' / 'stage6_meta' / 'mlp' / 'mlp_meta.pth'
MAXVIT_PATH = BASE_DIR / 'checkpoints' / 'CustomMaxViT_final.pth'

# Confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
UNRELATED_IMAGE_THRESHOLD = 0.40  # Below this, likely not a sugarcane image

# Image preprocessing
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class CompactStudentModel(nn.Module):
    """Knowledge Distillation Student Model (Stage 7)"""
    
    def __init__(self, num_classes: int = 13):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = self._make_stage(32, 64, num_blocks=2, stride=2)
        self.stage2 = self._make_stage(64, 128, num_blocks=3, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=3, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            layers.append(self._make_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MetaMLPController(nn.Module):
    """Meta-Ensemble MLP Controller (Stage 6)"""
    
    def __init__(self, input_dim: int = 143, num_classes: int = 13):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# STYLING AND THEMES
# =============================================================================

class Theme:
    """Modern dark scientific theme"""
    
    # Main colors
    BG_DARK = '#1a1a2e'
    BG_MEDIUM = '#16213e'
    BG_LIGHT = '#0f3460'
    ACCENT = '#e94560'
    ACCENT_LIGHT = '#ff6b6b'
    TEXT_PRIMARY = '#ffffff'
    TEXT_SECONDARY = '#a0a0a0'
    SUCCESS = '#00d26a'
    WARNING = '#ffc107'
    ERROR = '#ff4757'
    INFO = '#3498db'
    
    # Severity colors
    SEVERITY_NONE = '#27AE60'
    SEVERITY_LOW = '#F1C40F'
    SEVERITY_MODERATE = '#E67E22'
    SEVERITY_HIGH = '#E74C3C'
    SEVERITY_CRITICAL = '#8E44AD'
    
    # Chart colors
    CHART_COLORS = [
        '#e94560', '#00d26a', '#3498db', '#f39c12', '#9b59b6',
        '#1abc9c', '#e74c3c', '#2ecc71', '#3498db', '#9b59b6',
        '#f1c40f', '#e67e22', '#1abc9c'
    ]
    
    # Fonts
    FONT_TITLE = ('Segoe UI', 24, 'bold')
    FONT_SUBTITLE = ('Segoe UI', 14)
    FONT_HEADING = ('Segoe UI', 12, 'bold')
    FONT_BODY = ('Segoe UI', 10)
    FONT_SMALL = ('Segoe UI', 9)
    FONT_MONO = ('Consolas', 10)


# =============================================================================
# CLASSIFICATION ENGINE
# =============================================================================

class ClassificationEngine:
    """Handles model loading and inference"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.current_model = None
        self.transform = self._create_transform()
        
        # Initialize image validator for filtering non-sugarcane images
        self.image_validator = ImageValidator(use_deep_learning=False)
        
    def _create_transform(self):
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def load_student_model(self) -> bool:
        """Load the distilled student model"""
        try:
            if not STUDENT_MODEL_PATH.exists():
                return False
            
            model = CompactStudentModel(num_classes=13)
            state_dict = torch.load(STUDENT_MODEL_PATH, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models['student'] = model
            return True
        except Exception as e:
            print(f"Error loading student model: {e}")
            return False
    
    def load_maxvit_model(self) -> bool:
        """Load CustomMaxViT backbone as fallback"""
        try:
            if not MAXVIT_PATH.exists():
                return False
            
            # Import the architecture
            from models.architectures import create_custom_backbone
            model = create_custom_backbone('CustomMaxViT', num_classes=13)
            
            state_dict = torch.load(MAXVIT_PATH, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            model.to(self.device)
            model.eval()
            
            self.models['maxvit'] = model
            return True
        except Exception as e:
            print(f"Error loading MaxViT model: {e}")
            return False
    
    def load_all_models(self, callback=None):
        """Load all models with progress callback"""
        results = {}
        
        if callback:
            callback("Loading Student Model (Stage 7)...", 0.33)
        results['student'] = self.load_student_model()
        
        if callback:
            callback("Loading CustomMaxViT Backbone...", 0.66)
        results['maxvit'] = self.load_maxvit_model()
        
        if callback:
            callback("Models loaded!", 1.0)
        
        # Set default model
        if results['student']:
            self.current_model = 'student'
        elif results['maxvit']:
            self.current_model = 'maxvit'
        
        return results
    
    def classify_image(self, image_path: str, model_name: str = None) -> Dict[str, Any]:
        """
        Classify a single image with pre-validation filtering
        
        Returns:
            dict with: predicted_class, confidence, all_probabilities, is_valid, validation_result
        """
        if model_name is None:
            model_name = self.current_model
        
        if model_name not in self.models:
            return {'error': f'Model {model_name} not loaded'}
        
        model = self.models[model_name]
        
        try:
            # ============================================================
            # STEP 1: Pre-validate image using multi-factor analysis
            # ============================================================
            validation = self.image_validator.validate(image_path)
            
            if not validation.is_valid:
                # Image rejected - return early without running model
                return {
                    'predicted_class': 'REJECTED',
                    'predicted_idx': -1,
                    'confidence': 0.0,
                    'all_probabilities': {name: 0.0 for name in CLASS_NAMES},
                    'entropy': 0.0,
                    'is_valid': False,
                    'is_rejected': True,
                    'rejection_reason': validation.message,
                    'validation_scores': {
                        'quality': validation.quality_score / 100.0,
                        'vegetation': validation.vegetation_score,
                        'sugarcane': validation.sugarcane_score
                    },
                    'validation_details': validation.details,
                    'validation_confidence': validation.confidence,
                    'suggestions': validation.suggestions,
                    'model_used': 'validator',
                    'image_path': image_path
                }
            
            # ============================================================
            # STEP 2: Run model inference on validated images
            # ============================================================
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Get prediction
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            # Create probability dict
            prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
            
            # Secondary check using model confidence and entropy
            max_confidence = float(confidence)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            model_confident = max_confidence > UNRELATED_IMAGE_THRESHOLD and entropy < 2.0
            
            # Combined validity: image validator + model confidence
            is_valid = model_confident  # Already passed pre-validation
            
            return {
                'predicted_class': CLASS_NAMES[predicted_idx],
                'predicted_idx': int(predicted_idx),
                'confidence': float(confidence),
                'all_probabilities': prob_dict,
                'entropy': float(entropy),
                'is_valid': is_valid,
                'is_rejected': False,
                'validation_scores': {
                    'quality': validation.quality_score / 100.0,
                    'vegetation': validation.vegetation_score,
                    'sugarcane': validation.sugarcane_score
                },
                'validation_confidence': validation.confidence,
                'model_used': model_name,
                'image_path': image_path
            }
            
        except Exception as e:
            return {'error': str(e), 'image_path': image_path}


# =============================================================================
# MAIN GUI APPLICATION
# =============================================================================

class DiseaseClassifierGUI:
    """Main application class"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🌿 Sugarcane Disease Classifier - Scientific Analysis Suite")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure dark theme
        self.root.configure(bg=Theme.BG_DARK)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()
        
        # State
        self.engine = ClassificationEngine()
        self.current_image_path = None
        self.current_result = None
        self.confidence_threshold = tk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
        self.selected_model = tk.StringVar(value='student')
        self.history = []
        
        # Build UI
        self._create_layout()
        
        # Load models in background
        self.root.after(100, self._load_models_async)
    
    def _configure_styles(self):
        """Configure ttk styles for dark theme"""
        
        # Frame styles
        self.style.configure('Dark.TFrame', background=Theme.BG_DARK)
        self.style.configure('Medium.TFrame', background=Theme.BG_MEDIUM)
        self.style.configure('Light.TFrame', background=Theme.BG_LIGHT)
        
        # Label styles
        self.style.configure('Title.TLabel',
            background=Theme.BG_DARK,
            foreground=Theme.TEXT_PRIMARY,
            font=Theme.FONT_TITLE)
        
        self.style.configure('Subtitle.TLabel',
            background=Theme.BG_DARK,
            foreground=Theme.TEXT_SECONDARY,
            font=Theme.FONT_SUBTITLE)
        
        self.style.configure('Heading.TLabel',
            background=Theme.BG_MEDIUM,
            foreground=Theme.TEXT_PRIMARY,
            font=Theme.FONT_HEADING)
        
        self.style.configure('Body.TLabel',
            background=Theme.BG_MEDIUM,
            foreground=Theme.TEXT_PRIMARY,
            font=Theme.FONT_BODY)
        
        self.style.configure('Status.TLabel',
            background=Theme.BG_DARK,
            foreground=Theme.ACCENT,
            font=Theme.FONT_BODY)
        
        # Button styles
        self.style.configure('Accent.TButton',
            background=Theme.ACCENT,
            foreground=Theme.TEXT_PRIMARY,
            font=Theme.FONT_HEADING,
            padding=(20, 10))
        
        self.style.map('Accent.TButton',
            background=[('active', Theme.ACCENT_LIGHT)])
        
        # Scale style
        self.style.configure('Threshold.Horizontal.TScale',
            background=Theme.BG_DARK,
            troughcolor=Theme.BG_LIGHT)
        
        # Progressbar
        self.style.configure('Accent.Horizontal.TProgressbar',
            background=Theme.ACCENT,
            troughcolor=Theme.BG_LIGHT)
    
    def _create_layout(self):
        """Create the main layout"""
        
        # Main container
        self.main_frame = tk.Frame(self.root, bg=Theme.BG_DARK)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        self._create_header()
        
        # Content area
        content_frame = tk.Frame(self.main_frame, bg=Theme.BG_DARK)
        content_frame.pack(fill='both', expand=True, pady=10)
        
        # Left panel - Image and controls
        self._create_left_panel(content_frame)
        
        # Right panel - Results and charts
        self._create_right_panel(content_frame)
        
        # Status bar
        self._create_status_bar()
    
    def _create_header(self):
        """Create header section"""
        header = tk.Frame(self.main_frame, bg=Theme.BG_DARK)
        header.pack(fill='x', pady=(0, 10))
        
        # Title
        title_frame = tk.Frame(header, bg=Theme.BG_DARK)
        title_frame.pack(side='left')
        
        title = tk.Label(title_frame,
            text="🌿 Sugarcane Disease Classifier",
            font=Theme.FONT_TITLE,
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_PRIMARY)
        title.pack(anchor='w')
        
        subtitle = tk.Label(title_frame,
            text="15-COIN Ensemble System • Scientific Analysis Suite",
            font=Theme.FONT_SUBTITLE,
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY)
        subtitle.pack(anchor='w')
        
        # Model selector and info
        info_frame = tk.Frame(header, bg=Theme.BG_DARK)
        info_frame.pack(side='right')
        
        # Model selection
        model_label = tk.Label(info_frame,
            text="Active Model:",
            font=Theme.FONT_BODY,
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY)
        model_label.pack(anchor='e')
        
        self.model_combo = ttk.Combobox(info_frame,
            textvariable=self.selected_model,
            values=['student', 'maxvit'],
            state='readonly',
            width=20)
        self.model_combo.pack(anchor='e', pady=2)
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        # Device info
        device_text = f"Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}"
        device_label = tk.Label(info_frame,
            text=device_text,
            font=Theme.FONT_SMALL,
            bg=Theme.BG_DARK,
            fg=Theme.SUCCESS if torch.cuda.is_available() else Theme.WARNING)
        device_label.pack(anchor='e', pady=(5, 0))
    
    def _create_left_panel(self, parent):
        """Create left panel with image display and controls"""
        left_panel = tk.Frame(parent, bg=Theme.BG_MEDIUM, width=500)
        left_panel.pack(side='left', fill='both', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Image display area
        image_frame = tk.Frame(left_panel, bg=Theme.BG_LIGHT, relief='flat', bd=2)
        image_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.image_label = tk.Label(image_frame,
            text="📷\n\nDrag & Drop Image\nor\nClick 'Load Image' below",
            font=Theme.FONT_SUBTITLE,
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_SECONDARY,
            compound='center')
        self.image_label.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Control buttons
        controls_frame = tk.Frame(left_panel, bg=Theme.BG_MEDIUM)
        controls_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        # Load image button
        self.load_btn = tk.Button(controls_frame,
            text="📂 Load Image",
            font=Theme.FONT_HEADING,
            bg=Theme.ACCENT,
            fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.ACCENT_LIGHT,
            activeforeground=Theme.TEXT_PRIMARY,
            relief='flat',
            cursor='hand2',
            command=self._load_image)
        self.load_btn.pack(fill='x', pady=5)
        
        # Classify button
        self.classify_btn = tk.Button(controls_frame,
            text="🔬 Analyze Disease",
            font=Theme.FONT_HEADING,
            bg=Theme.SUCCESS,
            fg=Theme.TEXT_PRIMARY,
            activebackground='#00ff7f',
            activeforeground=Theme.TEXT_PRIMARY,
            relief='flat',
            cursor='hand2',
            state='disabled',
            command=self._classify_current_image)
        self.classify_btn.pack(fill='x', pady=5)
        
        # Batch process button
        self.batch_btn = tk.Button(controls_frame,
            text="📁 Batch Process Folder",
            font=Theme.FONT_BODY,
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.INFO,
            activeforeground=Theme.TEXT_PRIMARY,
            relief='flat',
            cursor='hand2',
            command=self._batch_process)
        self.batch_btn.pack(fill='x', pady=5)
        
        # Confidence threshold
        threshold_frame = tk.Frame(left_panel, bg=Theme.BG_MEDIUM)
        threshold_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        threshold_label = tk.Label(threshold_frame,
            text="Confidence Threshold:",
            font=Theme.FONT_BODY,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY)
        threshold_label.pack(anchor='w')
        
        threshold_scale = tk.Scale(threshold_frame,
            from_=0.0, to=1.0,
            resolution=0.05,
            orient='horizontal',
            variable=self.confidence_threshold,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            troughcolor=Theme.BG_LIGHT,
            activebackground=Theme.ACCENT,
            command=self._on_threshold_change)
        threshold_scale.pack(fill='x')
        
        self.threshold_value_label = tk.Label(threshold_frame,
            text=f"Current: {self.confidence_threshold.get():.0%}",
            font=Theme.FONT_SMALL,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_SECONDARY)
        self.threshold_value_label.pack(anchor='w')
    
    def _create_right_panel(self, parent):
        """Create right panel with results and charts"""
        right_panel = tk.Frame(parent, bg=Theme.BG_DARK)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Results section
        self._create_results_section(right_panel)
        
        # Probability chart
        self._create_chart_section(right_panel)
    
    def _create_results_section(self, parent):
        """Create results display section"""
        results_frame = tk.Frame(parent, bg=Theme.BG_MEDIUM)
        results_frame.pack(fill='x', pady=(0, 10))
        
        # Prediction result
        self.prediction_frame = tk.Frame(results_frame, bg=Theme.BG_MEDIUM)
        self.prediction_frame.pack(fill='x', padx=15, pady=15)
        
        pred_title = tk.Label(self.prediction_frame,
            text="🔬 Analysis Result",
            font=Theme.FONT_HEADING,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY)
        pred_title.pack(anchor='w')
        
        # Main prediction display
        self.pred_class_label = tk.Label(self.prediction_frame,
            text="No image loaded",
            font=('Segoe UI', 28, 'bold'),
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_SECONDARY)
        self.pred_class_label.pack(anchor='w', pady=(10, 5))
        
        # Confidence display
        self.confidence_label = tk.Label(self.prediction_frame,
            text="Confidence: ---%",
            font=Theme.FONT_SUBTITLE,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_SECONDARY)
        self.confidence_label.pack(anchor='w')
        
        # Confidence bar
        self.confidence_canvas = tk.Canvas(self.prediction_frame,
            height=20, bg=Theme.BG_LIGHT,
            highlightthickness=0)
        self.confidence_canvas.pack(fill='x', pady=10)
        
        # Disease info
        self.disease_info_frame = tk.Frame(results_frame, bg=Theme.BG_LIGHT)
        self.disease_info_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.severity_label = tk.Label(self.disease_info_frame,
            text="Severity: ---",
            font=Theme.FONT_BODY,
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_PRIMARY)
        self.severity_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        self.action_label = tk.Label(self.disease_info_frame,
            text="Recommended Action: ---",
            font=Theme.FONT_BODY,
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_PRIMARY,
            wraplength=500,
            justify='left')
        self.action_label.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Model info
        self.model_info_label = tk.Label(self.disease_info_frame,
            text="Model: ---",
            font=Theme.FONT_SMALL,
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_SECONDARY)
        self.model_info_label.pack(anchor='w', padx=10, pady=(0, 10))
    
    def _create_chart_section(self, parent):
        """Create probability distribution chart"""
        chart_frame = tk.Frame(parent, bg=Theme.BG_MEDIUM)
        chart_frame.pack(fill='both', expand=True)
        
        chart_title = tk.Label(chart_frame,
            text="📊 Probability Distribution",
            font=Theme.FONT_HEADING,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY)
        chart_title.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Matplotlib figure
        self.fig = Figure(figsize=(8, 5), facecolor=Theme.BG_MEDIUM)
        self.ax = self.fig.add_subplot(111)
        self._setup_empty_chart()
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True, padx=15, pady=(0, 15))
    
    def _setup_empty_chart(self):
        """Setup empty chart placeholder"""
        self.ax.clear()
        self.ax.set_facecolor(Theme.BG_LIGHT)
        self.ax.text(0.5, 0.5, 'Load an image to see probability distribution',
            ha='center', va='center',
            fontsize=12, color=Theme.TEXT_SECONDARY,
            transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout()
    
    def _create_status_bar(self):
        """Create status bar"""
        status_frame = tk.Frame(self.main_frame, bg=Theme.BG_DARK)
        status_frame.pack(fill='x', pady=(10, 0))
        
        self.status_label = tk.Label(status_frame,
            text="⏳ Initializing...",
            font=Theme.FONT_BODY,
            bg=Theme.BG_DARK,
            fg=Theme.ACCENT)
        self.status_label.pack(side='left')
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame,
            style='Accent.Horizontal.TProgressbar',
            mode='determinate',
            length=200)
        self.progress.pack(side='right')
    
    def _load_models_async(self):
        """Load models in background"""
        def load():
            def callback(msg, progress):
                self.root.after(0, lambda: self._update_status(msg, progress))
            
            results = self.engine.load_all_models(callback)
            
            # Update UI
            self.root.after(0, lambda: self._on_models_loaded(results))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_models_loaded(self, results):
        """Handle model loading completion"""
        loaded = [k for k, v in results.items() if v]
        failed = [k for k, v in results.items() if not v]
        
        if loaded:
            self.status_label.config(
                text=f"✅ Models ready: {', '.join(loaded)}",
                fg=Theme.SUCCESS)
            
            # Update model selector
            self.model_combo['values'] = loaded
            if self.engine.current_model:
                self.selected_model.set(self.engine.current_model)
        else:
            self.status_label.config(
                text="❌ No models loaded! Check model files.",
                fg=Theme.ERROR)
        
        self.progress['value'] = 100
    
    def _update_status(self, message: str, progress: float = None):
        """Update status bar"""
        self.status_label.config(text=message)
        if progress is not None:
            self.progress['value'] = progress * 100
    
    def _load_image(self):
        """Load image from file dialog"""
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('All files', '*.*')
        ]
        
        filepath = filedialog.askopenfilename(
            title='Select Image',
            filetypes=filetypes
        )
        
        if filepath:
            self._display_image(filepath)
    
    def _display_image(self, filepath: str):
        """Display image in the preview area"""
        try:
            self.current_image_path = filepath
            
            # Load and resize image for display
            img = Image.open(filepath)
            img.thumbnail((450, 450), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo, text='')
            
            # Enable classify button
            self.classify_btn.config(state='normal')
            
            # Auto-classify
            self._classify_current_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def _classify_current_image(self):
        """Classify the current image"""
        if not self.current_image_path:
            return
        
        self._update_status("🔬 Analyzing image...", 0.5)
        
        # Run classification
        result = self.engine.classify_image(
            self.current_image_path,
            self.selected_model.get()
        )
        
        if 'error' in result:
            self._update_status(f"❌ Error: {result['error']}", 1.0)
            return
        
        self.current_result = result
        self._display_result(result)
        self._update_chart(result)
        
        # Add to history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'image': self.current_image_path,
            'result': result
        })
        
        self._update_status("✅ Analysis complete", 1.0)
    
    def _display_result(self, result: Dict):
        """Display classification result"""
        pred_class = result['predicted_class']
        confidence = result['confidence']
        is_valid = result['is_valid']
        is_rejected = result.get('is_rejected', False)
        
        # Get disease info
        info = DISEASE_INFO.get(pred_class, {})
        severity = info.get('severity', 'Unknown')
        action = info.get('action', 'Consult an expert')
        color = info.get('color', Theme.TEXT_PRIMARY)
        
        # Check confidence threshold
        threshold = self.confidence_threshold.get()
        
        if is_rejected:
            # Image rejected by pre-validator (not a sugarcane image)
            rejection_reason = result.get('rejection_reason', 'Image does not appear to be a sugarcane leaf')
            validation_scores = result.get('validation_scores', {})
            suggestions = result.get('suggestions', [])
            
            self.pred_class_label.config(
                text="🚫 Image Rejected",
                fg=Theme.ERROR)
            self.confidence_label.config(
                text=f"Reason: {rejection_reason}",
                fg=Theme.WARNING)
            self.severity_label.config(text="Validation Failed")
            
            # Show validation details and suggestions
            details_text = "Please upload a clear image of a sugarcane leaf.\n"
            if validation_scores:
                details_text += f"Scores: Quality={validation_scores.get('quality', 0):.0%}, "
                details_text += f"Vegetation={validation_scores.get('vegetation', 0):.0%}, "
                details_text += f"Sugarcane={validation_scores.get('sugarcane', 0):.0%}"
            if suggestions:
                details_text += f"\n• {suggestions[0]}" if suggestions else ""
            self.action_label.config(text=details_text)
            
        elif not is_valid:
            # Passed pre-validation but model is uncertain
            self.pred_class_label.config(
                text="⚠️ Uncertain Prediction",
                fg=Theme.WARNING)
            self.confidence_label.config(
                text=f"Low confidence: {confidence:.1%} - Model is uncertain",
                fg=Theme.WARNING)
            self.severity_label.config(text="Severity: N/A")
            self.action_label.config(text="Try a clearer image or consult an expert")
        elif confidence < threshold:
            # Below threshold
            self.pred_class_label.config(
                text=f"⚠️ {pred_class}",
                fg=Theme.WARNING)
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1%} (below {threshold:.0%} threshold)",
                fg=Theme.WARNING)
            self.severity_label.config(text=f"Severity: {severity}")
            self.action_label.config(text=f"⚠️ Low confidence - {action}")
        else:
            # Good prediction
            self.pred_class_label.config(
                text=f"✓ {pred_class}",
                fg=color if pred_class != 'Healthy' else Theme.SUCCESS)
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1%}",
                fg=Theme.SUCCESS if confidence > 0.9 else Theme.TEXT_PRIMARY)
            self.severity_label.config(text=f"Severity: {severity}")
            self.action_label.config(text=f"Recommended: {action}")
        
        # Update confidence bar
        self._draw_confidence_bar(confidence, is_valid, is_rejected)
        
        # Update model info
        model_name = result['model_used']
        model_info = {
            'student': 'Distilled Student (93.2% acc, 6.2M params)',
            'meta_mlp': 'Meta-MLP Ensemble (96.6% acc)',
            'maxvit': 'CustomMaxViT Backbone (95.4% acc, 24.8M params)'
        }
        self.model_info_label.config(text=f"Model: {model_info.get(model_name, model_name)}")
    
    def _draw_confidence_bar(self, confidence: float, is_valid: bool, is_rejected: bool = False):
        """Draw confidence progress bar"""
        self.confidence_canvas.delete('all')
        
        width = self.confidence_canvas.winfo_width()
        height = 20
        
        if width < 10:
            width = 500
        
        # Background
        self.confidence_canvas.create_rectangle(0, 0, width, height, fill=Theme.BG_LIGHT, outline='')
        
        # Determine color based on state
        if is_rejected:
            color = Theme.ERROR
            # Show rejection pattern (striped bar)
            for i in range(0, width, 20):
                self.confidence_canvas.create_rectangle(i, 0, min(i+10, width), height, fill=Theme.ERROR, outline='')
            return  # Don't draw confidence bar for rejected images
        elif not is_valid:
            color = Theme.WARNING
        elif confidence > 0.9:
            color = Theme.SUCCESS
        elif confidence > 0.7:
            color = Theme.INFO
        elif confidence > 0.5:
            color = Theme.WARNING
        else:
            color = Theme.ERROR
        
        # Fill bar
        fill_width = int(width * confidence)
        self.confidence_canvas.create_rectangle(0, 0, fill_width, height, fill=color, outline='')
        
        # Threshold marker
        threshold = self.confidence_threshold.get()
        threshold_x = int(width * threshold)
        self.confidence_canvas.create_line(threshold_x, 0, threshold_x, height, fill=Theme.ACCENT, width=2)
    
    def _update_chart(self, result: Dict):
        """Update probability distribution chart"""
        self.ax.clear()
        
        # Handle rejected images differently
        if result.get('is_rejected', False):
            # Show validation scores instead of classification probabilities
            validation_scores = result.get('validation_scores', {})
            if validation_scores:
                labels = ['Image Quality', 'Vegetation', 'Sugarcane Match']
                values = [
                    validation_scores.get('quality', 0),
                    validation_scores.get('vegetation', 0),
                    validation_scores.get('sugarcane', 0)
                ]
                
                y_pos = np.arange(len(labels))
                colors = [Theme.ERROR if v < 0.3 else Theme.WARNING if v < 0.5 else Theme.SUCCESS for v in values]
                
                bars = self.ax.barh(y_pos, values, color=colors, height=0.7)
                self.ax.set_yticks(y_pos)
                self.ax.set_yticklabels(labels, fontsize=9)
                self.ax.set_xlabel('Validation Score', fontsize=10, color=Theme.TEXT_PRIMARY)
                self.ax.set_xlim(0, 1)
                self.ax.invert_yaxis()
                self.ax.set_title('Image Validation Scores', fontsize=11, color=Theme.ERROR)
                
                # Add threshold line at 0.5
                self.ax.axvline(x=0.5, color=Theme.ACCENT, linestyle='--', linewidth=1.5, label='Pass Threshold')
            else:
                self.ax.text(0.5, 0.5, 'Image Rejected\n\nNo validation data', 
                            ha='center', va='center', fontsize=14, color=Theme.ERROR)
            
            self.ax.set_facecolor(Theme.BG_LIGHT)
            self.ax.tick_params(colors=Theme.TEXT_PRIMARY)
            for spine in self.ax.spines.values():
                spine.set_color(Theme.BG_LIGHT)
            self.fig.tight_layout()
            self.chart_canvas.draw()
            return
        
        probs = result['all_probabilities']
        
        # Sort by probability
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        classes = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(classes))
        
        # Color bars based on confidence
        colors = []
        threshold = self.confidence_threshold.get()
        for i, (cls, val) in enumerate(sorted_items):
            if i == 0:  # Top prediction
                if val > threshold:
                    colors.append(Theme.SUCCESS)
                else:
                    colors.append(Theme.WARNING)
            else:
                colors.append(Theme.CHART_COLORS[i % len(Theme.CHART_COLORS)])
        
        # Plot
        bars = self.ax.barh(y_pos, values, color=colors, height=0.7)
        
        # Styling
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(classes, fontsize=9)
        self.ax.set_xlabel('Probability', fontsize=10, color=Theme.TEXT_PRIMARY)
        self.ax.set_xlim(0, 1)
        self.ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            label_x = width + 0.01 if width < 0.9 else width - 0.08
            color = Theme.TEXT_PRIMARY if width < 0.9 else Theme.BG_DARK
            self.ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}',
                va='center', ha='left' if width < 0.9 else 'right',
                fontsize=8, color=color)
        
        # Add threshold line
        self.ax.axvline(x=threshold, color=Theme.ACCENT, linestyle='--', 
            linewidth=1.5, label=f'Threshold ({threshold:.0%})')
        
        # Style axes
        self.ax.set_facecolor(Theme.BG_LIGHT)
        self.ax.tick_params(colors=Theme.TEXT_PRIMARY)
        for spine in self.ax.spines.values():
            spine.set_color(Theme.BG_LIGHT)
        
        self.ax.legend(loc='lower right', fontsize=8)
        
        self.fig.tight_layout()
        self.chart_canvas.draw()
    
    def _on_threshold_change(self, value):
        """Handle threshold slider change"""
        self.threshold_value_label.config(
            text=f"Current: {float(value):.0%}")
        
        # Re-display current result if exists
        if self.current_result:
            self._display_result(self.current_result)
            self._draw_confidence_bar(
                self.current_result['confidence'],
                self.current_result['is_valid'],
                self.current_result.get('is_rejected', False))
    
    def _on_model_change(self, event):
        """Handle model selection change"""
        new_model = self.selected_model.get()
        self.engine.current_model = new_model
        
        # Re-classify if image is loaded
        if self.current_image_path:
            self._classify_current_image()
    
    def _batch_process(self):
        """Process a folder of images"""
        folder = filedialog.askdirectory(title='Select Folder with Images')
        
        if not folder:
            return
        
        # Find all images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in Path(folder).iterdir() 
                  if f.suffix.lower() in image_extensions]
        
        if not images:
            messagebox.showinfo("No Images", "No image files found in selected folder")
            return
        
        # Process in background
        def process():
            results = []
            total = len(images)
            
            for i, img_path in enumerate(images):
                self.root.after(0, lambda i=i: self._update_status(
                    f"Processing {i+1}/{total}...", (i+1)/total))
                
                result = self.engine.classify_image(str(img_path))
                result['filename'] = img_path.name
                results.append(result)
            
            # Save results
            output_path = Path(folder) / f'classification_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.root.after(0, lambda: self._on_batch_complete(results, output_path))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _on_batch_complete(self, results, output_path):
        """Handle batch processing completion"""
        # Count results
        total = len(results)
        valid = sum(1 for r in results if r.get('is_valid', False))
        above_threshold = sum(1 for r in results 
            if r.get('confidence', 0) >= self.confidence_threshold.get())
        
        self._update_status(f"✅ Batch complete: {total} images processed", 1.0)
        
        messagebox.showinfo("Batch Complete",
            f"Processed {total} images\n\n"
            f"Valid sugarcane images: {valid}\n"
            f"Above confidence threshold: {above_threshold}\n\n"
            f"Results saved to:\n{output_path}")


# =============================================================================
# SPLASH SCREEN
# =============================================================================

class SplashScreen:
    """Animated splash screen"""
    
    def __init__(self, root):
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        
        # Center on screen
        width, height = 500, 300
        x = (self.window.winfo_screenwidth() - width) // 2
        y = (self.window.winfo_screenheight() - height) // 2
        self.window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Dark background
        self.window.configure(bg=Theme.BG_DARK)
        
        # Content
        frame = tk.Frame(self.window, bg=Theme.BG_DARK)
        frame.pack(expand=True, fill='both', padx=30, pady=30)
        
        # Logo/title
        title = tk.Label(frame,
            text="🌿",
            font=('Segoe UI', 48),
            bg=Theme.BG_DARK,
            fg=Theme.ACCENT)
        title.pack(pady=(20, 10))
        
        name = tk.Label(frame,
            text="Sugarcane Disease Classifier",
            font=Theme.FONT_TITLE,
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_PRIMARY)
        name.pack()
        
        version = tk.Label(frame,
            text="15-COIN Ensemble System v1.0",
            font=Theme.FONT_SUBTITLE,
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY)
        version.pack(pady=(5, 20))
        
        # Progress
        self.progress = ttk.Progressbar(frame,
            style='Accent.Horizontal.TProgressbar',
            mode='indeterminate',
            length=300)
        self.progress.pack(pady=10)
        self.progress.start(15)
        
        self.status = tk.Label(frame,
            text="Loading...",
            font=Theme.FONT_BODY,
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY)
        self.status.pack()
    
    def update_status(self, text):
        self.status.config(text=text)
        self.window.update()
    
    def close(self):
        self.progress.stop()
        self.window.destroy()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    root = tk.Tk()
    root.withdraw()  # Hide main window initially
    
    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')
    
    # Show splash
    splash = SplashScreen(root)
    splash.update_status("Initializing...")
    
    # Small delay for splash
    root.after(500, lambda: splash.update_status("Loading models..."))
    root.after(1500, lambda: splash.update_status("Preparing interface..."))
    
    def start_app():
        splash.close()
        root.deiconify()  # Show main window
        app = DiseaseClassifierGUI(root)
    
    root.after(2000, start_app)
    root.mainloop()


if __name__ == '__main__':
    main()
