# Sprint 3-SEG: Segmentation-Aware Pipeline Rebuild

## Status: 📋 AWAITING APPROVAL — v2.1 (Revised)

---

## 1. Executive Summary

**What**: Rebuild the entire training pipeline from scratch with segmentation integrated into every backbone, using existing trained models as pretrained weights.

**Why**: The heuristic image validator (color/vein/edge analysis) has irreconcilable overlap between real diseased sugarcane (scores 0.54–0.62) and non-sugarcane images (0.51–0.62). No threshold works. We need every backbone to **understand spatial structure** — what's leaf, what's disease, what's background — not just classify the whole image blindly.

**How**: Each backbone gets a **dual-head architecture** (segmentation + classification) trained with the current `*_final.pth` weights as initialization. The segmentation head teaches the network WHERE to look, the classification head tells it WHAT it sees. Then the enhanced 12-stage ensemble pipeline is rebuilt with additional stages for coverage.

**Preserves**: ALL existing code, checkpoints, ensemble results, plots — nothing deleted. New pipeline lives in `V2_segmentation/` alongside original.

**Revised in v2.0**: Concrete per-model GPU memory budgets based on profiled data, multi-layer pseudo-label noise mitigation protocol, and strict evaluation isolation with per-stage holdout enforcement.

---

## 2. Current State (Baseline)

### 2.1 Individual Backbone Performance (Stage 1)

| Backbone | Test Acc | Embedding Dim | Tier |
|----------|----------|---------------|------|
| CustomCSPDarkNet | 96.04% | 1024 | 🟢 Top |
| CustomGhostNetV2 | 94.53% | 1280 | 🟢 Top |
| CustomDynamicConvNet | 94.53% | 1024 | 🟢 Top |
| CustomMobileOne | 94.25% | 384 | 🟢 Top |
| CustomRegNet | 93.87% | 1536 | 🟡 Mid |
| CustomInceptionV4 | 93.97% | 512 | 🟡 Mid |
| CustomResNetMish | 93.59% | 2048 | 🟡 Mid |
| CustomDenseNetHybrid | 93.69% | 512 | 🟡 Mid |
| CustomEfficientNetV4 | 93.31% | 1280 | 🟡 Mid |
| CustomConvNeXt | 92.93% | 768 | 🟡 Mid |
| CustomDeiTStyle | 91.33% | 768 | 🟠 Low |
| CustomViTHybrid | 91.23% | 768 | 🟠 Low |
| CustomCoAtNet | 86.43% | 768 | 🔴 Weak |
| CustomSwinTransformer | 85.30% | 1024 | 🔴 Weak |
| CustomMaxViT | 85.30% | 768 | 🔴 Weak |

### 2.2 Ensemble Performance (Stages 2–7)

| Stage | Method | Test Acc |
|-------|--------|----------|
| Stage 5 | Mixture of Experts | 95.48% |
| Stage 6 | Meta-Learner (MLP) | **96.61%** |
| Stage 6 | Meta-Learner (XGBoost) | 96.42% |
| Stage 7 | Distilled Student | 93.21% |

### 2.3 Current Validation System (Heuristic — FAILING)

| Test Set | Result |
|----------|--------|
| 38 non-sugarcane images | 78.9% rejected (8 leaked at threshold 0.50) |
| After threshold tuning to 0.55 | 100% rejected BUT 8.5–20.8% real sugarcane also blocked |
| Red_rot (all 10 samples) | Score 0.53–0.60 — heuristics think red/brown = "not sugarcane" |
| Wilt / Pokkah_boeng | Score 0.00–0.63 — severely wilted leaves rejected |

**Root cause**: Pixel-level heuristics cannot distinguish "reddish diseased sugarcane" from "random red image". A learned model can.

### 2.4 Hardware Specification (Profiled)

| Resource | Value |
|----------|-------|
| GPU | NVIDIA RTX 4500 Ada Generation |
| VRAM Total | 24.0 GB |
| VRAM Free (typical) | 22.7 GB |
| PyTorch | 2.6.0 + CUDA 12.4 |
| AMP (FP16) | Supported and verified |
| Gradient Checkpointing | Supported for all backbones |

---

## 3. GPU Memory Profiling (Measured)

> All measurements taken with AMP (FP16 mixed precision), AdamW optimizer, full forward+backward+optimizer step. This data drives every batch size and gradient accumulation decision in this plan.

### 3.1 V1 Backbone Memory Profile (Classification Only)

| Backbone | Params | BS=2 | BS=4 | BS=8 | BS=16 | BS=32 | Memory Tier |
|----------|--------|------|------|------|-------|-------|-------------|
| CustomViTHybrid | 136.2M | 2.57 GB | 2.57 GB | 2.84 GB | 4.34 GB | 7.37 GB | 🔴 HEAVY |
| CustomCoAtNet | 117.4M | 2.23 GB | 2.26 GB | 3.36 GB | 5.56 GB | 9.99 GB | 🔴 HEAVY |
| CustomMaxViT | 106.4M | 2.03 GB | 2.21 GB | 3.37 GB | 5.69 GB | 10.35 GB | 🔴 HEAVY |
| CustomDeiTStyle | 93.9M | 1.20 GB | 1.77 GB | 2.01 GB | 3.11 GB | 5.34 GB | 🔴 HEAVY |
| CustomSwinTransformer | 89.3M | 1.71 GB | 1.71 GB | 2.05 GB | 3.21 GB | 5.54 GB | 🔴 HEAVY |
| CustomDynamicConvNet | 72.6M | 1.72 GB | 2.65 GB | 4.49 GB | 8.20 GB | 15.61 GB | 🟠 HIGH |
| CustomConvNeXt | 27.8M | 0.41 GB | 0.54 GB | 0.77 GB | 1.23 GB | 2.18 GB | 🟡 MEDIUM |
| CustomResNetMish | 23.5M | 0.37 GB | 0.49 GB | 0.73 GB | 1.20 GB | 2.16 GB | 🟡 MEDIUM |
| CustomRegNet | 18.7M | 0.38 GB | 0.56 GB | 0.93 GB | 1.66 GB | 3.08 GB | 🟡 MEDIUM |
| CustomGhostNetV2 | 9.6M | 0.22 GB | 0.33 GB | 0.55 GB | 0.99 GB | 1.85 GB | 🟢 LIGHT |
| CustomMobileOne | 4.7M* | 0.06 GB | 0.07 GB | 0.10 GB | 0.13 GB | 0.20 GB | 🟢 LIGHT |
| CustomCSPDarkNet | 3.9M | 0.70 GB | 1.34 GB | 2.62 GB | 5.16 GB | 10.27 GB | 🟠 HIGH** |
| CustomInceptionV4 | 2.4M | 0.20 GB | 0.37 GB | 0.69 GB | 1.33 GB | 2.62 GB | 🟢 LIGHT |
| CustomDenseNetHybrid | 2.1M | 0.14 GB | 0.24 GB | 0.44 GB | 0.84 GB | 1.63 GB | 🟢 LIGHT |
| CustomEfficientNetV4 | 1.9M | 0.07 GB | 0.11 GB | 0.19 GB | 0.36 GB | 0.68 GB | 🟢 LIGHT |

*MobileOne has 9.9M params during training (multi-branch), 4.7M after reparameterization.
**CSPDarkNet has low param count but high activation memory due to cross-stage partial connections.

### 3.2 V2 Dual-Head Memory Budget (Estimated)

Adding a segmentation decoder (ASPP + skip connections + upsampling) adds overhead that varies by tier. To prevent a universal decoder width from pushing HEAVY models into OOM, **decoder channels are scaled per tier**:

| Memory Tier | Decoder Channels | ASPP Channels | Est. Decoder Params | Est. Decoder Activations (BS=8) | Total Overhead |
|-------------|-----------------|---------------|--------------------|---------------------------------|----------------|
| 🟢 LIGHT | 256 | 256 | ~8.0M | ~1.5 GB | ~1.4× |
| 🟡 MEDIUM | 256 | 256 | ~8.0M | ~1.5 GB | ~1.5× |
| 🟠 HIGH | 192 | 192 | ~4.5M | ~1.0 GB | ~1.6× |
| 🔴 HEAVY | 128 | 128 | ~2.2M | ~0.6 GB | ~1.8×* |

*HEAVY tier uses gradient checkpointing **in the decoder as well**, reducing activation memory by ~40%.

Additional per-image costs (all tiers):
- **Segmentation loss computation**: ~0.3–0.6 GB (mask tensors at 224×224×5, multi-label sigmoid)
- **Total overhead multiplier**: 1.4–1.8× the classification-only memory (tier-dependent)

**Dual-head memory estimates (Phase B joint training, BS shown is per-GPU effective):**

| Memory Tier | Backbones | V1 @ BS=8 | Est. V2 @ BS=8 | Safe V2 BS | Grad Accum Steps | Effective BS |
|-------------|-----------|-----------|----------------|------------|-----------------|-------------|
| 🔴 HEAVY | ViTHybrid, CoAtNet, MaxViT | 2.8–3.4 GB | 5.5–6.5 GB | **4** | **8** | **32** |
| 🔴 HEAVY | DeiTStyle, SwinTransformer | 2.0–2.1 GB | 3.8–4.2 GB | **8** | **4** | **32** |
| 🟠 HIGH | DynamicConvNet | 4.5 GB | 7.5–9.0 GB | **4** | **8** | **32** |
| 🟠 HIGH | CSPDarkNet | 2.6 GB | 4.5–5.5 GB | **8** | **4** | **32** |
| 🟡 MEDIUM | ConvNeXt, ResNetMish, RegNet | 0.7–0.9 GB | 1.5–2.0 GB | **16** | **2** | **32** |
| 🟢 LIGHT | GhostNetV2, MobileOne, InceptionV4, DenseNetHybrid, EfficientNetV4 | 0.1–0.7 GB | 0.3–1.5 GB | **32** | **1** | **32** |

**CRITICAL RULE**: Every backbone achieves effective batch size 32 through gradient accumulation. No model is trained with a smaller effective batch than any other. The only difference is how many physical micro-batches compose one optimizer step.

### 3.3 GPU Memory Safety Protocol

```
BEFORE every training run:
  1. torch.cuda.empty_cache()
  2. gc.collect()
  3. Log torch.cuda.memory_allocated() and torch.cuda.memory_reserved()
  4. Assert free_memory > estimated_peak × 1.15 (15% safety margin)
  5. If assertion fails → halve batch size, double grad_accum_steps

DURING training (per-epoch):
  1. Monitor torch.cuda.max_memory_allocated() after first batch
  2. If peak > 22.0 GB (leaving 2 GB headroom for OS/driver) → OOM_RISK flag
  3. On OOM_RISK: checkpoint current state, reduce BS by 50%, resume
  4. All intermediate tensors use .detach() where not needed for autograd

ON OOM EXCEPTION:
  1. torch.cuda.empty_cache(); gc.collect()
  2. Log the batch size and model that caused OOM to oom_log.json
  3. Halve batch size, double grad_accum_steps
  4. Resume from last saved checkpoint (saved every epoch)
  5. If BS=1 still OOMs → enable gradient checkpointing for that backbone
  6. If gradient checkpointing + BS=1 still OOMs → FATAL: skip backbone, log reason

GRADIENT CHECKPOINTING (for tier 🔴 HEAVY models):
  - Pre-emptively enabled for: ViTHybrid, CoAtNet, MaxViT
  - Uses torch.utils.checkpoint.checkpoint() on each transformer block
  - Trades ~30% more compute time for ~40% less memory
  - With gradient checkpointing, all HEAVY models fit at BS=4 comfortably
```

---

## 4. Architecture: Segmentation-Aware Dual-Head Pipeline

### 4.1 Core Concept

```
                    ┌──────────────────────────────┐
                    │     EXISTING BACKBONE         │
                    │  (pretrained from V1 weights) │
                    │                               │
Input Image ──────► │  Stem → Stage1 → Stage2 →     │
   224×224          │  Stage3 → Stage4              │
                    │       │           │            │
                    │       │     Feature Maps       │
                    │       │    (multi-scale)       │
                    └───────┼───────────┼────────────┘
                            │           │
                 ┌──────────┘           └──────────┐
                 ▼                                  ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │  SEGMENTATION HEAD  │          │  CLASSIFICATION HEAD │
    │                     │          │                      │
    │  ASPP Module        │          │  Global Avg Pool     │
    │  Skip Connections   │          │  FC Layers           │
    │  Upsampling Path    │          │  Softmax (13 cls)    │
    │  Sigmoid (per-pixel)│          │                      │
    └──────────┬──────────┘          └──────────┬───────────┘
               │                                │
               ▼                                ▼
         Segmentation Mask              Class Probabilities
         224×224 (5 channels)            (13 classes)
         - Ch0: Background
         - Ch1: Healthy Plant Tissue
         - Ch2: Structural Anomaly
         - Ch3: Surface Disease Sign
         - Ch4: Tissue Degradation
```

### 4.2 Why Dual-Head Works

1. **Shared features**: The backbone learns features useful for BOTH tasks — this creates richer representations
2. **Spatial awareness**: Segmentation forces the backbone to understand WHERE objects are, not just what the overall image "looks like"
3. **Disease localization**: The segmentation head learns to identify WHERE each disease manifests — surface lesions, structural anomalies, tissue degradation — which directly improves classification
4. **Validation gate**: If the segmentation head finds <8% sugarcane tissue across ALL channels, the image is not sugarcane — no ambiguous thresholds

### 4.3 Multi-Scale Feature Extraction

Each backbone already has progressive stages that downsample spatial resolution:

```
Stage 1: 56×56  (low-level: edges, textures)    → Skip to Decoder Level 4
Stage 2: 28×28  (mid-level: leaf shapes)         → Skip to Decoder Level 3
Stage 3: 14×14  (high-level: patterns)           → Skip to Decoder Level 2
Stage 4: 7×7    (semantic: disease features)     → Decoder Level 1 + Classifier
```

The segmentation decoder uses **skip connections** from these intermediate stages for precise pixel-level predictions, while the classifier uses only the final stage.

---

## 5. Pseudo-Label Generation & Noise Mitigation

> **Risk Assessment**: Pseudo-label noise is rated **HIGH** — noisy masks can propagate errors into the backbone, degrading both segmentation quality AND classification accuracy. This section specifies a concrete multi-layer defence.

### 5.1 Generation Methods (Three Independent Sources)

```
Method 1: GrabCut + Color Segmentation
  ├── Use green channel dominance to seed foreground/background
  ├── OpenCV GrabCut refines boundaries
  ├── Morphological operations to clean masks
  └── Strength: Reliable for healthy green leaves
  └── Weakness: Fails on heavily diseased/wilted tissue

Method 2: GradCAM-Guided Pseudo-Labels (FROM EXISTING MODELS)
  ├── Run each V1 backbone on training images
  ├── Extract GradCAM activation maps from final conv layer
  ├── High-activation regions = "important for classification" = leaf/disease
  ├── Threshold + morphology → pseudo-mask
  ├── ENSEMBLE the GradCAMs from all 15 backbones for robust masks
  └── Strength: Uses learned knowledge; good at disease regions
  └── Weakness: GradCAM can be noisy/diffuse on some architectures

Method 3: SAM (Segment Anything Model) — if available
  ├── Zero-shot segmentation with automatic prompts
  ├── Produces high-quality masks without training
  └── Strength: Best quality boundaries
  └── Weakness: Requires SAM weights (~2.4GB); may not distinguish leaf vs non-leaf
```

### 5.2 Multi-Layer Noise Mitigation Protocol

**Layer 1: Multi-Source Fusion with Confidence Weighting**
```
For each pixel in each image:
  1. Collect predictions from all 3 methods (or 2 if SAM unavailable)
  2. Weighted pixel-wise vote:
     - GrabCut weight: 0.3 (reliable for background, weak on lesions)
     - GradCAM weight: 0.5 (best signal from learned models)
     - SAM weight: 0.2 (boundaries only, not semantic class)
  3. Pixel class = argmax(weighted votes)
  4. Pixel confidence = max(weighted votes) / sum(weighted votes)
  5. Mark pixels with confidence < 0.6 as UNCERTAIN (channel -1)
```

**Layer 2: Per-Image Quality Score**
```
For each generated mask:
  1. connected_components = count connected regions of "leaf" class
  2. largest_component_ratio = largest_component_area / total_leaf_area
  3. boundary_smoothness = perimeter² / (4π × area)  [circularity proxy]
  4. uncertain_ratio = uncertain_pixels / total_pixels

  QUALITY_SCORE = (
      0.4 × largest_component_ratio           # Prefer single coherent region
    + 0.3 × (1 - uncertain_ratio)             # Penalize uncertainty
    + 0.2 × min(1.0, plant_ratio / 0.2)      # Expect ≥20% sugarcane tissue (any channel)
    + 0.1 × (1 - boundary_smoothness_penalty) # Prefer smooth boundaries
  )
  NOTE: plant_ratio = Ch1 + Ch2 + Ch3 + Ch4 (any sugarcane-related pixel,
        not just green leaf). This ensures Smut, Grassy_shoot, Wilt images
        with little visible leaf are NOT penalized.

  TIER assignment:
    quality ≥ 0.8  → TIER_A (high confidence, use with full weight)
    quality ≥ 0.5  → TIER_B (medium confidence, use with reduced loss weight)
    quality < 0.5  → TIER_C (low confidence, EXCLUDE from training)
```

**Layer 3: Class-Specific Validation Rules (Disease-Aware)**
```
Per disease class, apply domain-specific sanity checks based on HOW each
disease actually manifests (see §12.1 Disease ↔ Channel Map).

"plant_ratio" = Ch1 + Ch2 + Ch3 + Ch4 (any sugarcane-related pixel, not just leaf)
"anomaly_ratio" = Ch2 / total (structural abnormalities)
"surface_ratio" = Ch3 / total (visible marks on tissue)
"degradation_ratio" = Ch4 / total (wilting, rotting, yellowing)

Healthy:               plant_ratio > 0.40, surface+anomaly+degradation < 0.05
Red_rot:               degradation_ratio > 0.03 OR surface_ratio > 0.03
                       (may show external discoloration OR internal rot signs)
Brown_spot:            surface_ratio > 0.03 (spots on leaf surface)
Mosaic:                surface_ratio > 0.03 (mosaic pattern on tissue)
Smut:                  anomaly_ratio > 0.02 (whip structure present)
                       NOTE: leaf/lesion ratios are IRRELEVANT for Smut
Grassy_shoot_disease:  anomaly_ratio > 0.02 (proliferating thin shoots)
                       NOTE: standard leaf coverage rules DO NOT apply
Wilt:                  degradation_ratio > 0.05 (whole-tissue drying)
                       plant_ratio CAN be very low (≥ 0.10) — plant is dying
Pokkah_boeng:          anomaly_ratio > 0.02 OR degradation_ratio > 0.02
                       (malformed growth point + possible tissue rot)
Leaf_flecking:         surface_ratio > 0.02 (small flecks present)
Leaf_scorching:        degradation_ratio > 0.03 (browning/drying margins)
Ring_spot:             surface_ratio > 0.03 (ring patterns visible)
Yellow_leaf_Disease:   degradation_ratio > 0.03 (yellowing on tissue)
Black_stripe:          surface_ratio > 0.03 (dark stripes on tissue)

UNIVERSAL RULE: plant_ratio > 0.08 for ALL classes (at least 8% of the image
  must be recognized as sugarcane in ANY channel — otherwise the image may
  not contain sugarcane at all)

If a mask violates its class rule → downgrade one tier (A→B, B→C)
```

**Layer 4: Human Spot-Check Protocol (NOT Optional)**
```
1. Randomly sample 10 images per class × 3 tiers = 390 images
2. Display: Original image | Generated mask overlay | Quality score + tier
3. For each image, mark: ✅ Accept | ⚠️ Marginal | ❌ Reject
4. Compute acceptance rates per tier:
   - TIER_A acceptance must be ≥ 90% or → regenerate with adjusted params
   - TIER_B acceptance must be ≥ 70% or → move all TIER_B to TIER_C
   - If TIER_C images > 30% of dataset → HALT, review generation pipeline
5. Save spot-check results to: pseudo_labels/audit/spot_check_results.json
6. This audit is BLOCKING — training cannot proceed until thresholds pass
```

**Layer 5: Training-Time Noise Handling**
```
During segmentation head training:
  1. TIER_A masks: loss_weight = 1.0 (full signal)
  2. TIER_B masks: loss_weight = 0.5 (reduced contribution)
  3. TIER_C masks: EXCLUDED entirely (not loaded, not augmented)
  4. UNCERTAIN pixels (channel -1): loss_weight = 0.0 (masked out)
     → Loss is only computed on pixels where we have confidence

  Concrete implementation:
    seg_loss = dice_loss(pred, target, weight_mask=confidence_mask)
    where confidence_mask[uncertain_pixels] = 0.0
    and confidence_mask *= tier_weight  # 1.0 for A, 0.5 for B
```

**Layer 6: Iterative Refinement (Post-Phase-A)**
```
After Phase A training (seg head on frozen backbone):
  1. Run the TRAINED seg head on all training images
  2. Compare trained predictions vs original pseudo-labels
  3. Where they agree → reinforce label confidence (potentially upgrade B→A)
  4. Where they disagree → FLAG for manual review (20 samples per class)
  5. Update pseudo-labels using trained model predictions (self-training)
  6. Re-run quality scoring on updated masks
  7. Log all changes to pseudo_labels/audit/refinement_log.json
  8. Phase B uses the REFINED masks, not the original ones

  CONVERGENCE CHECK: If refinement changes > 15% of pixels → run one more round
  MAX REFINEMENT ROUNDS: 3 (prevent infinite loop)
  DIVERGENCE GUARD: If round N has MORE pixel changes than round N-1, stop and use
                    round N-1 labels (refinement is making things worse)

  DRIFT TRACKING (logged + charted after every round):
    drift_pct[round] = changed_pixels / total_pixels × 100
    Save to: pseudo_labels/audit/refinement_drift.json
    Plot:    pseudo_labels/audit/refinement_drift_chart.tiff
    Example: Round 1: 22% → Round 2: 9% → Round 3: 4% (healthy convergence)

  OSCILLATION BLOCKING GATE (enforced after round 2):
    IF drift_pct[round] > drift_pct[round-1] for round ≥ 3:
      → HALT refinement, use round (N-1) labels (DIVERGENCE GUARD above)
    IF drift_pct[round] > 15% for any round ≥ 3:
      → BLOCKING: Refinement is NOT converging. Halt pipeline.
      → Manual review required: inspect which classes/images are oscillating.
      → Save oscillating image list to: pseudo_labels/audit/oscillating_samples.json
      → After manual fix, restart refinement from round 1.
```

### 5.3 Mask Storage Format

```
segmentation_masks/
  ├── train/
  │   ├── Black_stripe/
  │   │   ├── img_001_mask.png      # 5-channel PNG (BG, Healthy, Structural, Surface, Degradation)
  │   │   ├── img_001_conf.npy      # Per-pixel confidence [0.0–1.0]
  │   │   ├── img_001_tier.txt      # "A", "B", or "C"
  │   │   └── ...
  │   ├── Brown_spot/
  │   └── ...
  ├── val/                          # Same structure
  ├── test/                         # Masks generated but NEVER used for training
  └── audit/
      ├── spot_check_results.json
      ├── quality_scores.csv        # All images with scores + tiers
      ├── refinement_log.json
      └── tier_distribution.tiff    # Visualization of A/B/C distribution per class
```

---

## 6. Heatmap & Feature Analysis (Pre-Training Audit)

**Before** starting any V2 training, we perform a comprehensive analysis of what V1 models learned:

```
Analysis 1: GradCAM Heatmaps (all 15 backbones × all 13 classes)
  ├── Generate heatmaps for 10 images per class per backbone
  ├── Identify: Which backbones look at CORRECT regions?
  ├── Identify: Which backbones are "cheating" (looking at background)?
  ├── Output: 15 × 13 grid of average heatmaps
  └── Saved: analysis/gradcam_heatmaps/

Analysis 2: Feature Activation Statistics
  ├── Per-stage activation magnitudes
  ├── Dead neuron detection
  ├── Feature diversity analysis (are different backbones learning different things?)
  └── Saved: analysis/feature_stats/

Analysis 3: Error Pattern Analysis
  ├── Which images does EVERY backbone get wrong? (hard examples)
  ├── Which images does ONLY ONE backbone get wrong? (specialist opportunities)
  ├── Confusion matrix overlap between backbones
  └── Saved: analysis/error_patterns/

Analysis 4: Class-wise Attention Regions
  ├── For each disease class, WHERE do models focus?
  ├── Red_rot: Do models look at red lesions or green surrounding tissue?
  ├── Mosaic: Do models detect the mosaic pattern or just green color?
  └── This directly informs segmentation label categories
```

---

## 7. Training Strategy

### 7.1 Three-Phase Training (Per Backbone)

```
Phase A: Segmentation Head Training (backbone FROZEN)
  ├── Load V1 weights into backbone (e.g., CustomConvNeXt_final.pth)
  ├── Freeze all backbone parameters
  ├── Train ONLY the new segmentation decoder + ASPP
  ├── Loss: Dice + BCE on pseudo-masks (with tier weighting + uncertainty masking)
  ├── Epochs: 30  |  LR: 1e-3  |  Patience: 5
  ├── After completion: Run iterative refinement (Section 5.2, Layer 6)
  └── Purpose: Decoder learns to decode features without disturbing backbone

Phase B: Joint Fine-tuning (backbone UNFROZEN, both heads active)
  ├── Unfreeze backbone
  ├── Joint loss: λ_seg × L_seg + λ_cls × L_cls
  ├── λ_seg = 0.4, λ_cls = 0.6 (classification is primary task)
  ├── Uses REFINED pseudo-labels from post-Phase-A refinement
  ├── Epochs: 25  |  LR: 1e-5 (backbone), 1e-4 (heads)  |  Patience: 5
  └── Purpose: Backbone learns spatially-aware features for BOTH tasks

Phase C: Classification Refinement (seg head frozen, cls head fine-tuned)
  ├── Freeze segmentation decoder
  ├── Fine-tune classifier with hard example mining
  ├── Use segmentation masks to crop/weight loss on leaf regions
  ├── Epochs: 15  |  LR: 1e-6  |  Patience: 3
  └── Purpose: Squeeze last % of classification accuracy
```

### 7.2 Per-Backbone Training Schedule (Concrete)

Training order: Light → Medium → High → Heavy (build confidence on easy models first)

```
WAVE 1 — 🟢 LIGHT MODELS (BS=32, grad_accum=1, ~45 min each)
  Order: EfficientNetV4 → DenseNetHybrid → InceptionV4 → MobileOne → GhostNetV2
  Expected time: 5 × 45min = ~3.75 hours

WAVE 2 — 🟡 MEDIUM MODELS (BS=16, grad_accum=2, ~1.5h each)
  Order: ConvNeXt → ResNetMish → RegNet
  Expected time: 3 × 1.5h = ~4.5 hours

WAVE 3 — 🟠 HIGH MODELS (BS=8, grad_accum=4, ~2.5h each)
  Order: CSPDarkNet → DynamicConvNet
  Expected time: 2 × 2.5h = ~5 hours
  Note: CSPDarkNet has low params but very high activation memory

WAVE 4 — 🔴 HEAVY MODELS (BS=4, grad_accum=8, gradient checkpointing ON, ~4h each)
  Order: DeiTStyle → SwinTransformer → MaxViT → CoAtNet → ViTHybrid
  Expected time: 5 × 4h = ~20 hours
  Note: Gradient checkpointing pre-emptively enabled for all 5

TOTAL ESTIMATED TIME: ~33 hours (can run overnight across 2 nights)
```

### 7.3 Checkpoint Strategy

```
Per backbone, save:
  1. {backbone}_v2_phaseA_best.pth     — best seg-head-only model
  2. {backbone}_v2_phaseA_refined.pth  — after pseudo-label refinement
  3. {backbone}_v2_phaseB_best.pth     — best joint model
  4. {backbone}_v2_phaseC_best.pth     — best classifier-refined model
  5. {backbone}_v2_final.pth           — final model (copy of phaseC_best)

Each checkpoint contains:
  - model_state_dict
  - optimizer_state_dict
  - epoch
  - best_val_acc (classification)
  - best_val_iou (segmentation)
  - training_config (batch_size, grad_accum, lr, etc.)
  - pseudo_label_tier_distribution (how many A/B/C masks used)
  - run_seed (integer seed used for this run)
  - git_hash (output of `git rev-parse HEAD` at run start)
  - pytorch_version, cuda_version, cudnn_version
  - pip_freeze_hash (SHA-256 of `pip freeze` output)
  - timestamp_utc (ISO 8601)

Resume capability: Every phase starts by checking for existing checkpoints.
If interrupted, training resumes from last saved epoch automatically.
```

### 7.4 Reproducibility & Deterministic Seeds

```
AT THE START OF EVERY TRAINING RUN:
  RUN_SEED = int(os.environ.get('V2_RUN_SEED', 42))

  import random, numpy as np, torch
  random.seed(RUN_SEED)
  np.random.seed(RUN_SEED)
  torch.manual_seed(RUN_SEED)
  torch.cuda.manual_seed_all(RUN_SEED)

  # CuDNN determinism vs performance trade-off:
  torch.backends.cudnn.deterministic = True   # Exact reproducibility
  torch.backends.cudnn.benchmark = False       # Disable auto-tuner

  # NOTE: Full determinism has ~5-15% performance penalty. For production
  # training waves, set deterministic=False and benchmark=True for speed,
  # accepting minor non-determinism. Record the choice in checkpoint metadata.

  # For K-fold splits:
  StratifiedKFold(n_splits=5, shuffle=True, random_state=RUN_SEED)

  # For DataLoader worker seeding:
  def seed_worker(worker_id):
      worker_seed = RUN_SEED + worker_id
      np.random.seed(worker_seed)
      random.seed(worker_seed)

  DataLoader(..., worker_init_fn=seed_worker,
             generator=torch.Generator().manual_seed(RUN_SEED))

EVERY CHECKPOINT records: RUN_SEED + git_hash + pip_freeze_hash
→ Any run can be exactly reproduced from these three values.
```

---

## 8. Evaluation Isolation Protocol

> **Risk Assessment**: With 12 ensemble stages, 3 training phases, pseudo-label refinement, cascaded training, and adversarial boosting, there are **multiple vectors for data leakage and overfitting**. This section specifies strict isolation rules.

### 8.1 Data Partition Rules

```
PARTITION STRUCTURE (from V1, PRESERVED EXACTLY):
  ├── train/    80% of data  — ONLY partition used for model weight updates
  ├── val/      10% of data  — ONLY partition used for early stopping / model selection
  └── test/     10% of data  — SACRED. Used ONLY for final reported metrics.

ABSOLUTE RULES:
  1. Test set images are NEVER used for:
     - Pseudo-label generation (masks generated from train+val only)
     - GradCAM heatmap analysis (use train+val only)
     - Hyperparameter tuning
     - Model selection between variants
     - K-fold cross-validation
     - Cascaded training (Stage 9) hard example identification
     - Adversarial boosting (Stage 10) sample reweighting
     - Referee training (Stage 11)

  2. Validation set is used ONLY for:
     - Early stopping decisions during Phase A/B/C training
     - Model selection (which checkpoint to keep as "best")
     - Per-stage ensemble model selection
     - Pseudo-label quality assessment (NOT generation)

  3. Training set is used ONLY for:
     - Model weight updates (backpropagation)
     - Pseudo-label mask generation
     - GradCAM extraction for pseudo-label creation
     - K-fold cross-validation splits (within train set only)
```

### 8.2 Per-Stage Holdout Enforcement

Each ensemble stage requires its own data handling to prevent leakage:

```
STAGE 1 (Individual V2 Predictions):
  ├── Train: Phase A/B/C training
  ├── Val: Early stopping, checkpoint selection
  ├── Test: Final individual accuracy numbers
  └── No leakage vector (standard training)

STAGE 2-4 (Score Ensembles, Stacking, Fusion):
  ├── STACKING REQUIRES SPECIAL HANDLING:
  │   ├── Stacker models (LR, MLP, XGBoost) MUST be trained on out-of-fold predictions
  │   ├── Use 5-fold CV on the TRAINING SET ONLY to generate stacking features
  │   ├── For each fold: train 15 backbones on 4/5 of train, predict on 1/5
  │   ├── Concatenate 5 folds → training features for stacker
  │   ├── Stacker trained on these OOF predictions, validated on val set
  │   ├── Test set: predictions from models trained on FULL training set
  │   └── THIS PREVENTS THE STACKER FROM MEMORIZING TRAINING SET BACKBONE OUTPUTS
  └── Validation: stacker model selection only

STAGE 5 (Mixture of Experts):
  ├── Gating network trained on TRAINING SET OOF predictions (same as Stage 3)
  ├── Gating network validated on val set
  └── Test set used only for final metric

STAGE 6 (Meta-Learner):
  ├── Meta-learner input = outputs from Stages 1-5
  ├── CRITICAL: Stage 1-5 outputs must be OOF for training set
  ├── Meta-learner trained on training OOF features
  ├── Meta-learner model selected on val set
  └── Test set for final metric only

STAGE 7 (Knowledge Distillation):
  ├── Student trained on training set with teacher soft labels
  ├── Teacher soft labels generated by running teacher on training set (OK: teacher is frozen)
  ├── Student validated on val set (with teacher soft labels for val)
  └── Test set for final metric

STAGE 8 (Segmentation-Informed Ensemble):
  ├── Segmentation quality (IoU per backbone) computed on VALIDATION SET
  ├── IoU scores determine voting weights
  ├── Voting weights are FIXED after computing on val set
  ├── Applied to test set predictions
  └── No training involved — just weighted voting

STAGE 9 (Cascaded Sequential Training):
  ├── CRITICAL ISOLATION: Hard examples identified on TRAINING SET OOF predictions
  ├── Model_1: Best individual (e.g., CSPDarkNet) — already trained
  ├── Model_2 training data: images from TRAINING SET where Model_1 OOF was wrong
  │   ├── NOT: images where Model_1's regular predictions are wrong (that would leak)
  │   ├── Use K-fold OOF to identify hard examples: for each fold, get Model_1's
  │   │   prediction on held-out 1/5 → if wrong, mark as "hard"
  │   └── Model_2 trains on these hard examples with 5-fold OOF
  ├── Model_3 training data: images BOTH Model_1 OOF AND Model_2 OOF got wrong
  ├── Cascade threshold tuned on VALIDATION SET
  ├── Confidence thresholds (>0.95 use Model_1, etc.) tuned on val set
  └── Test set for final metric only

STAGE 10 (Adversarial Boosting):
  ├── Sample weights initialized uniformly on TRAINING SET
  ├── After each boosting round, weights updated based on TRAINING SET OOF errors
  ├── NOT based on training set direct predictions (would memorize)
  ├── Boosting rounds: 5-10, each round trains a new model variant
  ├── Each round uses 5-fold OOF to compute error rates
  ├── Final boosting weights tuned on val set
  └── Test set for final metric only

STAGE 11 (Cross-Architecture Referee):
  ├── Consensus/ambiguous regions identified from TRAINING SET OOF predictions
  │   (backbones disagree on > 40% of their OOF predictions for that sample)
  ├── Referee model trained on TRAINING SET ambiguous cases
  │   ├── Ambiguous = backbone disagreement ratio > 0.4
  │   ├── Identified using OOF predictions on training set
  │   └── Referee validated on val set ambiguous cases
  └── Test set for final metric only

STAGE 12 (V2 Distillation):
  ├── Teacher ensemble = Stages 1-11 combined (frozen, no updates)
  ├── Student trained on training set with teacher soft labels
  ├── Attention transfer loss uses teacher attention maps (frozen)
  ├── Student validated on val set
  └── Test set for final metric only
```

### 8.3 Leakage Detection Checks

```
AUTOMATED CHECKS (run after each stage):

  1. TRAIN-TEST OVERLAP CHECK:
     - Hash all file paths used in training vs test
     - Assert: intersection = ∅ (empty set)
     - Log: "Stage X: 0 train-test overlaps confirmed"

  2. OVERFIT DETECTION:
     - Compute train_acc - val_acc gap
     - If gap > 5%: WARNING, log and continue
     - If gap > 10%: HALT, investigate before proceeding

  3. INFORMATION FLOW AUDIT:
     - For stacking stages: verify OOF predictions used (not direct)
     - Check: stacker input features have exactly len(train_set) rows
     - Check: OOF prediction indices match K-fold split indices

  4. TEST SET VIRGINITY CHECK:
     - Before final evaluation: log SHA-256 hash of test set file list
     - Compare against hash from V1: must be identical
     - Confirm test set was not modified, augmented, or re-split

  5. PER-STAGE METRIC MONOTONICITY (SOFT CHECK):
     - If stage N+1 performs WORSE than stage N on val set by > 2%:
       WARNING — may indicate the stage is fitting noise
     - Does NOT halt (sometimes intermediate stages legitimately dip)
     - All warnings logged to: evaluation/leakage_audit.json
```

### 8.4 K-Fold Cross-Validation Rules

```
K-FOLD IS USED FOR:
  1. Out-of-fold (OOF) prediction generation (Stages 2-6, 9, 10)
  2. Robustness estimation (error bars on accuracy)

K-FOLD IS APPLIED TO:
  - TRAINING SET ONLY (never includes val or test)
  - StratifiedKFold with n_splits=5 (matching V1)
  - Random seed FIXED at 42 (reproducibility)

K-FOLD IS NOT USED FOR:
  - Selecting between fundamentally different architectures (use val set)
  - Generating pseudo-labels (use full training set)
  - Hyperparameter search across phases (use val set)

OOF PROCEDURE:
  For backbone B and K=5:
    For fold k in 1..5:
      train_idx_k = all training indices EXCEPT fold k
      holdout_idx_k = fold k indices
      Model_k = train B on train_idx_k images
      OOF_predictions[holdout_idx_k] = Model_k.predict(holdout_idx_k images)
    Final OOF: concatenate all holdout predictions
    → These OOF predictions are "honest" — each sample was predicted by a model
      that NEVER saw it during training
```

---

## 9. Enhanced Ensemble System (V2)

### 9.1 Existing 7-Stage Pipeline (PRESERVED)

```
Stage 1: Individual predictions    → 15 backbone predictions
Stage 2: Score ensembles           → Hard/Soft/Weighted/Logit voting
Stage 3: Stacking                  → LR/MLP/XGBoost stackers
Stage 4: Feature fusion            → Attention/Bilinear/Concat
Stage 5: Mixture of Experts        → Gating network selects top-K
Stage 6: Meta-learner              → Combines all previous stages (96.61%)
Stage 7: Knowledge distillation    → Compact student model (93.21%)
```

### 9.2 New Stages Added (V2)

```
Stage 8: Segmentation-Informed Ensemble
  ├── Weight each backbone's vote by its segmentation quality (IoU on val set)
  ├── Backbones that "see" the leaf clearly get higher voting weight
  ├── Backbones that focus on background get penalized
  ├── ISOLATION: IoU computed on val set only, applied to test set
  └── Expected: Eliminates noise from poorly-attending models

Stage 9: Cascaded Sequential Training
  ├── Train a sequence of classifiers where each one focuses on ERRORS of previous
  ├── Model_1: Best individual backbone (CustomCSPDarkNet, 96.04%)
  ├── Model_2: Trained ONLY on images Model_1 OOF got wrong (5-fold on train set)
  ├── Model_3: Trained ONLY on images Model_1+Model_2 both OOF-wrong
  ├── ... continue until convergence (max 5 cascade levels)
  ├── At inference: Sequential decision — if Model_1 confidence > threshold, use it
  │   else consult Model_2, etc.
  ├── ISOLATION: Hard examples identified via OOF only; thresholds tuned on val set
  └── Expected: Eliminates the "long tail" of hard examples

Stage 10: Adversarial Boosting Ensemble
  ├── AdaBoost-style reweighting of training samples
  ├── Each round trains with sample weights from previous round's OOF errors
  ├── Hard examples get exponentially higher weight
  ├── 5-10 boosting rounds, each with 5-fold OOF
  ├── Final prediction: Weighted combination based on per-sample difficulty
  ├── ISOLATION: All reweighting via OOF; final weights tuned on val set
  └── Expected: Forces ensemble to cover ALL failure modes

Stage 11: Cross-Architecture Knowledge Transfer
  ├── After all V2 backbones trained, extract "consensus features"
  ├── Where ALL 15 backbones agree on segmentation → high confidence regions
  ├── Where they disagree → ambiguous regions needing special attention
  ├── Train a final "referee" model on ambiguous training examples (OOF-identified)
  ├── ISOLATION: Ambiguity computed via OOF on train set; referee validated on val set
  └── Expected: Handles edge cases that individual models struggle with

Stage 12: Upgraded Distillation
  ├── Multi-teacher distillation (all 15 V2 backbones → new student)
  ├── Include segmentation knowledge in distillation loss
  ├── Attention transfer: Student mimics teacher attention maps
  ├── Progressive distillation: Student grows in capacity until convergence
  ├── ISOLATION: Teachers frozen; student trained on train set, validated on val set
  └── Expected: Compact model that retains segmentation awareness
```

### 9.3 Ensemble Architecture Diagram

```
                           V2 ENSEMBLE PIPELINE
                           ====================

     ┌─────────────────────────────────────────────────────────┐
     │              15 Dual-Head Backbones (V2)                │
     │  Each: Segmentation Head + Classification Head          │
     │  Initialized from V1 *_final.pth weights               │
     └────────────────────────┬────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      Seg Masks (15)   Class Probs (15)  Embeddings (15)
              │               │               │
              ▼               │               │
     ┌────────────────┐       │               │
     │ Stage 8:       │       │               │
     │ Seg-Informed   │───────┤               │
     │ Weighting      │       │               │
     └────────────────┘       │               │
                              ▼               ▼
     ┌────────────────────────────────────────────────────┐
     │  Stages 2–7 (PRESERVED from V1, re-run on V2)     │
     │  Score Ensembles → Stacking → Fusion → MoE        │
     │  → Meta-Learner → Distillation                    │
     └────────────────────────┬───────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────────┐ ┌──────────┐ ┌────────────────┐
     │ Stage 9:       │ │ Stage 10:│ │ Stage 11:      │
     │ Cascaded       │ │ Adversar.│ │ Cross-Arch     │
     │ Sequential     │ │ Boosting │ │ Referee         │
     └───────┬────────┘ └────┬─────┘ └───────┬────────┘
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                   ┌──────────────────┐
                   │  Stage 12:       │
                   │  V2 Student      │
                   │  (Multi-Teacher  │
                   │   Distillation)  │
                   └──────────────────┘
```

---

## 10. Segmentation-Based Validation Gate

### 10.1 Replacing the Heuristic Validator

```
OLD (Heuristic — fails on diseased sugarcane):
  Image → Color Match (0.35) → Vein Detection (0.30) → Elongation (0.25)
        → Edge Density (0.10) → Score → Threshold → Accept/Reject
  Problem: Red_rot scores 0.54, broccoli scores 0.62 — OVERLAP

NEW (Learned — understands spatial structure of ALL disease types):
  Image → V2 Backbone → 5-Channel Segmentation Mask → Region Analysis → Accept/Reject

  Decision Logic:
    1. plant_area = pixels in ANY of Ch1 (Healthy) + Ch2 (Structural) +
                    Ch3 (Surface Disease) + Ch4 (Tissue Degradation)
       → This captures ALL sugarcane tissue regardless of disease type
    2. total_area = H × W
    3. plant_ratio = plant_area / total_area
    4. IF plant_ratio < per_class_threshold → REJECT ("Insufficient sugarcane tissue")
       (per_class_threshold calibrated in Phase 0.5 from gold set; default 0.15)
    5. IF largest_connected_component(plant_area) < 500px² → REJECT ("No coherent plant")
    6. IF Ch2 + Ch3 + Ch4 > 0 → PASS (any disease signal = confirmed sugarcane context)
    7. IF Ch1 > 0.10 → PASS (healthy tissue alone = sugarcane present)
    8. ELSE → REJECT (nothing recognized as sugarcane)

  WHY THIS WORKS FOR ALL DISEASE TYPES:
    - Smut (whip, no leaf lesion):  Ch2 fires → step 6 PASS
    - Red_rot (internal rot):       Ch4 fires → step 6 PASS
    - Grassy_shoot (thin shoots):   Ch2 fires → step 6 PASS
    - Wilt (dying plant, low leaf): Ch4 fires → step 6 PASS
    - Brown_spot (leaf lesion):     Ch3 fires → step 6 PASS
    - Healthy sugarcane:            Ch1 fires → step 7 PASS
    - Broccoli / screenshot:        ALL channels 0 → step 8 REJECT
```

### 10.2 Why This Solves the Problem

| Image | Heuristic Score | Seg Mask Result | Outcome |
|-------|----------------|-----------------|---------|
| Red_rot (stalk cut showing rot) | 0.54 (FAIL) | Ch1=30% + Ch4=25% = 55% | ✅ PASS (degradation signal) |
| Smut (whip structure, no leaf) | 0.38 (FAIL) | Ch2=15% = 15% | ✅ PASS (structural anomaly) |
| Grassy_shoot (thin shoots) | 0.41 (FAIL) | Ch2=20% + Ch1=10% = 30% | ✅ PASS (structural anomaly) |
| Wilt (entire plant dying) | 0.00 (FAIL) | Ch4=35% + Ch1=10% = 45% | ✅ PASS (degradation signal) |
| Brown_spot (leaf lesion) | 0.67 (PASS) | Ch1=55% + Ch3=12% = 67% | ✅ PASS (surface disease) |
| Healthy sugarcane | 0.72 (PASS) | Ch1=70% = 70% | ✅ PASS (healthy tissue) |
| Broccoli | 0.62 (LEAK) | ALL channels = 0% | ❌ REJECT |
| Screenshot | 0.55 (LEAK) | ALL channels = 0% | ❌ REJECT |
| Random noise | 0.47 (PASS) | ALL channels = 0% | ❌ REJECT |

### 10.3 Recommended Serving Model for Validation Gate

For the production validation gate (inference server), use a **LIGHT-tier** segmentation model to minimize latency and VRAM footprint:

| Candidate | Params (backbone + decoder) | Est. Seg Infer VRAM | Latency (est.) | Recommendation |
|-----------|-----------------------------|--------------------|--------------------|----------------|
| CustomMobileOne | 4.7M + 2.2M = 6.9M | ~80 MB | <10 ms | ⭐ **Primary** (fastest) |
| CustomGhostNetV2 | 9.6M + 2.2M = 11.8M | ~180 MB | <15 ms | Backup (higher IoU expected) |
| CustomEfficientNetV4 | 1.9M + 2.2M = 4.1M | ~70 MB | <10 ms | Alternative if IoU sufficient |

**Selection rule**: After Phase 3, pick the LIGHT backbone with the highest val-set mean IoU (across all 5 channels) ≥ 0.65. If no LIGHT model reaches 0.65, fall back to the best MEDIUM model (ConvNeXt or ResNetMish).

**Swapping for full evaluation**: During batch evaluation or offline analysis, use the backbone with the overall highest IoU (regardless of tier) by setting `SEG_GATE_BACKBONE` in `V2_segmentation/config.py`. The server reloads the model on config change.

**Coexistence**: The gate seg model runs BEFORE the 15-backbone classification ensemble and is a separate model instance. It must be lightweight to avoid adding latency to every request.

---

## 11. Directory Structure

```
F:\DBT-Base-DIr\
├── Base_backbones.py                    # PRESERVED — original V1 pipeline
├── image_validator.py                   # PRESERVED — heuristic validator (V1)
├── checkpoints/                         # PRESERVED — all V1 checkpoints
├── ensembles/                           # PRESERVED — all V1 ensemble stages 1-7
├── plots_metrics/                       # PRESERVED — all V1 plots
├── ensemble_system/                     # PRESERVED — V1 ensemble pipeline code
│
├── V2_segmentation/                     # ★ NEW — entire V2 pipeline
│   ├── __init__.py
│   ├── config.py                        # V2 hyperparameters, paths, memory budgets
│   │
│   ├── analysis/                        # Pre-training audit
│   │   ├── __init__.py
│   │   ├── gradcam_generator.py         # GradCAM heatmaps from V1 models
│   │   ├── feature_analyzer.py          # Activation statistics per backbone
│   │   ├── error_pattern_analysis.py    # Cross-backbone error patterns
│   │   ├── class_attention_maps.py      # Per-class attention visualization
│   │   └── run_full_analysis.py         # Orchestrator for all analysis
│   │
│   ├── pseudo_labels/                   # Mask generation (no manual annotation)
│   │   ├── __init__.py
│   │   ├── grabcut_generator.py         # OpenCV GrabCut masks
│   │   ├── gradcam_mask_generator.py    # GradCAM-based masks from V1 models
│   │   ├── sam_generator.py             # SAM-based masks (optional)
│   │   ├── mask_combiner.py             # Weighted fusion of all methods
│   │   ├── mask_quality_scorer.py       # Per-image quality score + tier assignment
│   │   ├── class_sanity_checker.py      # Domain-specific validation rules per class
│   │   ├── spot_check_ui.py             # Human review tool (display + accept/reject)
│   │   └── iterative_refiner.py         # Post-Phase-A self-training refinement
│   │
│   ├── models/                          # V2 dual-head architectures
│   │   ├── __init__.py
│   │   ├── decoder.py                   # DeepLabV3+ decoder with ASPP
│   │   ├── dual_head.py                 # Wrapper: backbone + seg head + cls head
│   │   ├── attention_transfer.py        # Cross-layer attention mechanisms
│   │   └── model_factory.py             # Creates V2 model from V1 checkpoint
│   │
│   ├── losses/                          # Training objectives
│   │   ├── __init__.py
│   │   ├── dice_loss.py                 # Dice loss with uncertainty masking
│   │   ├── focal_loss.py                # Focal loss for hard examples
│   │   ├── joint_loss.py                # λ_seg × L_seg + λ_cls × L_cls
│   │   └── distillation_loss.py         # KD loss with attention transfer
│   │
│   ├── data/                            # Dataset handling
│   │   ├── __init__.py
│   │   ├── seg_dataset.py               # Image + mask + tier + confidence loader
│   │   ├── augmentations.py             # Joint image+mask augmentations
│   │   └── hard_example_sampler.py      # Weighted sampling for difficult images
│   │
│   ├── training/                        # Training pipeline
│   │   ├── __init__.py
│   │   ├── memory_manager.py            # OOM prevention, batch sizing, grad accum
│   │   ├── train_v2_backbone.py         # 3-phase training (A → B → C)
│   │   ├── train_all_backbones.py       # Orchestrator: waves 1-4 by memory tier
│   │   ├── metrics.py                   # IoU, Dice, mAP, classification metrics
│   │   └── checkpoint_manager.py        # V2 checkpoint save/load/resume
│   │
│   ├── ensemble_v2/                     # Enhanced ensemble pipeline
│   │   ├── __init__.py
│   │   ├── stage1_individual_v2.py      # Re-extract predictions from V2 models
│   │   ├── stage2_to_7_rerun.py         # Re-run existing stages on V2 predictions
│   │   ├── stage8_seg_informed.py       # NEW: Segmentation-weighted ensemble
│   │   ├── stage9_cascaded.py           # NEW: Sequential error-coverage training
│   │   ├── stage10_adversarial.py       # NEW: AdaBoost-style boosting
│   │   ├── stage11_referee.py           # NEW: Cross-architecture referee
│   │   ├── stage12_distillation_v2.py   # NEW: Multi-teacher + attention distill
│   │   └── ensemble_orchestrator.py     # Runs full 12-stage pipeline
│   │
│   ├── evaluation/                      # ★ NEW — Isolation enforcement
│   │   ├── __init__.py
│   │   ├── leakage_checker.py           # Train-test overlap, OOF verification
│   │   ├── overfit_detector.py          # Train-val gap monitoring per stage
│   │   ├── oof_generator.py             # Out-of-fold prediction generator (5-fold)
│   │   └── audit_reporter.py            # Generates leakage_audit.json per stage
│   │
│   ├── validation/                      # Segmentation-based validator
│   │   ├── __init__.py
│   │   ├── seg_validator.py             # Learned validation using seg masks
│   │   ├── region_analyzer.py           # Connected component analysis
│   │   └── calibrate_gate.py            # Phase 0.5: per-class threshold tuning
│   │
│   ├── visualization/                   # All plots and graphs
│   │   ├── __init__.py
│   │   ├── seg_overlay.py               # Segmentation mask overlaid on image
│   │   ├── heatmap_grid.py              # GradCAM comparison grids
│   │   ├── training_curves.py           # Loss/accuracy plots (V2)
│   │   ├── ensemble_comparison.py       # V1 vs V2 ensemble comparison
│   │   ├── validation_demo.py           # Before/after validation examples
│   │   └── tier_distribution.py         # Pseudo-label tier A/B/C visualizations
│   │
│   └── run_pipeline_v2.py              # ★ MAIN ENTRY POINT — runs everything
│
├── scripts/                             # Pre-flight smoke tests (CI-invokable)
│   ├── smoke_dual_head.py               # Dual-head fwd+bwd memory test (1 batch)
│   ├── smoke_oof_dryrun.py              # OOF pipeline dry run on 50 images
│   └── smoke_export_v2.py               # V2 export sanity check
│
├── gold_labels/                         # 200 manually annotated masks (Phase 0.5)
│   ├── Healthy/                         # ~15 images per class with binary masks
│   ├── Red_rot/
│   ├── Wilt/
│   └── ...                              # All 13 classes
│
├── segmentation_masks/                  # Generated pseudo-labels
│   ├── train/
│   │   ├── Black_stripe/               # Per-class: *_mask.png, *_conf.npy, *_tier.txt
│   │   ├── Brown_spot/
│   │   └── ...
│   ├── val/
│   ├── test/                           # Masks generated but NEVER used for training
│   └── audit/                          # Spot-check results, quality scores, logs
│
├── checkpoints_v2/                      # V2 model checkpoints
│   ├── CustomConvNeXt_v2_phaseA_best.pth
│   ├── CustomConvNeXt_v2_phaseA_refined.pth
│   ├── CustomConvNeXt_v2_phaseB_best.pth
│   ├── CustomConvNeXt_v2_phaseC_best.pth
│   ├── CustomConvNeXt_v2_final.pth
│   ├── ...
│   ├── oom_log.json                    # OOM events & recovery actions
│   └── ensembles_v2/
│       ├── stage8_seg_informed/
│       ├── stage9_cascaded/
│       ├── stage10_adversarial/
│       ├── stage11_referee/
│       └── stage12_student_v2/
│
├── plots_metrics_v2/                    # V2 visualization outputs
│   ├── analysis/                        # Pre-training audit plots
│   ├── segmentation/                    # Seg mask quality + tier distribution
│   ├── training/                        # V2 training curves
│   └── ensemble_v2/                     # V2 ensemble comparison plots
│
├── evaluation/                          # Leakage audit outputs
│   ├── leakage_audit.json              # Per-stage isolation verification
│   ├── oof_predictions/                # Stored OOF predictions per backbone per fold
│   └── overfit_reports/                # Train-val gap reports per stage
│
└── analysis_output/                     # Heatmap & feature analysis results
    ├── gradcam_heatmaps/
    ├── feature_stats/
    ├── error_patterns/
    └── class_attention_maps/
```

---

## 12. Segmentation Classes (5-Channel Output)

> **Design principle**: Not all sugarcane diseases are leaf lesions. Some are internal
> (Red_rot), some are structural abnormalities (Smut whips, Grassy_shoot proliferation),
> some are whole-tissue degradation (Wilt, Leaf_scorching), and some are growth-point
> deformations (Pokkah_boeng). The segmentation channels must capture ALL of these
> manifestation types — not just "leaf" and "spot".

| Channel | Class | Description | Color (Vis.) |
|---------|-------|-------------|----------|
| 0 | Background | Non-plant: soil, sky, debris, non-sugarcane objects | Black |
| 1 | Healthy Plant Tissue | Green leaf, intact stem, healthy growing point | Green |
| 2 | Structural Anomaly | Smut whip, grassy shoot proliferation, Pokkah_boeng malformation, abnormal growth | Yellow |
| 3 | Surface Disease Sign | Visible spots, stripes, mosaic pattern, ring marks, flecks, discoloration on tissue surface | Red |
| 4 | Tissue Degradation | Wilting, scorching, yellowing, rotting tissue (internal rot showing externally), drying | Orange |

**Why 5 channels?**
- Channels 2–4 capture the **three fundamentally different ways** sugarcane diseases manifest:
  - **Structural** (Ch 2): The plant grows something abnormal (whip, thin shoots, deformed leaves)
  - **Surface** (Ch 3): Visible marks/patterns appear ON the tissue (spots, stripes, mosaic)
  - **Degradation** (Ch 4): The tissue itself deteriorates (wilts, scorches, rots, yellows)
- This prevents the model from being leaf-lesion-biased — diseases like Smut and
  Grassy_shoot_disease have ZERO surface lesions, but their structural anomalies
  light up Ch 2 strongly.
- Channel 1 (Healthy Plant Tissue) includes BOTH leaf AND stem, because the gate
  question is "is this a sugarcane plant?" — not "is this a leaf?"

### 12.1 Disease ↔ Channel Manifestation Map

| Disease | Primary Channel | Secondary Channel | How It Manifests |
|---------|----------------|-------------------|------------------|
| Healthy | Ch 1 (Healthy) | — | Green, intact tissue throughout |
| Red_rot | Ch 4 (Degrad.) | Ch 3 (Surface) | Internal stalk rot; may show red discoloration when cut; external wilting |
| Brown_spot | Ch 3 (Surface) | — | Visible brown spots on leaf surface |
| Mosaic | Ch 3 (Surface) | — | Yellow-green mosaic pattern on leaf |
| Smut | Ch 2 (Structural) | — | Black whip-like structure from growing point; no leaf lesion |
| Grassy_shoot_disease | Ch 2 (Structural) | — | Thin, grass-like proliferation of shoots; growth pattern, not lesion |
| Wilt | Ch 4 (Degrad.) | — | Entire plant drying/wilting; no discrete lesion |
| Leaf_scorching | Ch 4 (Degrad.) | — | Leaf margins drying and browning |
| Yellow_leaf_Disease | Ch 4 (Degrad.) | Ch 3 (Surface) | Yellowing midrib spreading to blade |
| Pokkah_boeng | Ch 2 (Structural) | Ch 4 (Degrad.) | Growing point malformation + tissue rot |
| Ring_spot | Ch 3 (Surface) | — | Concentric ring patterns on leaf |
| Leaf_flecking | Ch 3 (Surface) | — | Small flecks/spots scattered on leaf |
| Black_stripe | Ch 3 (Surface) | — | Dark stripes along the leaf |

**Critical insight**: Channels 2, 3, 4 are NOT mutually exclusive on a single pixel.
A pixel can be both Structural Anomaly AND Tissue Degradation (e.g., Pokkah_boeng
showing deformed AND rotting tissue). The segmentation head outputs independent
sigmoid activations per channel (multi-label), not softmax (single-label).

---

## 13. Implementation Phases & Timeline

### Phase 0: Pre-Training Analysis [~6 files, ~1,200 lines]
```
Objective: Understand what V1 models learned before rebuilding
Duration: ~3 hours (GPU-bound for GradCAM generation)
Files: analysis/*.py

Steps:
  1. Generate GradCAM heatmaps for all 15 backbones × all 13 classes (10 imgs each)
     Memory: Load one backbone at a time → max ~1.1 GB for ViTHybrid inference
  2. Analyze feature activation patterns per backbone stage
  3. Identify cross-backbone error patterns
  4. Generate per-class attention region maps
  5. Produce comprehensive analysis report with plots

Output:
  - analysis_output/gradcam_heatmaps/ (15 × 13 = 195 heatmap images)
  - analysis_output/error_patterns/confusion_overlap.tiff
  - analysis_output/feature_stats/activation_distributions.tiff

GATE: Analysis informs pseudo-label generation strategy. Must complete before Phase 1.
```

### Phase 0.5: Validation Gate Threshold Calibration [~1 file, ~150 lines]
```
Objective: Tune per-class min_plant_ratio on a small gold-labeled set so the
  validation gate does not accidentally reject legitimate low-coverage classes
  (e.g., Smut with only a whip visible, Wilt with a dying stalk, Grassy_shoot
  with thin proliferating shoots — these have low total plant area in frame).
Duration: ~2 hours (manual labeling + calibration script)
Files: V2_segmentation/validation/calibrate_gate.py

Steps:
  1. SAMPLE 200 images from Data/ (~15 per class, stratified random):
     Run: python scripts/sample_gold_set.py --n_per_class 15 --seed 42
     Output: gold_labels/{class_name}/img_XXX.jpg (copies, not moves)
     Priority oversample: Wilt (20), Grassy_shoot_disease (20), Smut (20)
     — these are the low-plant-coverage classes most at risk of false rejection.

  2. GENERATE DRAFT MASKS (semi-automated, saves ~70% manual effort):
     a) Run GrabCut on each image → draft mask separating plant vs background
     b) Run SAM (if available) → refine boundaries
     c) Save drafts to: gold_labels/{class_name}/img_XXX_draft_mask.png
     Script: python scripts/generate_draft_gold_masks.py
     Draft masks are 5-channel (BG, Healthy, Structural, Surface, Degradation)
     — auto-assignment uses class label as hint (e.g., Smut images auto-assign
     plant pixels to Ch2-Structural). Human correction refines this.

  3. HUMAN CORRECTION (mandatory, this is the manual step):
     Open each draft mask in a labeling tool (LabelMe, CVAT, or the built-in
     spot_check_ui.py with edit mode). For each image:
       - Assign pixels to correct channels: Healthy (Ch1), Structural (Ch2),
         Surface Disease (Ch3), Tissue Degradation (Ch4), Background (Ch0)
       - For Smut: mark whip structure as Ch2, surrounding tissue as Ch1
       - For Red_rot: mark stalk rot as Ch4, healthy parts as Ch1
       - For Wilt: mark dried tissue as Ch4, remaining green as Ch1
       - Save corrected mask: gold_labels/{class_name}/img_XXX_mask.png
     Estimated time: ~3-4 min per image × 200 images = ~10-13 hours total.
     Can be split across sessions — progress tracked in gold_labels/progress.json.

  4. COMPUTE per-class plant_ratio distribution from corrected gold masks:
     plant_ratio = (Ch1 + Ch2 + Ch3 + Ch4) / total_pixels for each gold image.
     Set per-class min_plant_ratio = 5th percentile of gold set distribution
     (i.e., only 5% of real images of that class would fall below this threshold).

  5. STORE calibrated thresholds in V2_segmentation/config.py as:
       GATE_THRESHOLDS = {
         'Healthy': 0.35, 'Red_rot': 0.12, 'Wilt': 0.10,
         'Grassy_shoot_disease': 0.10, 'Smut': 0.08, ...  # tuned from gold set
       }
     Default for unlisted classes: 0.15 (current global value).

  6. VALIDATE: Run gate on gold set — must pass ≥98% of real sugarcane images.
     If any class fails: inspect the failing images, adjust threshold, re-validate.

Output:
  - gold_labels/{class_name}/img_XXX.jpg + img_XXX_mask.png (200 pairs)
  - gold_labels/progress.json (tracks which images are human-verified)
  - scripts/sample_gold_set.py, scripts/generate_draft_gold_masks.py
  - V2_segmentation/config.py updated with per-class thresholds
  - calibration_report.json (per-class 5th/25th/50th percentile plant ratios)

GATE: Per-class thresholds must be set before Phase 5 (server integration).
  Gold set creation can happen in parallel with Phases 1–3.
```

### Phase 1: Pseudo-Label Generation [~9 files, ~1,200 lines]
```
Objective: Create segmentation masks with quality tiers + uncertainty maps
Duration: ~4 hours (GPU-bound for GradCAM extraction from all 15 backbones)
Files: pseudo_labels/*.py

Steps:
  1. GrabCut: Green-channel seeded foreground extraction
  2. GradCAM: Ensemble GradCAM from all 15 V1 models → leaf region masks
     Memory: One backbone loaded at a time, inference only → max 1.1 GB
  3. SAM (optional): Zero-shot segmentation for high-quality boundaries
  4. Combine: Weighted pixel-wise fusion across methods
  5. Score: Per-image quality scoring → TIER_A / TIER_B / TIER_C assignment
  6. Class-specific sanity check (domain rules per disease)
  7. Human spot-check: 10 images × 13 classes × 3 tiers = 390 images
  8. Split masks into train/val/test matching existing dataset split

Output:
  - segmentation_masks/{train,val,test}/{class_name}/*.png + *.npy + *.txt
  - audit/spot_check_results.json, quality_scores.csv

GATE: Spot-check must pass thresholds (Section 5.2, Layer 4) before Phase 3.
```

### Phase 2: V2 Model Architecture [~7 files, ~1,200 lines]
```
Objective: Build dual-head architecture for all 15 backbones + memory manager
Duration: ~1 day (development, no GPU training)
Files: models/*.py, losses/*.py, training/memory_manager.py

Steps:
  1. Implement DeepLabV3+ decoder with ASPP module
  2. Create DualHeadModel wrapper (backbone + seg head + cls head)
  3. Implement skip connection extraction for each backbone architecture
  4. Build model factory: V1 checkpoint → V2 dual-head model
  5. Implement joint loss (Dice + BCE + CrossEntropy + Focal) with tier weighting
  6. Implement memory_manager.py (batch sizing, grad accum, OOM recovery)
  7. Verify forward pass for all 15 architectures at their tier batch sizes
  8. Create pre-flight smoke test scripts in scripts/:
     a) smoke_dual_head.py — For each backbone: create dual-head model, run 1
        forward + backward pass at tier batch size, assert peak memory < budget,
        assert gradients flow to all unfrozen params.
        CLI: python scripts/smoke_dual_head.py --backbone CustomConvNeXt
             python scripts/smoke_dual_head.py --all  (runs all 15, ~5 min)
     b) smoke_oof_dryrun.py — Run 1-fold OOF on 50 images with 1 LIGHT backbone,
        verify OOF array shape matches held-out set size, verify no train-test
        file overlap.
        CLI: python scripts/smoke_oof_dryrun.py --n_images 50
     These scripts run in CI on every PR touching V2_segmentation/ (< 2 min each).

Output:
  - V2 model classes that load V1 weights and add segmentation capability
  - Memory manager with per-backbone profiles from Section 3
  - scripts/smoke_dual_head.py, scripts/smoke_oof_dryrun.py (CI pre-flight checks)

NOTE: Phase 2 can be developed in parallel with Phase 0 and Phase 1.
```

### Phase 3: V2 Training Pipeline [~6 files, ~1,500 lines]
```
Objective: Train all 15 dual-head backbones (Phases A → B → C)
Duration: ~33 hours total (see Wave schedule in Section 7.2)
Files: training/*.py, data/*.py

Steps:
  1. Build segmentation dataset loader (image + mask + tier + confidence)
  2. Implement joint augmentations (same spatial transforms for image AND mask)
  3. Phase A training: Seg head only (backbone frozen), 30 epochs per backbone
     └── After Phase A: Run iterative refinement (update pseudo-labels)
     └── GATE: Refinement convergence check before Phase B
  4. Phase B training: Joint fine-tuning (both heads), 25 epochs per backbone
     └── Uses REFINED pseudo-labels
     └── BLOCKING ROLLBACK CHECK after Phase B (per backbone):
         IF val_acc_v2 < val_acc_v1 − 0.5%  → REVERT backbone to V1 weights
         IF mean_IoU < 0.50                  → REVERT backbone to V1 weights
         Reverted backbones still participate in ensemble using V1 *_final.pth
         Log rollback decisions to: checkpoints_v2/rollback_log.json
  5. Phase C training: Cls refinement (seg frozen), 15 epochs per backbone
     └── Only non-reverted backbones proceed to Phase C
  6. Save V2 checkpoints (5 per backbone) to checkpoints_v2/
  7. Generate training curves and segmentation quality plots

Memory management per wave:
  Wave 1 (LIGHT):   BS=32, grad_accum=1, no gradient checkpointing
  Wave 2 (MEDIUM):  BS=16, grad_accum=2, no gradient checkpointing
  Wave 3 (HIGH):    BS=8,  grad_accum=4, no gradient checkpointing
  Wave 4 (HEAVY):   BS=4,  grad_accum=8, gradient checkpointing ON

Output:
  - checkpoints_v2/{backbone}_v2_*.pth (5 × 15 = 75 checkpoint files)
  - plots_metrics_v2/training/{backbone}_v2_training_history.tiff (15 files)
  - plots_metrics_v2/segmentation/{backbone}_seg_samples.tiff (15 files)
  - checkpoints_v2/oom_log.json (if any OOM events occurred)
```

### Phase 4: Enhanced Ensemble (Stages 1–12) [~13 files, ~2,500 lines]
```
Objective: Rebuild and extend ensemble pipeline with V2 models + strict isolation
Duration: ~12 hours (OOF generation is compute-intensive)
Files: ensemble_v2/*.py, evaluation/*.py

⚠️ INFERENCE MEMORY RULE: During ALL ensemble stages, load ONE backbone at a
  time into GPU, run inference, save predictions to disk, then unload before
  loading the next. NEVER load multiple HEAVY backbones simultaneously — even
  two HEAVY models in inference mode exceed 2 GB combined. For production
  deployment, use the same sequential pattern or distribute across devices.
  Template: load → predict → save_to_disk → del model → torch.cuda.empty_cache()

Steps:
  1. Generate 5-fold OOF predictions for all 15 V2 backbones on training set
     Memory: One backbone at a time, inference only → all fit at BS=32
  2. Stage 1: Extract V2 predictions (15 backbones × train_OOF/val/test)
  3. Stages 2–7: Re-run existing ensemble methods on V2 OOF predictions
     └── Stackers trained on OOF predictions, validated on val set
  4. Stage 8: Implement segmentation-informed weighting (IoU from val set)
  5. Stage 9: Implement cascaded sequential training (OOF hard examples)
  6. Stage 10: Implement adversarial boosting (OOF error reweighting)
  7. Stage 11: Implement cross-architecture referee (OOF ambiguity)
  8. Stage 12: Multi-teacher distillation with attention transfer
  9. Per-stage: Run leakage checker + overfit detector + audit reporter
  10. Generate all comparison plots (V1 vs V2 at every stage)

Output:
  - checkpoints_v2/ensembles_v2/stage{8-12}/ (models + metrics)
  - evaluation/leakage_audit.json (per-stage isolation verification)
  - evaluation/oof_predictions/ (stored for reproducibility)
  - plots_metrics_v2/ensemble_v2/ (comparison charts)
  - Final V2 accuracy report

GATE: leakage_audit.json must show 0 violations before reporting results.
```

### Phase 5: Validation Gate + Server Integration [~4 files, ~600 lines]
```
Objective: Replace heuristic validator with segmentation-based validator
Duration: ~4 hours
Files: validation/*.py, inference_server updates

Steps:
  1. Implement seg_validator.py using best V2 segmentation model
     (pick backbone with highest val IoU — likely a LIGHT model for speed)
  2. Implement region_analyzer.py (connected components, area ratios)
  3. Update inference_server/engine/validation.py to use seg model
  4. Add POST /segment endpoint (returns mask + overlay)
  5. Re-test all 38 non-sugarcane images → must be 100% rejected
  6. Re-test all 130 real sugarcane images → must be >97% passed
  7. Specifically verify: Red_rot 100% pass, Wilt >90% pass

Output:
  - Updated server with segmentation-based validation
  - Test results logged to validation/gate_test_results.json
```

### Phase 6: Visualization & Reporting [~7 files, ~800 lines]
```
Objective: Comprehensive plots, comparisons, and labeled outputs
Duration: ~4 hours
Files: visualization/*.py

Steps:
  1. V1 vs V2 accuracy comparison (per backbone, per ensemble stage)
  2. Segmentation overlay visualizations (per class, per backbone)
  3. GradCAM comparison: V1 attention vs V2 attention
  4. Validation gate before/after demo images
  5. Pseudo-label tier distribution visualization
  6. Final pipeline architecture diagram
  7. All plots saved as labeled .tiff at 1200 DPI (matching V1 convention)

Output:
  - plots_metrics_v2/ with all visualization artifacts
  - Ready for publication / presentation
```

---

## 14. Hyperparameters

### V2 Training Configuration

| Parameter | Phase A (Seg Head) | Phase B (Joint) | Phase C (Cls Refine) |
|-----------|-------------------|-----------------|---------------------|
| Epochs | 30 | 25 | 15 |
| Backbone LR | 0 (frozen) | 1e-5 | 1e-6 |
| Seg Head LR | 1e-3 | 1e-4 | 0 (frozen) |
| Cls Head LR | 0 (frozen) | 1e-4 | 1e-4 |
| Effective Batch Size | 32 (all tiers) | 32 (all tiers) | 32 (all tiers) |
| Physical Batch Size | Tier-dependent (4–32) | Tier-dependent (4–32) | Tier-dependent (4–32) |
| Gradient Accumulation | Tier-dependent (1–8) | Tier-dependent (1–8) | Tier-dependent (1–8) |
| Image Size | 224 | 224 | 224 |
| Loss Weights | λ_seg=1.0, λ_cls=0.0 | λ_seg=0.4, λ_cls=0.6 | λ_seg=0.0, λ_cls=1.0 |
| Seg Loss Tier Weight | A=1.0, B=0.5, C=excluded | A=1.0, B=0.5, C=excluded | N/A |
| Patience | 5 | 5 | 3 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW | AdamW |
| Scheduler | CosineAnnealing + Warmup | CosineAnnealing + Warmup | CosineAnnealing |
| AMP | Enabled (all tiers) | Enabled (all tiers) | Enabled (all tiers) |
| Gradient Checkpointing | HEAVY tier only | HEAVY tier only | OFF (seg head frozen) |

### Memory Tier → Batch Size Mapping

| Memory Tier | Backbones | Physical BS | Grad Accum | Effective BS | Grad Ckpt |
|-------------|-----------|-------------|------------|-------------|-----------|
| 🟢 LIGHT | EfficientNetV4, DenseNetHybrid, InceptionV4, MobileOne, GhostNetV2 | 32 | 1 | 32 | OFF |
| 🟡 MEDIUM | ConvNeXt, ResNetMish, RegNet | 16 | 2 | 32 | OFF |
| 🟠 HIGH | CSPDarkNet, DynamicConvNet | 8 | 4 | 32 | OFF |
| 🔴 HEAVY | DeiTStyle, SwinTransformer, MaxViT, CoAtNet, ViTHybrid | 4 | 8 | 32 | ON |

### Segmentation-Specific

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Seg Classes | 5 | Background, Healthy Tissue, Structural Anomaly, Surface Disease, Tissue Degradation |
| ASPP Rates | [6, 12, 18] | Standard DeepLabV3+ dilation rates |
| Decoder Channels (🟢 LIGHT) | 256 | Full capacity, ample headroom |
| Decoder Channels (🟡 MEDIUM) | 256 | Full capacity, fits in budget |
| Decoder Channels (🟠 HIGH) | 192 | Reduced to save ~1 GB activation memory |
| Decoder Channels (🔴 HEAVY) | 128 | Minimal decoder + gradient ckpt in decoder |
| Skip Channels | 48 (LIGHT/MED), 32 (HIGH/HEAVY) | Scaled with decoder width |
| Dice Smooth | 1.0 | Numerical stability |
| Focal Alpha | 0.25 | Class imbalance handling |
| Focal Gamma | 2.0 | Hard example focus |

### Validation Gate

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Min Plant Ratio (global default) | 0.15 | Fallback for classes without gold-set calibration |
| Min Plant Ratio (per-class) | Tuned in Phase 0.5 | Calibrated from 200-image gold set (5th percentile) |
| NOTE | plant_ratio = Ch1+Ch2+Ch3+Ch4 | Counts ALL sugarcane tissue, not just green leaf |
| Min Component Size | 500px² | Reject scattered small green patches |
| Min Lesion Confidence | 0.3 | Disease regions count toward acceptance |
| Ensemble Seg Models | 3 | Top-3 seg models vote for mask |
| Gate Backbone | LIGHT tier (see §10.3) | Fastest model with mean IoU ≥ 0.65 |

### Pseudo-Label Quality

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Fusion weights (GrabCut:GradCAM:SAM) | 0.3 : 0.5 : 0.2 | GradCAM most reliable signal |
| Uncertainty threshold | 0.6 | Pixels below this are masked out in loss |
| TIER_A quality threshold | ≥ 0.80 | High confidence masks |
| TIER_B quality threshold | ≥ 0.50 | Medium confidence, reduced loss weight |
| TIER_C quality threshold | < 0.50 | Excluded from training |
| Max refinement rounds | 3 | Prevent infinite self-training loop |
| Refinement convergence threshold | 15% pixel change | Below this → converged |

---

## 15. Success Criteria

### Classification Performance

| Metric | V1 Baseline | V2 Target | Stretch Goal |
|--------|-------------|-----------|-------------|
| Best Individual Backbone | 96.04% | **97.5%+** | 98%+ |
| Meta-Learner Ensemble | 96.61% | **98.0%+** | 99%+ |
| Distilled Student | 93.21% | **96.0%+** | 97%+ |
| Weak Backbone Floor (CoAtNet/Swin/MaxViT) | 85–86% | **92%+** | 95%+ |

### Segmentation Quality

| Metric | Target |
|--------|--------|
| Mean IoU (across all 5 channels) | > 0.65 |
| Mean IoU (Healthy Tissue, Ch1) | > 0.75 |
| Mean IoU (Structural Anomaly, Ch2) | > 0.55 |
| Mean IoU (Surface Disease, Ch3) | > 0.60 |
| Mean IoU (Tissue Degradation, Ch4) | > 0.55 |
| Dice Score (all channels macro-avg) | > 0.70 |
| Per-class F1 (worst class) | > 0.50 |

### Validation Gate

| Metric | V1 Heuristic | V2 Segmentation Target |
|--------|-------------|----------------------|
| Non-sugarcane rejection | 78.9% (30/38) | **100%** (38/38) |
| Real sugarcane pass-through | ~80% (threshold-dependent) | **>97%** |
| Red_rot acceptance | 0% (all blocked) | **100%** |
| Wilt acceptance | 0–40% | **>90%** |

### Ensemble Completeness

| Metric | V1 | V2 Target |
|--------|-----|-----------|
| Ensemble Stages | 7 | **12** |
| Error Coverage (% images with ≥1 correct model) | ~99% | **>99.9%** |
| Hard Example Resolution | Unknown | Tracked per-image |

### Evaluation Integrity

| Metric | Requirement |
|--------|------------|
| Train-test data overlap | **0 images** (verified by hash) |
| Per-stage leakage violations | **0** (verified by audit) |
| OOF verification (stacking stages) | **All stage 2-6, 9, 10 use OOF features** |
| Test set SHA-256 unchanged | **Matches V1 hash exactly** |

---

## 16. Estimated Line Counts

| Component | Files | Lines | Complexity |
|-----------|-------|-------|-----------|
| Phase 0: Analysis | 6 | ~1,200 | Medium |
| Phase 0.5: Gate Calibration | 1 | ~150 | Low |
| Phase 1: Pseudo-labels (expanded) | 9 | ~1,200 | High |
| Phase 2: V2 Models + Memory Mgr + Smoke Tests | 10 | ~1,400 | High |
| Phase 3: Training | 6 | ~1,500 | High |
| Phase 4: Ensemble V2 + Evaluation | 13 | ~2,500 | High |
| Phase 5: Validation + Server | 5 | ~750 | Medium |
| Phase 6: Visualization | 7 | ~800 | Low |
| **TOTAL** | **~57 files** | **~9,500 lines** | |

---

## 17. Risk Mitigation (Revised)

| Risk | Probability | Impact | Mitigation (Concrete) |
|------|------------|--------|----------------------|
| Pseudo-labels too noisy | **HIGH** | High | 6-layer noise protocol (Section 5.2): multi-source fusion → quality scoring → tier assignment → class-specific rules → human spot-check (BLOCKING) → iterative refinement. TIER_C excluded entirely. Uncertain pixels masked in loss. |
| GPU OOM during dual-head training | **HIGH** for HEAVY tier | High | Profiled every backbone (Section 3). Tier-based batch sizing: HEAVY=BS4+grad_accum8+gradient_checkpointing. OOM recovery: auto-halve BS, resume from checkpoint. BS=1+grad_ckpt is fallback. Per-epoch memory monitoring with 2GB headroom. |
| Evaluation leakage across 12 stages | **MEDIUM** | Critical | Full isolation protocol (Section 8): OOF predictions for all stacking/boosting stages. Automated leakage checker + overfit detector after every stage. Test set SHA-256 verified. leakage_audit.json must show 0 violations (BLOCKING). |
| Segmentation hurts classification accuracy | Low | High | BLOCKING rollback after Phase B: if V2 val_acc < V1 val_acc − 0.5% OR mean IoU < 0.50, revert that backbone to V1 weights for ensemble. Reverted backbones skip Phase C. Logged in rollback_log.json. |
| 12-stage ensemble overfitting | Medium | Medium | OOF used for all stacking inputs. Train-val gap monitored per stage (halt if >10%). Cross-validation within training set only. |
| SAM not available / too large | Low | Low | GrabCut + GradCAM sufficient; SAM is optional. Fusion weights adjust automatically. |
| Training time too long (>33h) | Medium | Low | Wave scheduling: LIGHT first (3.75h) → validation → MEDIUM → HIGH → HEAVY. Can checkpoint and resume across multiple sessions. |
| Pseudo-label refinement diverges | Low | Medium | Max 3 refinement rounds. Convergence check: <15% pixel change = stop. Divergence guard: if round N worse than N-1, use N-1 labels. |

---

## 18. Execution Order & Dependencies

```
Phase 0 ──► Phase 1 ──┬──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
Analysis    Masks      │    Training    Ensemble    Server      Plots
                       │       │           │           ▲
Phase 2 ───────────────┘       │           │           │
Models (parallel)              │           ▼      Phase 0.5
                               │    Leakage Audit  Gate Calibration
Phase 0.5 ─────────────────────│───────────────────►(must complete
Gate Calibration (parallel)    │                    before Phase 5)
                               │
                               ▼
                          V2 Checkpoints ──► Pseudo-Label Refinement ──► Phase B/C
                          (Phase A)          (between A and B)
```

**BLOCKING GATES:**
1. Phase 1 → Phase 3: Spot-check MUST pass thresholds (Section 5.2, Layer 4)
2. Phase 3 (between A and B): Pseudo-label refinement convergence + oscillation gate (§5.2 Layer 6)
3. Phase 3 (after B, per backbone): Rollback check — val_acc drop ≥0.5% OR mean IoU <0.50 → revert to V1
4. Phase 4 (per stage): leakage_audit.json must show 0 violations
5. Phase 0.5 → Phase 5: Per-class gate thresholds calibrated on gold set
6. Phase 5: 100% non-sugarcane rejection + >97% real sugarcane pass-through

**PARALLEL TRACKS:**
- Phase 0 (Analysis) and Phase 2 (Model architecture) can be developed in parallel
- Phase 6 (Visualization) code can be written alongside Phase 3/4 (run after data ready)

---

## 19. Key Principles

1. **NOTHING IS DELETED** — V1 code, checkpoints, plots all preserved untouched
2. **V1 WEIGHTS ARE V2 INITIALIZATION** — every backbone starts from its trained state
3. **SEGMENTATION TEACHES WHERE, CLASSIFICATION TEACHES WHAT** — dual-head synergy
4. **HEATMAPS BEFORE TRAINING** — understand V1 before building V2
5. **ENSEMBLE COVERS ALL GAPS** — sequential/adversarial/referee stages eliminate failure modes
6. **ALL OUTPUTS LABELED AND PLOTTED** — every checkpoint, every plot, every metric tracked
7. **NO STAGE SKIPPED, AVOIDED, STALLED OR BROKEN** — every phase runs to completion with gating checks
8. **EVERY MODEL FITS IN 24GB** — profiled and budgeted per-backbone; OOM recovery is automatic
9. **PSEUDO-LABEL NOISE IS CONTAINED** — 6-layer defence with blocking human audit gate
10. **EVALUATION IS LEAK-PROOF** — OOF for stacking, per-stage audits, test set never touched
11. **EVERY RUN IS REPRODUCIBLE** — deterministic seeds, git hash + pip freeze in every checkpoint, DataLoader worker seeding
12. **EVERY CLASS IS TREATED EQUALLY** — no class is "harder" or "less important"; class imbalance in sample count does NOT reduce a class's training weight or evaluation priority. All 13 diseases are first-class citizens.
13. **DISEASES ARE NOT ALL LEAF LESIONS** — Smut is a whip, Grassy_shoot is a growth pattern, Red_rot is internal, Wilt is whole-plant. The 5-channel segmentation captures structural anomalies, surface signs, AND tissue degradation — not just "leaf + spot".

---

*Sprint 3-SEG Plan v2.1 — Added per-backbone decoder sizing, formal rollback gates, validation gate calibration (Phase 0.5), inference memory safety, serving model recommendation, smoke test scripts, deterministic seeds, and pseudo-label drift tracking. Awaiting approval to proceed with Phase 0 (Analysis).*
