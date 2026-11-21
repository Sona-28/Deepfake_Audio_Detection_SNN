# Generated from user-provided Rate_Coding_AISurrey(2).ipynb

import os
from pathlib import Path
import torch

# ==================================
# DEVICE
# ==================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU unavailable. Run inside a Slurm GPU allocation."
    )

device = torch.device("cuda")

print("Using:", device)
print("GPU:", torch.cuda.get_device_name(0))


# ==================================
# PROJECT DIRECTORIES
# ==================================

PROJECT_ROOT = Path(
    "/mnt/fast/nobackup/users/ss06262/snn-deepfake"
)

# Dataset downloaded using data.py
DATA_ROOT = PROJECT_ROOT / "data" / "asvspoof2019" / "LA" / "LA"

# Best trained models
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Experiment results
RESULTS_DIR = PROJECT_ROOT / "results"

# Slurm logs
LOG_DIR = PROJECT_ROOT / "logs"


# ==================================
# CREATE OUTPUT DIRECTORIES
# ==================================

for directory in [
    CHECKPOINT_DIR,
    RESULTS_DIR,
    LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ==================================
# VERIFY PATHS
# ==================================

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_ROOT}"
    )

print("\nProject root:", PROJECT_ROOT)
print("Dataset root:", DATA_ROOT)
print("Checkpoints:", CHECKPOINT_DIR)
print("Results:", RESULTS_DIR)
print("Logs:", LOG_DIR)


import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

class ASVspoofDataset(Dataset):
    def __init__(
            self, 
            protocol_path, 
            audio_file, 
            sr=16000, 
            fixed_length=64000,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        ):
        self.audio_file = audio_file
        self.sr = sr
        self.fixed_length = fixed_length

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, 
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB()

        self.data = []

        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()

                utt_id = parts[1]
                label = 1 if parts[-1] == 'bonafide' else 0
                self.data.append((utt_id, label))

    def __len__(self):
        return len(self.data)

    def pad_or_trim(self, waveform):
        waveform = waveform.squeeze(0)  # Remove channel dimension if present
        if waveform.shape[-1] < self.fixed_length:
           pad_len = self.fixed_length - waveform.shape[-1]
           waveform = F.pad(waveform, (0, pad_len))
        else:
           waveform = waveform[:self.fixed_length]  # Trim to fixed length

        return waveform

    def __getitem__(self, idx):
        utt_id, label = self.data[idx]
        audio_path = os.path.join(self.audio_file, f"{utt_id}.flac")

        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(waveform)

        waveform = self.pad_or_trim(waveform)

        mel = self.mel(waveform)
        logmel = self.to_db(mel)

        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)  # Normalize to zero mean and unit variance
        logmel = torch.sigmoid(logmel)  # Apply sigmoid activation to ensure values are in [0, 1]
        logmel = logmel.unsqueeze(0)  # Add channel dimension

        return logmel, torch.tensor(label, dtype=torch.long)


import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
from pathlib import Path


class ConvSNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.lif1 = snn.Leaky(
            beta=0.90, spike_grad=spike_grad
        )

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.lif2 = snn.Leaky(
            beta=0.90, spike_grad=spike_grad
        )

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.lif3 = snn.Leaky(
            beta=0.90, spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(128, num_classes)

    
    def forward(self, x, collect_traces=False):

        # Record only during evaluation
        record = collect_traces and not self.training

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        outputs = []

        traces = None
        if record:
            traces = {
                "lif1_spikes": [],
                "lif1_membrane": [],
                "lif2_spikes": [],
                "lif2_membrane": [],
                "lif3_spikes": [],
                "lif3_membrane": []
            }

        def save_activity(layer, spikes, membrane):
            if not record:
                return

            traces[f"{layer}_spikes"].append(
                spikes.detach().to(torch.uint8).cpu()
            )
            traces[f"{layer}_membrane"].append(
                membrane.detach().to(torch.float16).cpu()
            )

        for t in range(x.size(0)):
            xt = x[t]

            # LIF 1
            cur1 = self.conv1(xt)
            spk1, mem1 = self.lif1(cur1, mem1)

            save_activity("lif1", spk1, mem1)
            spk1 = self.pool1(spk1)

            # LIF 2
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            save_activity("lif2", spk2, mem2)
            spk2 = self.pool2(spk2)

            # Reduce spatial dimensions
            spk2 = self.adaptive_pool(spk2)

            # LIF 3
            flat = spk2.flatten(1)
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            save_activity("lif3", spk3, mem3)

            out = self.fc2(spk3)
            outputs.append(out)

        # [timesteps, batch, num_classes]
        outputs = torch.stack(outputs)

        if record:
            traces = {
                key: torch.stack(value)
                for key, value in traces.items()
            }

        return (
            outputs,
            traces
        )



from sklearn.metrics import roc_curve

def compute_eer(scores, labels):

    fpr, tpr, thresholds = roc_curve(labels, scores)

    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx]

    return eer, threshold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

train_dataset = ASVspoofDataset(
    protocol_path=DATA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
    audio_file=DATA_ROOT / "ASVspoof2019_LA_train" / "flac",
)

val_dataset = ASVspoofDataset(
    protocol_path=DATA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
    audio_file=DATA_ROOT / "ASVspoof2019_LA_dev" / "flac",
)

test_dataset = ASVspoofDataset(
    protocol_path=DATA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
    audio_file=DATA_ROOT / "ASVspoof2019_LA_eval" / "flac",
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Ensure deterministic behavior for all operations


import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def sync_device(device):
    """Wait for queued CUDA operations to finish."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_model(
    model,
    train_loader,
    val_loader,
    encoder,
    criterion,
    optimizer,
    device,
    seed,
    method,
    num_epochs=20
):

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_eer = float("inf")
    best_epoch = None

    best_path = (
        CHECKPOINT_DIR /
        f"best_model_{seed}_{method}.pth"
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_eer": [],
        "train_times": [],
        "val_times": [],
        "checkpoint_times": [],
        "epoch_times": []
    }

    sync_device(device)
    total_start = time.perf_counter()

    for epoch in range(num_epochs):

        sync_device(device)
        epoch_start = time.perf_counter()

        # ==================================
        # TRAINING
        # ==================================

        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        sync_device(device)
        train_start = time.perf_counter()

        for waveform, label in train_loader:

            waveform = waveform.to(
                device, non_blocking=True
            )
            label = label.long().to(
                device, non_blocking=True
            )

            spikes = encoder.encode(waveform)

            optimizer.zero_grad(set_to_none=True)

            # No traces during training
            outputs, _ = model(
                spikes,
                collect_traces=False
            )

            logits = outputs.mean(dim=0)

            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            batch_size = label.size(0)

            train_loss_sum += (
                loss.detach().item() * batch_size
            )
            train_samples += batch_size

        sync_device(device)
        train_time = time.perf_counter() - train_start

        train_loss = train_loss_sum / train_samples

        # ==================================
        # VALIDATION
        # ==================================

        model.eval()

        val_loss_sum = 0.0
        val_samples = 0

        val_scores = []
        val_labels = []

        sync_device(device)
        val_start = time.perf_counter()

        with torch.inference_mode():

            for waveform, label in val_loader:

                waveform = waveform.to(
                    device, non_blocking=True
                )
                label = label.long().to(
                    device, non_blocking=True
                )

                spikes = encoder.encode(waveform)

                # No traces during validation
                outputs, _ = model(
                    spikes,
                    collect_traces=False
                )

                logits = outputs.mean(dim=0)

                loss = criterion(logits, label)

                batch_size = label.size(0)

                val_loss_sum += (
                    loss.item() * batch_size
                )
                val_samples += batch_size

                probs = F.softmax(
                    logits, dim=1
                )[:, 1]

                val_scores.append(
                    probs.cpu()
                )
                val_labels.append(
                    label.cpu()
                )

        sync_device(device)
        val_time = time.perf_counter() - val_start

        val_loss = val_loss_sum / val_samples

        val_scores = torch.cat(
            val_scores
        ).numpy()

        val_labels = torch.cat(
            val_labels
        ).numpy()

        # ==================================
        # EER CALCULATION
        # ==================================

        val_eer, _ = compute_eer(
            val_scores,
            val_labels
        )

        # ==================================
        # CHECKPOINT
        # ==================================

        checkpoint_time = 0.0

        if np.isfinite(val_eer) and val_eer < best_val_eer:

            best_val_eer = val_eer
            best_epoch = epoch + 1

            sync_device(device)
            checkpoint_start = time.perf_counter()

            torch.save(
                model.state_dict(),
                best_path
            )

            checkpoint_time = (
                time.perf_counter()
                - checkpoint_start
            )

        # ==================================
        # TOTAL EPOCH TIME
        # ==================================

        sync_device(device)
        epoch_time = (
            time.perf_counter() - epoch_start
        )

        # ==================================
        # HISTORY
        # ==================================

        history["train_loss"].append(
            float(train_loss)
        )
        history["val_loss"].append(
            float(val_loss)
        )
        history["val_eer"].append(
            float(val_eer)
        )

        history["train_times"].append(
            train_time
        )
        history["val_times"].append(
            val_time
        )
        history["checkpoint_times"].append(
            checkpoint_time
        )
        history["epoch_times"].append(
            epoch_time
        )

        print(
            f"\nEpoch {epoch + 1:02d}/{num_epochs}"
        )
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val EER: {val_eer:.4f}"
        )
        print(
            f"Train: {train_time:.2f}s | "
            f"Validation: {val_time:.2f}s | "
            f"Checkpoint: {checkpoint_time:.2f}s"
        )
        print(
            f"Total Epoch: {epoch_time:.2f}s"
        )
        print(
            f"Best EER: {best_val_eer:.4f} "
            f"(Epoch {best_epoch})",
            flush=True
        )

    # ==================================
    # OVERALL TRAINING TIME
    # ==================================

    sync_device(device)
    total_training_time = (
        time.perf_counter() - total_start
    )

    training_summary = {
        "seed": seed,
        "method": method,
        "device": str(device),
        "num_epochs": num_epochs,

        "total_training_time": total_training_time,

        "total_epoch_time": sum(
            history["epoch_times"]
        ),

        "average_epoch_time": float(
            np.mean(history["epoch_times"])
        ),

        "total_train_time": sum(
            history["train_times"]
        ),

        "total_validation_time": sum(
            history["val_times"]
        ),

        "total_checkpoint_time": sum(
            history["checkpoint_times"]
        ),

        "best_epoch": best_epoch,
        "best_val_eer": float(best_val_eer),

        "epoch_times": history["epoch_times"],
        "train_times": history["train_times"],
        "val_times": history["val_times"],
        "checkpoint_times": history["checkpoint_times"]
    }

    # Save experiment history and timing
    summary_path = (
        RESULTS_DIR /
        f"training_{seed}_{method}.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            {
                "summary": training_summary,
                "history": history
            },
            f,
            indent=2
        )

    print("\n========== TRAINING COMPLETE ==========")
    print(
        f"Total Time: {total_training_time:.2f}s"
    )
    print(
        f"Average Epoch: "
        f"{training_summary['average_epoch_time']:.2f}s"
    )
    print(
        f"Best Validation EER: {best_val_eer:.4f}"
    )
    print(
        f"Best Epoch: {best_epoch}"
    )
    print(
        f"Checkpoint: {best_path}"
    )
    print(
        f"Results: {summary_path}",
        flush=True
    )

    return (
        best_path,
        best_val_eer,
        training_summary,
        history
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch
from torch.utils.data import DataLoader, Subset


def evaluate(model, test_loader, encoder, device, seed, method):
    """Store all test outputs, and full neuron states for 20 fixed samples only."""
    model.eval()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_dir = RESULTS_DIR / "evaluation" / f"{method}_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_outputs, all_preds, all_labels = [], [], []
    with torch.inference_mode():
        for waveform, label in test_loader:
            waveform = waveform.to(device)
            spikes = encoder.encode(waveform)
            outputs, _ = model(spikes, collect_traces=False)
            preds = outputs.mean(dim=0).argmax(dim=1)
            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    all_outputs = torch.cat(all_outputs, dim=1)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    torch.save({
        "outputs": all_outputs,
        "predictions": all_preds,
        "labels": all_labels,
        "seed": seed,
        "method": method,
    }, RESULTS_DIR / f"outputs_{method}_seed{seed}.pt")

    # Choose identical samples for all seeds and encoding methods.
    # Selection is independent of model predictions.
    dataset = test_loader.dataset
    generator = torch.Generator().manual_seed(2026)
    selected_indices = []
    for target_label in (0, 1):
        candidates = [i for i in range(len(dataset))
                      if int(dataset.data[i][1]) == target_label]
        selected_indices.extend(
            candidates[j] for j in torch.randperm(
                len(candidates), generator=generator
            )[:min(10, len(candidates))].tolist()
        )
    selected_indices.sort()

    # Batch size 1 avoids keeping many large traces in CPU memory.
    trace_loader = DataLoader(Subset(dataset, selected_indices),
                              batch_size=1, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for k, (waveform, label) in enumerate(trace_loader):
            waveform = waveform.to(device)
            spikes = encoder.encode(waveform)
            outputs, traces = model(spikes, collect_traces=True)
            original_index = selected_indices[k]
            sample_id = dataset.data[original_index][0]
            torch.save({
                "sample_index": original_index,
                "sample_id": sample_id,
                "label": label.cpu(),
                "outputs": outputs.cpu(),
                "traces": traces,
                "seed": seed,
                "method": method,
            }, save_dir / f"sample_{original_index:06d}.pt")

    print(f"Saved all {len(all_labels)} outputs, and "
          f"{len(selected_indices)} full sample traces for seed {seed}", flush=True)
    return all_outputs, all_preds, all_labels



import os
import torch


def run_seed(
    seed,
    model_fn,
    train_loader,
    val_loader,
    test_loader,
    encoder,
    criterion,
    device,
    method
):

    # ==================================
    # 1. REPRODUCIBILITY
    # ==================================

    set_seed(seed)

    model = model_fn().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # ==================================
    # 2. TRAIN MODEL
    # ==================================

    best_path, best_val_eer, training_summary, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        seed=seed,
        method=method,
        num_epochs=20
    )

    # ==================================
    # 3. LOAD BEST CHECKPOINT
    # ==================================

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device,
            weights_only=True
        )
    )

    # ==================================
    # 4. MODEL INFORMATION
    # ==================================

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # Actual saved checkpoint size
    model_size_mb = (
        os.path.getsize(best_path) / (1024 ** 2)
    )

    # ==================================
    # 5. TEST EVALUATION
    # ==================================

    all_outputs, all_preds, all_labels = evaluate(
        model=model,
        test_loader=test_loader,
        encoder=encoder,
        device=device,
        seed=seed,
        method=method
    )

    # ==================================
    # 6. EXPERIMENT SUMMARY
    # ==================================

    experiment_summary = {
        "seed": seed,
        "method": method,
        "num_params": num_params,
        "model_size_mb": model_size_mb,
        "best_val_eer": best_val_eer,
        "best_checkpoint": str(best_path),
        "num_test_samples": len(all_labels),
        "output_shape": list(all_outputs.shape)
    }

    print("\n========== SEED COMPLETE ==========")
    print(f"Seed: {seed}")
    print(f"Method: {method}")
    print(f"Parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.2f} MiB")
    print(f"Best validation EER: {best_val_eer:.4f}")
    print(f"Test samples: {len(all_labels)}")
    print(f"Checkpoint: {best_path}", flush=True)

    return experiment_summary, training_summary, history


import snntorch.spikegen as spikegen

class SpikeEncoder:
    def __init__(self, num_steps=30):
        self.num_steps = num_steps
    def encode(self, x):
        return spikegen.rate(x, num_steps=self.num_steps)



import json

encoder = SpikeEncoder(num_steps=30)
criterion = nn.CrossEntropyLoss()

seeds = [42, 123, 999, 2026, 7]

final_rate_results = []
final_rate_training_summaries = []
final_rate_histories = []

# Directory for individual seed summaries
SEED_RESULTS_DIR = RESULTS_DIR / "rate_encoding"
SEED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for seed in seeds:

    print(f"\n===== Seed {seed} =====", flush=True)

    experiment_summary, training_summary, history = run_seed(
        seed=seed,
        model_fn=lambda: ConvSNN(num_classes=2),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        encoder=encoder,
        criterion=criterion,
        device=device,
        method="rate"
    )

    final_rate_results.append(experiment_summary)
    final_rate_training_summaries.append(training_summary)
    final_rate_histories.append(history)

    # Save this seed immediately
    seed_data = {
        "experiment_summary": experiment_summary,
        "training_summary": training_summary,
        "history": history
    }

    seed_path = SEED_RESULTS_DIR / f"seed_{seed}_summary.json"

    with open(seed_path, "w") as f:
        json.dump(seed_data, f, indent=2)

    print(f"Saved seed {seed}: {seed_path}", flush=True)

print("\nAll five seeds completed.", flush=True)



