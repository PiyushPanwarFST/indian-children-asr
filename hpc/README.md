# HPC Setup — IIT Jammu Agastya

## Connection
```bash
ssh -i Aamir.pem 2025rcs1026@agastya.iitjammu.ac.in
```

## Step 1: Transfer project (from local machine)
```bash
# From your local system — transfer everything except large files
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude 'checkpoints/' \
    --exclude '.git/' \
    --exclude 'ASER-Dataset/Data/' \
    --exclude 'ASER-Dataset/extracted/' \
    --exclude 'ASER-Dataset/.git/' \
    --exclude '*.pem' \
    --exclude '__pycache__/' \
    ~/Indain_children_spech/ \
    2025rcs1026@agastya.iitjammu.ac.in:~/Indain_children_spech/

# Transfer audio_processed separately (large, ~20GB)
rsync -avz --progress \
    ~/Indain_children_spech/ASER-Dataset/audio_processed/ \
    2025rcs1026@agastya.iitjammu.ac.in:~/Indain_children_spech/ASER-Dataset/audio_processed/

# Transfer split CSVs
rsync -avz --progress \
    ~/Indain_children_spech/ASER-Dataset/splits/ \
    2025rcs1026@agastya.iitjammu.ac.in:~/Indain_children_spech/ASER-Dataset/splits/
```

## Step 2: Setup environment (on HPC login node)
```bash
bash hpc/setup_env.sh
```

## Step 3: Submit jobs
```bash
# Semantic branch (Step 0 + Step 1) — ~24 hours
qsub hpc/run_semantic.pbs

# Acoustic branch (Step 2) — ~12 hours
qsub hpc/run_acoustic.pbs
```

## Step 4: Monitor
```bash
# Check job status
qstat -u 2025rcs1026

# Check output (while running)
tail -f hpc/semantic_job.out
tail -f hpc/acoustic_job.out
```

## Step 5: Download results (from local machine)
```bash
# Download checkpoints
rsync -avz --progress \
    2025rcs1026@agastya.iitjammu.ac.in:~/Indain_children_spech/checkpoints/ \
    ~/Indain_children_spech/checkpoints/

# Download logs
rsync -avz --progress \
    2025rcs1026@agastya.iitjammu.ac.in:~/Indain_children_spech/hpc/*.out \
    ~/Indain_children_spech/hpc/
```

## GPU Info
- Tesla V100-PCIE-32GB (4x local RTX 4060)
- Queue: gpu-ext (PBS scheduler)
