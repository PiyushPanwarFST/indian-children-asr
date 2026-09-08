#!/bin/bash
# ============================================================
# Cache IndicConformer model on login node (has internet)
# Run this ONCE before submitting the PBS job.
# ============================================================
# Usage: bash hpc/cache_indicconformer.sh
# ============================================================

cd /iitjfs/home/2025rcs1026/Indain_children_spech

module load codes/anaconda3-2023.09
source activate children_asr

echo "Caching IndicConformer 600M model..."
echo "This downloads ~2.5GB to HuggingFace cache."
echo ""

python -c "
from transformers import AutoModel
print('Downloading ai4bharat/indic-conformer-600m-multilingual...')
model = AutoModel.from_pretrained(
    'ai4bharat/indic-conformer-600m-multilingual',
    trust_remote_code=True,
)
print(f'Success! Languages: {list(model.vocab.keys())}')
print(f'Hindi vocab: {len(model.vocab[\"hi\"])} tokens')
print(f'Marathi vocab: {len(model.vocab[\"mr\"])} tokens')
print('Model cached. GPU job can now run offline.')
"

echo ""
echo "Done. Now submit: qsub hpc/run_step0_precompute.pbs"
