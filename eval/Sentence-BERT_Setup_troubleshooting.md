# Sentence-BERT Similarity Setup Notes

## Goal

To compute Sentence-BERT similarity in the evaluation notebook, the environment must:

- import `sentence_transformers`
- import compatible `transformers`
- load a Sentence-BERT model such as `paraphrase-multilingual-MiniLM-L12-v2`

In this notebook, Sentence-BERT similarity is computed via `sentence-transformers`, not from raw OpenRouter responses.

## Active environment

The active Jupyter kernel was:

```text
C:\Users\ritaMZ\miniconda3\envs\aqa-data\python.exe
```

Quick check:

```python
import sys
print(sys.executable)
```

## Required libraries

Minimum required stack:

- `sentence-transformers`
- `transformers`
- `torch`
- `numpy`
- `scipy`
- `scikit-learn`

Import test:

```python
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer

print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Sentence-BERT import OK")
```

## Recommended installation in notebook

Install packages in the current Jupyter kernel:

```python
%pip install "transformers==4.52.4" "sentence-transformers==5.1.2"
```

If conflicts exist:

```python
%pip uninstall -y transformers sentence-transformers mistral-common
%pip install "transformers==4.52.4" "sentence-transformers==5.1.2"
```

After installation:

1. Restart kernel.
2. Re-run import test.

## Common failure modes

### 1) `name 'nn' is not defined`

```text
Sentence-BERT import failed: name 'nn' is not defined
```

Likely cause: broken/incompatible `transformers` install.

### 2) `mistral-common` backend error

```text
Backend should be defined in the BACKENDS_MAPPING. Offending backend: mistral-common
```

Likely cause: internal conflict in HF package stack.

### 3) `peft` / `accelerate` mismatch

```text
ImportError: cannot import name 'clear_device_cache' from 'accelerate.utils.memory'
```

Likely cause: version mismatch between `peft` and `accelerate`.

### 4) `llava` dependency conflict

`llava` often requires older versions:

- `transformers==4.37.2`
- `tokenizers==0.15.1`

This conflicts with newer versions needed by current Sentence-BERT setup.

### 5) Kernel not restarted after reinstall

```text
ImportError: cannot load module more than once per process
```

Likely cause: binary packages were reinstalled without kernel restart.

### 6) `timm` conflict during model load

```text
ImportError: cannot import name 'ImageNetInfo' from 'timm.data'
```

For Sentence-BERT text similarity, `timm` is not required. You can remove it:

```python
%pip uninstall -y timm
```

Then restart kernel and run import test again.

## Practical repair sequence

### Step 1. Confirm active kernel

```python
import sys
print(sys.executable)
```

Expected path should point to:

```text
...\miniconda3\envs\aqa-data\python.exe
```

### Step 2. Remove common conflicts

```python
%pip uninstall -y timm mistral-common
```

If environment is heavily mixed:

```python
%pip uninstall -y transformers sentence-transformers
%pip install "transformers==4.52.4" "sentence-transformers==5.1.2"
```

### Step 3. Restart kernel

This step is required.

### Step 4. Run clean import test

```python
import numpy
import scipy
import sklearn
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer

print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Sentence-BERT import OK")
```

## Notebook-side debugging recommendation

Avoid silently swallowing Sentence-BERT import errors while debugging. Printed exceptions explicitly:

```python
def _mean_sbert_similarity(sub):
    gt, pred = _valid_text_pairs(sub)
    if len(gt) == 0:
        return np.nan

    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as e:
        print(f"Sentence-BERT import failed: {e}")
        return np.nan
```


