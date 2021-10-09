## Relation-aware instance refinement for weakly supervised visual grounding

### 1. build detectron2 by following the official instrucment

### 2. training for flickr30k entities
```
sh scripts/train.sh ## change the MODEL.VG.NETWORK 'RegRel' into "Baseline", 'Baseline_s2', 'Reg' to get the ablation study results
```

### 3. training for KAC models

```
sh scripts/train_kac.sh ## get the final results

```

### 4. data preparation

We will release the processed dataset later.