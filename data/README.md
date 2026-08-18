# Dataset

This dataset requires the State Farm Distracted Driver Detection dataset.
The expected structure should be:

```
data/
│
├── driver_imgs_list.csv
├── imgs/
│   ├── train/
│   │   ├── c0/
│   │   ├── c1/
│   │   └── ... (up to c9)
│   └── test/
```

Since the dataset is large, it should not be committed to Git. Please make sure the structure above exists before running the training script.
