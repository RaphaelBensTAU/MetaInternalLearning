# Meta Internal Learning

This repository is the official implementation of Meta Internal Learning by Raphael Bensadoun, Shir Gur, Tomer Galanti, Lior Wolf.

**[Project](PLACE_HOLDER) | [arXiv](PLACE_HOLDER) | [Code](https://github.com/RaphaelBensTAU/MetaInternalLearning.git)**

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training on a dataset

1. Create a folder X containing the images. (see structure in data folder)
2. Determine how many iterations Y to train by scale (depends mostly on the size of the dataset, you may refer to the appendix for reference).
3. Run 
```
python train.py --image-path X --batch-size 16 --visualize --niter Y --min-size 25 --checkname X_result --SAVE-MODEL --SAVE-IMGS
```

Generated images, tensorboard logs and trained models are stored in MetaInternalLearning/run/X_result.

The default input format is jpg. Use '--file-suffix png' for .png files.

## Examples from paper -

Places-50 -
```
python train.py --image-path data/places_50 --batch-size 16 --visualize --niter 4000  --min-size 28 --checkname places_50_result --SAVE-MODEL --SAVE-IMGS
```

LSUN-50
```
python train.py --image-path data/lsun_50 --batch-size 16 --visualize --niter 5000  --checkname lsun_50_result --SAVE-MODEL --SAVE-IMGS
```

Valley dataset can be downloaded here - http://places2.csail.mit.edu/download.html (256x256 small images) and can be divided into subsets as mentioned in the paper.

V500 -
```
python train.py --image-path data/V500 --batch-size 16 --visualize --niter 25000 --min-size 25 --checkname v500_result --ar 1 --SAVE-MODEL 
```

V2500 -
```
python train_dataset_parallel.py --image-path data/V2500 --batch-size 16 --niter 100000 --rec-weight 50 --min-size 25 --checkname v2500_result --ar 1 --SAVE-MODEL 
```

V5000 -
```
python train_dataset_parallel.py --image-path data/V5000 --batch-size 16 --niter 150000 --rec-weight 50 --min-size 25 --checkname v5000_result --ar 1 --SAVE-MODEL 
```

## Applications

Coming soon!
