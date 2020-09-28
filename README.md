## GraphDialog

This is the Tensorflow implementation of the paper:
**GraphDialog: Integrating Graph Knowledge into End-to-End Task-Oriented Dialogue Systems**

This code has been written using Tensorflow >= 2.0.0. If you find the source codes or the datasets included in this toolkit useful to your work, please kindly consider citing our paper. The bibtex is listed below:
<pre>

</pre>

## Architecture
<p align="left">
<img src="img/Encoder_0426.png" width="50%" />
</p>

<p align="right">
<img src="img/GraphCell.png" width="50%" />
</p>

## Dependencies
* Tensorflow >= 2.0.0
* Spacy 2.2.1
* cudatoolkit 10.0.130
* cudnn 7.6.0
* tqdm 4.36.1
* nltk 3.4.5


## Training
We created `myTrain.py` to train the models. You can run:
```console
python myTrain.py -lr=0.001 -hdd=128 -dr=0.2 -bsz=128 -ds=multiwoz -maxdeps=7 -revgraph=0 -graphhdd=128 -nheads=1 -l=1 -graph_dr=0.2 -graph_layer=1
```
While training, the model with the best validation results is stored. If you want to reuse a model, please add `-path=path_name_model` argument to the call. The model is evaluated by BLEU and Entity F1.

## Testing
We created `myTest.py` to restore the checkpoints and test the models. You can run:
```console
python myTest.py -ds=multiwoz -path=<path_to_saved_model>
```

## Reproducing
We've attached our checkpoint files to facilitate the reproduction of the results in the paper. The checkpoints is stored in the following folder:
```console
save/GraphDialog-MultiWOZ/
```
You can run:
```console
python myTest.py -ds=multiwoz -path=save/GraphDialog-MultiWOZ/xxx/ckpt-xx
```

## Enjoy!

