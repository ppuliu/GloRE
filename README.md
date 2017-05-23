## Global Relation Embedding for Relation Extraction (GloRE)
GloRE is a relation embedding model that can be used to augment existing relation extraction models and significantly improve their performance. Most remarkably, for the top 1,000 relational facts discovered by the best existing model (PCNN+ATT), the precision can be improved from 83.9% to 89.3%.

## Prerequisite
* Python 2.7
* Tensorflow 0.11

## Results
The result files from held-out and manual evaluations are included in [`results`](https://github.com/ppuliu/GloRE/tree/master/results). To reproduce the figures and tables in the paper, simply follow the IPython notebook:

```bash
plot.ipynb
```

We've also provided a pretrained model in [`runs/pretrained_model`](https://github.com/ppuliu/GloRE/tree/master/runs/pretrained_model). You can use it to re-generate the Precision-Recall files with the following command:

```bash
python steps.py --steps 2,4 --model_dir runs/pretrained_model/
```

## Data
We use the NYT dataset as an example to show how to use our model to improve **any** existing relation extraction tool. The original NYT dataset can be downloaded from http://iesl.cs.umass.edu/riedel/ecml/ or https://github.com/thunlp/NRE.

We've provided the following pre-processed files in [`data`](https://github.com/ppuliu/GloRE/tree/master/data):

* *data.train.gz / data.valid.gz* : training and validation files that are in the format of 
    ```
    textual relation [tab] KB relation [tab] weight
    ```
    As an example,
    ```
    <-nmod:for>##center##<-dobj>##created##<nsubj>##seminary##<compound>    /people/person/religion 0.666666666667
    ```
* *kb_relation2id.txt* : the set of target KB relations and their ids.
* *left.20000.vocab / right.-1.vocab* : vocabulary files. For the encoder input, we keep the most frequent 20,000 tokens in the vocabulary.
* *left.20000.word2vec.vocab.npy* : 300-dimensional word2vec vectors pre-trained on the Google News corpus.
* *train_textual_relation.gz / test_textual_relation.gz* : textual relations extracted from the training / testing corpus.
* *train_pcnn_att_scores.gz / test_pcnn_att_scores.gz* : relation extraction scores from PCNN+ATT (previously best performing model). The format is as follows:
    ```
    #
    entity1 [tab] entity2
    textual relaiton id (line number)
    one-hot encoding for the corresponding KB relation
    scores of each KB relation generated from the existing relation extraction tool
    ```
    As an example,
    ```
    #
    m.010016        m.0492jkz
    542694
    1       0       0       0       0       0       0       0       0       0       0       0       0       0       0
           0       0       0       0       0       0       0       0       0       0       0       0       0       0
           0       0       0       0       0       0       0       0       0       0       0       0       0       0
           0       0       0       0       0       0       0       0       0       0
    0.999106        0.000001        0.000000        0.000000        0.000000        0.000000        0.000054        0.000000        0.000001        0.000002        0.000000        0.000000        0.000001        0.000003        0.000000        0.000000        0.000000        0.000007        0.000002        0.000000        0.000000        0.000000
            0.000000        0.000004        0.000001        0.000000        0.000000        0.000000        0.000000
            0.000003        0.000000        0.000000        0.000000        0.000003        0.000000        0.000001
            0.000001        0.000003        0.000000        0.000000        0.000000        0.000000        0.000000
            0.000001        0.000000        0.000001        0.000000        0.000001        0.000443        0.000002
            0.000351        0.000000        0.000000
    #
    ```
## Step-by-Step
To run the training and testing all together, it can be as simple as
```bash
python steps.py
```
By default, this command will run through the following 4 steps:
1. Train a GloRE model
    
    You can run this step alone with
    ```
    python steps.py --steps 1
    ```
    OR
    ```bash
    python scripts/train_rel2vec.py your_model_directory
    ```
2. Extract GloRE scores
    
    You can run this step alone with
    ```
    python steps.py --steps 2 --model_dir your_model_directory
    ```
    OR
    ```bash
    python scripts/test_rel2vec.py your_model_directory
    ```
3. Train a merging model to combine GloRE scores with external scores
    
    You can run this step alone with
    ```
    python steps.py --steps 3 --model_dir your_model_directory
    ```
    OR
    ```bash
    python scripts/train_mlp.py your_model_directory
    ```
3. Generate final scores and Precision-Recall files
    
    You can run this step alone with
    ```
    python steps.py --steps 4 --model_dir your_model_directory
    ```
    OR
    ```bash
    python scripts/test_mlp.py your_model_directory
    ```
    The Precision-Recall files will be located in `your_model_directory/output`
    
We also provide some scripts that allow you to interactively explore the trained GloRE model. Given a textual relation, you can check GloRE's predictions of KB relations with
```bash
python scripts/interactive_rel2vec.py your_model_directory
```

## Configuration
All the hyperparameters can be changed by modifying files in [`scripts`](https://github.com/ppuliu/GloRE/tree/master/scripts).

## Cite
If you use the code in your research, please cite the following paper:
>**Global Relation Embedding for Relation Extraction**  
>Yu Su, Honglei Liu, Semih Yavuz, Izzeddin Gur, Huan Sun, and Xifeng Yan  
>arXiv preprint arXiv:1704.05958 (2017)
