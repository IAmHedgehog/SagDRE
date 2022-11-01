# SagDRE
Code for KDD 2022 paper [SagDRE: Sequence-Aware Graph-Based Document-Level Relation Extraction with Adaptive Margin Loss](https://dl.acm.org/doi/abs/10.1145/3534678.3539304).

If you make use of this code in your work, please kindly cite the following paper:

```bibtex
@inproceedings{wei2022sagdre,
  title={SagDRE: Sequence-Aware Graph-Based Document-Level Relation Extraction with Adaptive Margin Loss},
  author={Wei, Ying and Li, Qi},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2000--2008},
  year={2022}
}
```
## Requirements
* Python (tested on 3.7.4)
* CUDA (tested on 10.2)
* [PyTorch](http://pytorch.org/) (tested on 1.7.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 3.4.0)
* numpy (tested on 1.19.4)
* [spacy](https://spacy.io/) (tested on 3.2.4)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data).

The CDR dataset can be obtained by following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph).

The CHR dataset can be obtained at [link](http://nactem.ac.uk/CHR/).

Please process CDR and CHR datasets by following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
SagDRE
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- chr
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
```

## Training and Evaluation
### DocRED
Train the BERT model on DocRED with the following command:

```bash
>> cd scripts
>> sh run_docred.sh  # for BERT
```

### CDR and CHR
Train CDA and CHR model with the following command:
```bash
>> sh scripts/run_cdr.sh  # for CDR
>> sh scripts/run_chr.sh  # for CHR
```

<!-- ## Saving and Evaluating Models
You can save the model by setting the `--save_path` argument before training. The model corresponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks. -->

### This code is partially based on the code of [ATLOP](https://github.com/wzhouad/ATLOP)
