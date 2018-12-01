# NER evaluation for CoNLL 2003

## How to run and evaluate

```
* download BERT model

$ ls cased_L-12_H-768_A-12 uncased_L-12_H-768_A-12
cased_L-12_H-768_A-12:
bert_config.json  bert_model.ckpt.data-00000-of-00001  bert_model.ckpt.index  bert_model.ckpt.meta  vocab.txt

uncased_L-12_H-768_A-12:
bert_config.json  bert_model.ckpt.data-00000-of-00001  bert_model.ckpt.index  bert_model.ckpt.meta  vocab.txt

$ ln -s cased_L-12_H-768_A-12 checkpoint

$ ./run.sh -v -v

$ cat output/result_dir/predicted_results.txt
#1. fine-tuning, modeling.BertModel(..., is_training=is_training, ...), num_train_epochs=3
eval_f = 0.95154184
eval_precision = 0.9613734
eval_recall = 0.9507772
global_step = 1405
loss = 1.3503028

#2. feature-based, modeling.BertModel(..., is_training=False, ...), num_train_epochs=3
eval_f = 0.95870125
eval_precision = 0.9256314
eval_recall = 0.97033226
global_step = 1405
loss = 1.3940833

$ more output/result_dir/label_test.txt

[CLS]
[SEP]

SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
[CLS]
O
X
X
X
O
B-PER
X
X
...

$ python ext.py < output/result_dir/label_test.txt > label.txt
$ python tok.py --vocab_file checkpoint/vocab.txt --do_lower_case False < NERdata/test.txt > test.txt.tok
$ python merge.py --a_path test.txt.tok --b_path label.txt > pred.txt
$ perl conlleval.pl < pred.txt
#1. fine-tuning
processed 46476 tokens with 5596 phrases; found: 5706 phrases; correct: 5114.
accuracy:  98.14%; precision:  89.62%; recall:  91.39%; FB1:  90.50
              LOC: precision:  92.74%; recall:  92.68%; FB1:  92.71  1652
             MISC: precision:  74.23%; recall:  82.48%; FB1:  78.14  780
              ORG: precision:  88.69%; recall:  89.76%; FB1:  89.22  1680
              PER: precision:  94.92%; recall:  95.70%; FB1:  95.31  1594 

#2. feature-based
processed 46666 tokens with 5648 phrases; found: 5758 phrases; correct: 5147.
accuracy:  98.14%; precision:  89.39%; recall:  91.13%; FB1:  90.25
              LOC: precision:  93.12%; recall:  91.73%; FB1:  92.42  1643
             MISC: precision:  74.19%; recall:  81.91%; FB1:  77.86  775
              ORG: precision:  86.77%; recall:  90.01%; FB1:  88.36  1723
              PER: precision:  95.67%; recall:  95.67%; FB1:  95.67  1617
```

----

# README from source git

Tensorflow solution of NER task Using BiLSTM-CRF model with Google BERT Fine-tuning

使用谷歌的BERT模型在BLSTM-CRF模型上进行预训练用于中文命名实体识别的Tensorflow代码'

Welcome to star this repository!

The Chinese training data($PATH/NERdata/) come from:https://github.com/zjy-ucas/ChineseNER 
  
The CoNLL-2003 data($PATH/NERdata/ori/) come from:https://github.com/kyzhouhzau/BERT-NER 
  
The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py  


Try to implement NER work based on google's BERT code and BiLSTM-CRF network!


## How to train

#### 1.using config param in terminal

```
  python3 bert_lstm_ner.py   \
                  --task_name="NER"  \ 
                  --do_train=True   \
                  --do_eval=True   \
                  --do_predict=True
                  --data_dir=NERdata   \
                  --vocab_file=checkpoint/vocab.txt  \ 
                  --bert_config_file=checkpoint/bert_config.json \  
                  --init_checkpoint=checkpoint/bert_model.ckpt   \
                  --max_seq_length=128   \
                  --train_batch_size=32   \
                  --learning_rate=2e-5   \
                  --num_train_epochs=3.0   \
                  --output_dir=./output/result_dir/ 
 ```       
 #### 2. replace the BERT path and project path in bert_lstm_ner.py.py
 ```
 if os.name == 'nt':
    bert_path = '{your BERT model path}'
    root_path = '{project path}'
else:
    bert_path = '{your BERT model path}'
    root_path = '{project path}'
 ```

## result:
all params using default
#### In dev data set:
![](/picture1.png)

#### In test data set
![](/picture2.png)

## reference: 
+ The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://github.com/kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

+ [https://github.com/zjy-ucas/ChineseNER](https://github.com/zjy-ucas/ChineseNER)

> Any problem please email me(ma_cancan@163.com)
