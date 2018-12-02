# NER evaluation for CoNLL 2003

## Modification

- add multi-layered fused_lstm_layer() which uses LSTMBlockFusedCell.
- add tf.train.LoggingTensorHook for printing loss while training.
- add tf.estimator.train_and_evaluate() with stop_if_no_increase_hook()

## How to train and evaluate

### download BERT model
```
* download BERT model

$ ls cased_L-12_H-768_A-12 uncased_L-12_H-768_A-12
cased_L-12_H-768_A-12:
bert_config.json  bert_model.ckpt.data-00000-of-00001  bert_model.ckpt.index  bert_model.ckpt.meta  vocab.txt

uncased_L-12_H-768_A-12:
bert_config.json  bert_model.ckpt.data-00000-of-00001  bert_model.ckpt.index  bert_model.ckpt.meta  vocab.txt
```

### train
```
* edit 'bert_model_dir'
* edit 'lowercase=False' for cased BERT model, 'lowercase=True' for uncased.
$ ./run.sh -v -v

$ cat output/result_dir/predicted_results.txt
# cased, fine-tuning, modeling.BertModel(..., is_training=is_training, ...)
eval_f = 0.9771129
eval_precision = 0.98195297
eval_recall = 0.9791255
global_step = 13122
loss = 3.2886324

# cased, feature-based, modeling.BertModel(..., is_training=False, ...)
eval_f = 0.9801688
eval_precision = 0.86159223
eval_recall = 0.97741973
global_step = 9101
loss = 3.310536
```

### evaluate
```
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
$ lowercase='False'
$ python tok.py --vocab_file cased_L-12_H-768_A-12/vocab.txt --do_lower_case ${lowercase} < NERdata/test.txt > test.txt.tok
$ python merge.py --a_path test.txt.tok --b_path label.txt > pred.txt
$ perl conlleval.pl < pred.txt
# cased, fine-tuning
processed 46666 tokens with 5648 phrases; found: 5714 phrases; correct: 5188.
accuracy:  98.26%; precision:  90.79%; recall:  91.86%; FB1:  91.32
              LOC: precision:  92.47%; recall:  93.47%; FB1:  92.96  1686
             MISC: precision:  77.95%; recall:  83.62%; FB1:  80.69  753
              ORG: precision:  89.14%; recall:  90.43%; FB1:  89.78  1685
              PER: precision:  96.86%; recall:  95.24%; FB1:  96.04  1590

# cased, feautre-based
processed 46666 tokens with 5648 phrases; found: 5730 phrases; correct: 5146.
accuracy:  98.16%; precision:  89.81%; recall:  91.11%; FB1:  90.46
              LOC: precision:  92.00%; recall:  93.11%; FB1:  92.55  1688
             MISC: precision:  75.89%; recall:  81.62%; FB1:  78.65  755
              ORG: precision:  87.87%; recall:  89.40%; FB1:  88.63  1690
              PER: precision:  96.12%; recall:  94.93%; FB1:  95.52  1597
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
