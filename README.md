# NER evaluation for CoNLL 2003

## Modification

- modify the project for CoNLL 2003 data.
- add train.sh, predict.sh
- add multi-layered fused_lstm_layer() which uses LSTMBlockFusedCell.
- add tf.train.LoggingTensorHook for printing loss while training.
- add tf.estimator.train_and_evaluate() with stop_if_no_increase_hook()

## How to train and predict

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
$ ./train.sh -v -v
$ tensorboard --logdir output/result_dir/ --port 6008
```
![](/eval_f.png)
![](/loss.png)

### predict
```
* select the best model among output/result_dir and edit 'output/result_dir/checkpoint' file.
$ ./predict.sh -v -v

$ cat output/result_dir/predicted_results.txt

$ more output/result_dir/label_test.txt


SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
O
X
X
X
O
B-PER
X
X
...

$ perl conlleval.pl < pred.txt
processed 46435 tokens with 5648 phrases; found: 5689 phrases; correct: 5169.
accuracy:  98.26%; precision:  90.86%; recall:  91.52%; FB1:  91.19
              LOC: precision:  93.75%; recall:  92.57%; FB1:  93.15  1647
             MISC: precision:  79.75%; recall:  83.05%; FB1:  81.37  731
              ORG: precision:  88.29%; recall:  89.40%; FB1:  88.84  1682
              PER: precision:  95.58%; recall:  96.29%; FB1:  95.93  1629

processed 46435 tokens with 5648 phrases; found: 5715 phrases; correct: 5187.
accuracy:  98.25%; precision:  90.76%; recall:  91.84%; FB1:  91.30
              LOC: precision:  92.78%; recall:  93.17%; FB1:  92.97  1675
             MISC: precision:  79.46%; recall:  83.19%; FB1:  81.28  735
              ORG: precision:  88.33%; recall:  90.25%; FB1:  89.28  1697
              PER: precision:  96.39%; recall:  95.86%; FB1:  96.12  1608
```

### dev note
```
- ELMo
[epoch 33/70] dev precision, recall, f1(token):
precision, recall, fscore
[0.9978130380159136, 0.9583333333333334, 0.9263271939328277, 0.9706510138740662, 0.9892224788298691, 0.9701897018970189, 0.9464285714285714, 0.8797653958944281, 0.9323308270676691, 0.959865053513262]
[0.9979755671902268, 0.9433258762117822, 0.9273318872017353, 0.987513572204126, 0.9831675592960979, 0.9744148067501361, 0.9174434087882823, 0.8670520231213873, 0.9649805447470817, 0.9590840404510055]
[0.9978942959852019, 0.9507703870725291, 0.9268292682926829, 0.9790096878363832, 0.9861857252494244, 0.9722976643128735, 0.9317106152805951, 0.8733624454148471, 0.9483747609942639, 0.9594743880458168]
new best f1 score! : 0.9594743880458168
max model saved in file: ./checkpoint/model_max.ckpt

- BERT
[epoch 41/70]
INFO:tensorflow:Saving dict for global step 21000: eval_accuracy = 0.9930421, eval_f = 0.9581444, eval_loss = 1.6283314, eval_precision = 0.9583672, eval_recall = 0.9579217, global_step = 21000, loss = 1.5747236
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 21000: /data1/index.shin/BERT-BiLSTM-CRF-NER/output/result_dir/model.ckpt-21000
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
