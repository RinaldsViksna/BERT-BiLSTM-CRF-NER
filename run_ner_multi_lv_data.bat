SET MODEL_DIR=E:\BERT\models\multi_cased_L-12_H-768_A-12
SET OUT_DIR=E:\NER\BERT-BiLSTM-CRF-rinalds\bert_base_multi
SET DATA_DIR=E:\NER\Data\data_lv_bert_ner\marcis

  python bert_lstm_ner.py^
    --task_name="NER" ^
    --do_lower_case=False ^
    --crf=True ^
    --do_train=True ^
    --do_eval=True   ^
    --do_predict=True ^
    --data_dir=%DATA_DIR%   ^
    --vocab_file=%MODEL_DIR%\vocab.txt  ^
    --bert_config_file=%MODEL_DIR%\bert_config.json ^
    --init_checkpoint=%MODEL_DIR%\bert_model.ckpt   ^
    --max_seq_length=128   ^
    --train_batch_size=4  ^
    --learning_rate=2e-5   ^
    --num_train_epochs=4.0   ^
    --output_dir=%OUT_DIR%


perl conlleval.pl -d \t < %OUT_DIR%/pred.txt
