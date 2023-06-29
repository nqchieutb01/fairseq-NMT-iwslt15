TEXT=data
fairseq-preprocess \
    --source-lang en --target-lang vi \
    --trainpref $TEXT/train.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/test.bpe \
    --destdir data-bin/iwslt15.en-vi --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20 \
    --joined-dictionary	


CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt15.en-vi \
    --arch transformer_iwslt_de_en \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 2500 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --save-dir share_embedding_4k.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --max-epoch 30 \
    --log-format=json --log-interval=100 2>&1 | tee train.log


fairseq-generate data-bin/iwslt15.en-vi \
    --path share_embedding_4k/checkpoint_best.pt \
    --beam 5 --remove-bpe=sentencepiece
