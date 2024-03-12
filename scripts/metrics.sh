#!/bin/bash
FILE=log.txt

# running loss during training 
grep  -E -i -w  'qtransform.run.train\]\[INFO\] -   batch'  $FILE | awk '{print $7}' | sed 's/.$//' |  awk '{printf("%f\n",$1 )}'

# number of entries per epoch, can be used to split the above
sed -n '0,/eval_data/p' $FILE  | sed  -n '/train_data/,/eval/p' | wc -l | xargs bash -c 'echo $(($0 - 2))'

# eval loss after epochs
grep  -E -i -w  'AVERAGE EVAL LOSS FOR EPOCH'  $FILE | awk '{print $10}' | awk '{printf("%f\n",$1 )}'

# last loss before eval (loss after evry epoch)
grep  -E -i -w  'last train loss was'  $FILE | awk '{print $8}' |  awk '{printf("%f\n",$1 )}'

# checkpoints paths
grep  -E -i -w  'Model checkpoint saved'  $FILE | awk '{print $8}'

# nuber of params
grep  -E -i -w  'number of parameters:'  $FILE | awk '{print $7}'

# model config
grep  -E -i -w  'Model config:'  $FILE | cut -d '-' -f 4

# exported model, might nnot reutrn anything if model was not exported
grep -E -i  -w 'exporting...'  $FILE | awk -F "exporting..." '{print $2}'sociated | tr -d ' '