#!/bin/bash
FILE=$1

# find . -name 'log.txt' | xargs  -I % ~/eki/eki-transformer-dev/scripts/metrics.sh % > logs.toml
echo "[[runs]]"
echo "log_file = \"$1\""

# batch_size
# 'batch_size': 32
output=$(grep  -E -i -w  'Dataloader cfg:' $FILE  | awk '{print substr($0, index($0, "batch_size") )}' | cut -d "," -f 1 | grep -o -P '[\d]*')
echo "batch_size = \"$output\""

# quant_config
output=$(grep  -E -i -w  'quantization\/model'  $FILE  | awk '{print substr($0, index($0, "quantization/model") )}' | cut -d "=" -f 2 | cut -d "'" -f 1)
echo "quant_config = \"$output\""

# model config
output=$(grep  -E -i -w  'Model config:'  $FILE | cut -d '-' -f 4 | sed -n '0,/config/p' | tr -d ' ')
echo "config = \"$output\""
# nuber of params
output=$(grep  -E -i -w  'number of parameters:'  $FILE | awk '{print $7}' | sed -n '0,/M/p')
echo "params = \"$output\""
# running loss during training 
output=$(grep  -E -i -w  'qtransform.run.train\]\[INFO\] -   batch'  $FILE | awk '{print $7}' | sed 's/.$//' |  awk '{printf(" %f ,",$1 )}')
echo "train_loss = [ $output ]"

## eval loss after epochs
#output=$(grep  -E -i -w  'AVERAGE EVAL LOSS FOR EPOCH'  $FILE | awk '{print $10}' | awk '{printf("%f ,",$1 )}')
#echo "eval_loss_epoch = [ $output ]"

# eval for batches
output=$(grep  -E -i -w  'AVERAGE EVAL LOSS FOR BATCHES'  $FILE | awk '{print $10}' | awk '{printf("%f ,",$1 )}')
best_loss=$(echo $output |  sed 's/,/\n/g' | sed -r '/^\s*$/d' | sort -h | head -n1)
echo "eval_loss = [ $output ]"

# number of entries per eval iters, can be used to split the above
# output=$(sed -n '0,/eval_data/p' $FILE  | sed  -n '/train_data/,/eval/p' | wc -l | xargs bash -c 'echo $(($0 - 3))')

echo "batches_per_eval_datapoint = 500"
echo "batches_per_train_datapoint = 10"

# best loss
echo "best_loss = $best_loss" 
# perplexity
ppl=$(echo "2,71828 $best_loss" | perl -lane 'print $F[0]**$F[1]')
#ppl=$(echo "e($best_loss)" | bc -l)
#let ppl=$((2,718281828**$best_loss))
echo "perplexity = $ppl"


# last loss before eval (loss after evry epoch)
#output=$(grep  -E -i -w  'last train loss was'  $FILE | awk '{print $8}' |  awk '{printf(" %f ,",$1 )}')
#echo "eval_loss_last_train = [ $output ]"
# checkpoints paths
output=$(grep  -E -i -w  'Model checkpoint saved'  $FILE | awk '{print "\042"$8"\042"}' | sort -u | awk '{print $1 ", "}')
echo "checkpoints = [ $output ]"
# exported model, might nnot reutrn anything if model was not exported
output=$(grep -E -i  -w 'exporting...'  $FILE | awk -F "exporting..." '{print $2}'sociated | tr -d ' ')
echo "export = \"$output\""
# all the logs
# find . -name 'log.txt' | xargs  -I % ~/eki/eki-transformer-dev/scripts/metrics.sh %