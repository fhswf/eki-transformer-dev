#!/bin/bash
FILE=$1

echo "[[runs]]"
echo "log_file = \"$1\""

# model config
output=$(grep  -E -i -w  'Model config:'  $FILE | cut -d '-' -f 4 | sed -n '0,/config/p' | tr -d ' ')
echo "config = \"$output\""
# nuber of params
output=$(grep  -E -i -w  'number of parameters:'  $FILE | awk '{print $7}' | sed -n '0,/M/p')
echo "params = \"$output\""
# running loss during training 
output=$(grep  -E -i -w  'qtransform.run.train\]\[INFO\] -   batch'  $FILE | awk '{print $7}' | sed 's/.$//' |  awk '{printf(" %f ,",$1 )}')
echo "train_loss = [ $output ]"

# eval loss after epochs
output=$(grep  -E -i -w  'AVERAGE EVAL LOSS FOR EPOCH'  $FILE | awk '{print $10}' | awk '{printf("%f ,",$1 )}')
echo "eval_loss_epoch = [ $output ]"

# eval for batches
output=$(grep  -E -i -w  'AVERAGE EVAL LOSS FOR BATCHES'  $FILE | awk '{print $10}' | awk '{printf("%f ,",$1 )}')
echo "eval_loss = [ $output ]"

# number of entries per eval iters, can be used to split the above
# output=$(sed -n '0,/eval_data/p' $FILE  | sed  -n '/train_data/,/eval/p' | wc -l | xargs bash -c 'echo $(($0 - 3))')
echo "eval_iter = 50"

# last loss before eval (loss after evry epoch)
#output=$(grep  -E -i -w  'last train loss was'  $FILE | awk '{print $8}' |  awk '{printf(" %f ,",$1 )}')
#echo "eval_loss_last_train = [ $output ]"
# checkpoints paths
output=$(grep  -E -i -w  'Model checkpoint saved'  $FILE | awk '{print "\042"$8"\042" ", "}')
echo "checkpoints = [ $output ]"
# exported model, might nnot reutrn anything if model was not exported
output=$(grep -E -i  -w 'exporting...'  $FILE | awk -F "exporting..." '{print $2}'sociated | tr -d ' ')
echo "export = \"$output\""
# all the logs
# find . -name 'log.txt' | xargs  -I % ~/eki/eki-transformer-dev/scripts/metrics.sh %