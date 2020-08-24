
source $( ws_find conda )/conda/etc/profile.d/conda.sh
conda activate neuro_learn

job_id=$( grep -oP "[0-9]+(?=\.job)" <<<"$1" )
log_file=$( ws_find Neural-Learning )/Neuro-Backprop/backprop/logs/${job_id}
i=0
while IFS= read -r line
do
  if [ $i == 0 ] 
  then
    dir_res="$line"
  else
    python -u $( ws_find Neural-Learning )/Neuro-Backprop/backprop/PyraLNet.py "yinyang" --config "$line" --dir "$dir_res" &> "${log_file}_${i}.log" &
  fi
  i=$((i+1))
done < "$1"

wait
