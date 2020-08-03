
source $( ws_find conda )/conda/etc/profile.d/conda.sh
conda activate neuro_learn

i=0
while IFS= read -r line
do
  if [ $i == 0 ] 
  then
    dir_res="$line"
  else
    python $( ws_find Neural-Learning )/Neuro-Backprop/backprop/PyraLNet.py "yinyang" --config "$line" --dir "$dir_res" &
  fi
  i=$((i+1))
done < "$1"

wait
