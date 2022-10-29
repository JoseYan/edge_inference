arch=$1
pref=$2
for model in vit_tiny_16_112 vit_tiny_8_112 vit_tiny_224 beit_base mobilevit_108 mobilevit_256
do
    python tune_defo_local.py $m $arch 2>&1 | tee $pref/1000trials_$model'_'$arch.log
    rm *json
done

for i in 1 2 4 8 16
do
    for j in dw og
    do
        m=defo'_'$i'_'$j
        echo $m
        python tune_defo_local.py $m $arch 2>&1 | tee $pref/1000trials_$m'_'$arch.log
        rm *json
    done
done
