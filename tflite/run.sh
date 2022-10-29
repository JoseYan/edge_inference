pref=$1

for j in og dw
do
    for i in 1 2 4 8 16
    do
        m=defo'_'$i'_'$j
        echo $m
    	python export_onnx.py $m 2>&1 | tee $pref/export_onnx_$m.log
    	python onnx_to_tf.py $m.onnx 2>&1 | tee $pref/to_tflite_$m.log
    done
done

for m in  vit_tiny_224 vit_tiny_16_112 vit_tiny_8_112 mobilevit_256 mobilevit_108  beit_base vit_base_224 vit_base_16_112
do
    python export_onnx.py $m 2>&1 | tee $pref/export_onnx_$m.log
    python onnx_to_tf.py $m.onnx 2>&1 | tee $pref/to_tflite_$m.log
done
