mpirun --hostfile c8 \
-np 8 \
-bind-to none \
-map-by slot \
-mca pml ob1 -mca btl ^openib \
python3.5 train_mpi.py \
--lr 0.8 \
--bs 128 \
--epoch 200 \
--matcha \
--budget 0.5 \
-n MATCHA \
--model res \
-p \
--description experiment \
--graphid 0 \
--dataset cifar10 \
--datasetRoot ../AdaDSGD/data/ \
--savePath ./exp_result_ \
--randomSeed 1234