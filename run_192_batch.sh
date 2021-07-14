SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/..
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip -d $SCRIPTPATH/datasets/shapenet_parts
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
cd -

export WANDB_API_KEY='4cc3ac6d5ce3acc51384ba12267c730229d39346'

python simclr_module.py --batch_size 192 --max_epochs 500 --num_workers 4 --online_ft 0