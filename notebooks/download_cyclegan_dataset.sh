# Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob
# /master/datasets/download_cyclegan_dataset.sh with fixed dataset

FILE=vangogh2photo;

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip;
ZIP_FILE=./$FILE.zip;
TARGET_DIR=./datasets/$FILE/;
wget -N $URL $ZIP_FILE;
mkdir $TARGET_DIR;
unzip $ZIP_FILE -d ./datasets/;
rm $ZIP_FILE;