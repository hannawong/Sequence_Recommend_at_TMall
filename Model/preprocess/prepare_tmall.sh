'''prepare tmall data'''
## Please put UserBehavior.csv.zip under directory ../data 

unzip ../data/UserBehavior.csv.zip
rm -rf ../data/UserBehavior.csv.zip
mv UserBehavior.csv ../data
python Tmall_preprocess.py