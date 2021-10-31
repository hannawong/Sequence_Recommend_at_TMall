import pickle as pkl
import pandas as pd
import random
import numpy as np

RAW_DATA_FILE = '../data/UserBehavior.csv'  # input

Test_File = "../data/taobao_test.txt" # output
Train_File = "../data/taobao_train.txt"
Feature_File = "../data/taobao_feature.pkl"

Train_out = open(Train_File, 'w')
Test_out = open(Test_File, 'w')

MAX_LEN_ITEM = 200   ### TODO: Get length distribution from *edav*

def remap(df):  
    '''
    remap every id into [0,total_feature_num]. For embedding table use.
    '''
    ##### map item id
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))
    df['iid'] = df['iid'].map(lambda x: item_map[x])
    ##### map user id
    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(item_len, item_len + user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])
    ## map category id
    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(user_len + item_len, user_len + item_len + cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])
    #### map btag id
    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(user_len + item_len + cate_len, user_len + item_len + cate_len + btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])
    return df, item_len, user_len + item_len + cate_len + btag_len + 1 #+1 is for unknow btag in prediction



def gen_dataset(user_df, item_df, item_cnt, feature_size):

    train_sample_list = []
    test_sample_list = []
    user_last_touch_time = []  ## the last interaction time of each user ###

    for uid, history in user_df:  ###history is a dataframe for each user, with column ["uid","iid","cid","btag","time"], sorted by timestamp
        user_last_touch_time.append(list(history['time'])[-1])

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    #### split test and training set with time, to prevent future information leak ####
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)] 

    cnt = 0
    for uid, history in user_df:
        cnt += 1
        print(cnt)
        ######### get history #######
        item_hist = list(history['iid'])
        cate_hist = list(history['cid'])
        btag_hist = list(history['btag'])

        ######## target item is the last touch ########
        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        target_item_btag = feature_size  # unknown btag
        target_item_time = list(history['time'])[-1]
        label = 1  # positive sample
        if target_item_time > split_time: # decide whether test or train
            test = True
        else:
            test = False

        #############    negative sampling   ###############
        ## TODO: only use random negative sampling now. Should change it to in-batch
        ## negative sampling, with hard negative sampling method or negative sampling
        ## based on frequency prediction (Youtube Two Tower)

        neg = random.randint(0, 1)
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]: ## must be different than target item
                target_item = random.randint(0, item_cnt - 1)
                target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]
                target_item_btag = feature_size

        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        item_part.append([uid, target_item, target_item_cate, target_item_btag])
        
        ###########      padding and truncating       ###########
        if len(item_part) <= MAX_LEN_ITEM:  ###if history length less than MAX_LEN, then pad it in the front
            item_part_pad =  [[0] * 4] * (MAX_LEN_ITEM - len(item_part)) + item_part
        else:  ## if history length greater than MAX_LEN, then cut it
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]

        if test:
            cat_list = []
            item_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            test_sample_list.append(str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(map(str, item_list)) + "\t" +",".join(map(str, cat_list))+"\n")
        else: ## train
            cat_list = []
            item_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            train_sample_list.append(str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(map(str, item_list)) + "\t" +",".join(map(str, cat_list))+"\n")

    train_sample_length_quant = len(train_sample_list)//256*256  #### grounding the length
    test_sample_length_quant = len(test_sample_list)//256*256
    train_sample_list = train_sample_list[:train_sample_length_quant]
    test_sample_list = test_sample_list[:test_sample_length_quant]

    random.shuffle(train_sample_list)

    return train_sample_list, test_sample_list

def write_to_file(train_sample_list, test_sample_list):
    '''
    Output training set and testset to file
    '''
    for train_sample in train_sample_list:
        # train_sample format:
        # uid \t target_item \t target_item_cate \t label \t hist_item_list \t hist_cat_list \t
        Train_out.write(train_sample)

    for test_sample in test_sample_list:
        Test_out.write(test_sample)


def main():
    ####### read in dataset...
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    print("read ok!!!")
    ####### remap id
    df, item_cnt, feature_size = remap(df) ## feature size is the total id cnt of uid,iid,cid and btag;
    print("item cnt:", item_cnt, " total feature size:", feature_size)
    feature_total_num = feature_size + 1
    pkl.dump(feature_total_num, open(Feature_File,"wb"))
    ####### group by
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    train_sample_list, test_sample_list = gen_dataset(user_df, item_df, item_cnt, feature_size)
    write_to_file(train_sample_list,test_sample_list)
    print("="*30,"preprocess finish!","="*30)

if __name__ == '__main__':
    main()