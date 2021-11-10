import multiprocessing
import time
import numpy as np

import numpy
import json
import pickle as pkl
import random
import numpy as np

class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None,
                 parall=False
                ):
        self.source = open(source, 'r')
        self.source_dicts = []
        
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty
        self.sort_by_length = sort_by_length
        self.source_buffer = []
        self.k = batch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        hist_item_list = []
        hist_cate_list = []
     
        
        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                uid = int(ss[0])
                item_id = int(ss[1])
                cate_id = int(ss[2])
                label = int(ss[3])

                hist_item = map(int, ss[4].split(","))
                hist_cate = map(int, ss[5].split(","))
                
                source.append([uid, item_id, cate_id])
                target.append([label, 1-label])
                hist_item_list.append(list(hist_item)[-self.maxlen:])
                hist_cate_list.append(list(hist_cate)[-self.maxlen:])
                
                

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
        
        uid_array = np.array(source)[:,0]
        item_array = np.array(source)[:,1]
        cate_array = np.array(source)[:,2]

        target_array = np.array(target)

        history_item_array = np.array(hist_item_list)        
        history_cate_array = np.array(hist_cate_list)
              
        
        history_mask_array = np.greater(history_item_array, 0)*1.0      

        
        
        return (uid_array, item_array, cate_array), (target_array, history_item_array, history_cate_array, history_mask_array)


def generator_queue(generator, max_q_size=20,
                    wait_time=0.1, nb_worker=1):

    generator_threads = []
    q = multiprocessing.Queue(maxsize=max_q_size)
    _stop = multiprocessing.Event()
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        #start_time = time.time()
                        generator_output = next(generator)
                        #end_time = time.time()
                        #print end_time - start_time
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    print("over1")
                    #raise

        for i in range(nb_worker):
            thread = multiprocessing.Process(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except Exception:
        _stop.set()
        for p in generator_threads:
            if p.is_alive():
                p.terminate()
        q.close()
        print("over")

    return q, _stop, generator_threads