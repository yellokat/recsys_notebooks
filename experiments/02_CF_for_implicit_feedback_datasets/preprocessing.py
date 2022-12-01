import os
import pandas as pd
from functools import reduce
from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql import functions as F

def get_unique_users_phase1():
    spark = SparkSession.builder.getOrCreate()
    path = 'data'
    fpaths = [os.path.join(path, fname) for fname in os.listdir(path) if 'combined_data' in fname]
    for idx, fpath in enumerate(fpaths):
        df = spark.read.options(delimiter=',').csv(fpath)
        df = df.filter(~df._c0.contains(':'))
        df = df.distinct()
        df.write.format("csv").save('processed_data_{}'.format(idx))

def get_unique_users_phase2():
    lst = []
    for i in range(4):
        path = 'processed_data_{}'.format(i)
        dfs = [pd.read_csv(os.path.join(path, fname), header=None) for fname in os.listdir(path) if fname.endswith('csv')]
        lst.append(pd.concat(dfs))
    df = pd.concat(lst).reset_index(drop=True)
    unique_users = sorted(df[0].unique())
    table = pd.DataFrame(unique_users).reset_index().rename(columns={0:'userId'}).set_index('userId')
    table.to_csv('userIdx.csv') 

def refine_data_phase1():
    path = 'data'
    fpaths = [os.path.join(path, fname) for fname in os.listdir(path) if 'combined_data' in fname]
    count = 0
    path = 'data'
    fpaths = [os.path.join(path, fname) for fname in os.listdir(path) if 'combined_data' in fname]
    for idx, fpath in enumerate(fpaths):
        newdata = open('refined_data_{}.txt'.format(idx), 'w')
        with open(fpath, 'rb') as f:
            for line in f:
                line = line.strip().decode()
                if line.endswith(':'):
                    movieId = line[:-1]
                else:
                    userId, rating, timestamp = line.split(',')
                    newline = ','.join([movieId, userId, rating, timestamp])
                    newdata.write(newline+'\n')
                    print(count, end='\r')
                    count += 1
        newdata.close()

def refine_data_phase2():
    spark = SparkSession.builder.getOrCreate()
    df1 = spark.read.csv('refined_data_0.txt')
    df2 = spark.read.csv('refined_data_1.txt')
    df3 = spark.read.csv('refined_data_2.txt')
    df4 = spark.read.csv('refined_data_3.txt')
    data = reduce(DataFrame.unionAll, [df1, df2, df3, df4]).toDF('movieId', 'userId', 'rating', 'timestamp')

    spec = Window.partitionBy('userId').orderBy('timestamp')
    data = data.withColumn('rank_in_partition', F.row_number().over(spec))
    counts = data.groupby('userId').count()
    joined = data.join(counts, on=['userId'], how='left')  
    with_rankp = joined.withColumn('rank_p', F.col('rank_in_partition')/F.col('count'))

    below_80 = with_rankp.filter(with_rankp.rank_p <= 0.8)
    over_80 = with_rankp.filter(with_rankp.rank_p > 0.8)
    validation, test = over_80.randomSplit([0.5, 0.5], seed=0)
    
    train = below_80.select('userId', 'movieId', 'rating').sort('userId', 'movieId')
    validation = validation.select('userId', 'movieId', 'rating').sort('userId', 'movieId')
    test = test.select('userId', 'movieId', 'rating').sort('userId', 'movieId')
    
    train.coelesce(1).write.csv('train')
    validation.coelesce(1).write.csv('validation')
    test.coelesce(1).write.csv('test')

def to_indices():
    # init spark
    spark = SparkSession.builder.getOrCreate()
    uid2idx = spark.read.csv('processed_data/userIdx.csv').toDF('userId', 'userIdx')
    #
    for mode in ['train', 'validation', 'test']:
        df = spark.read.csv('processed_data/{}.csv'.format(mode)).toDF('userId', 'movieId', 'rating')    
        join_userIdx = df.join(uid2idx, on='userId', how='left')
        join_movieIdx = join_userIdx.withColumn('movieIdx', F.col('movieId')-1)
        result = join_movieIdx.select('movieIdx', 'userIdx', 'rating').sort('movieIdx', 'userIdx')
        result.coalesce(1).write.csv('{}_idx'.format(mode))

if __name__ == "__main__":
    #get_unique_users_phase1()
    #get_unique_users_phase2()
    #refine_data_phase1()
    #refine_data_phase2()
    to_indices()
                


