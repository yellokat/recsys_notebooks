import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import vectorize, float64
from numba import jit
import time

class MatrixFactorization():
    def __init__(self, seed):
        np.random.seed(seed)
        self.n_users = 17770
        self.n_movies = 480189
        self.n_ratings = 80191701
        self.k = 50
        self.user_embeddings = np.random.rand(self.n_users, self.k)
        self.movie_embeddings = np.random.rand(self.n_movies, self.k)
        self.train_df = pd.read_csv('processed_data/train_idx.csv').values.astype(int)
        self.validation_df = pd.read_csv('processed_data/validation_idx.csv').values.astype(int)
        self.test_df = pd.read_csv('processed_data/test_idx.csv').values.astype(int)
        # learning rate adjustments
        self.user_velocity = np.zeros_like(self.user_embeddings)
        self.movie_velocity = np.zeros_like(self.movie_embeddings)
    
    def train(self, batch_size, epochs):
        dataset = self.train_df
        for _ in range(epochs):
            # shuffle dataset
            order = np.arange(dataset.shape[0])
            np.random.shuffle(order)
            #
            # compile update rules
            @jit(nopython=True, nogil=True)
            def next_velocity(batch_qi, batch_pu, user_velocity, movie_velocity, expanded_errors):
                lr = 1e-02
                reg_lambda = 0.1
                momentum = 0.9
                user_gradient = reg_lambda * batch_qi - expanded_errors * batch_pu
                movie_gradient = reg_lambda * batch_pu - expanded_errors * batch_qi
                user_velocity_next = momentum * user_velocity - lr * user_gradient
                movie_velocity_next = momentum * movie_velocity - lr * movie_gradient
                return user_velocity_next, movie_velocity_next

            n_batches = (self.n_ratings//batch_size)+1
            for batch_num in tqdm(np.arange(n_batches)):
                # create batch
                batch = dataset[order[batch_num*batch_size:(batch_num+1)*batch_size]]
                user_indices = batch[:, 0]
                movie_indices = batch[:, 1]
                batch_qi = self.user_embeddings[user_indices]
                batch_pu = self.movie_embeddings[movie_indices]
                user_velocity = self.user_velocity[user_indices]
                movie_velocity = self.movie_velocity[movie_indices]
                #
                # forward pass
                predictions = np.sum(batch_qi * batch_pu, axis=1)
                errors = batch[:, 2] - predictions
                #
                # calculate updates
                user_velocity_next, movie_velocity_next = next_velocity(
                    batch_qi,
                    batch_pu,
                    user_velocity,
                    movie_velocity,
                    np.expand_dims(errors, 1)
                )
                #
                # perform update
                self.user_velocity[batch[:, 0]] = user_velocity_next
                self.user_embeddings[batch[:, 0]] += user_velocity_next
                self.movie_velocity[batch[:, 1]] = movie_velocity_next
                self.movie_embeddings[batch[:, 1]] += movie_velocity_next
    #
    def evaluation(self, mode:str):
        if mode not in ['validation', 'test']:
            raise Exception('keyword argument mode must be one of "validation" and "test".')
        dataset = self.validation_df if mode == 'validation' else self.test_df
        # calculate rmse
        predictions = np.sum(self.user_embeddings[dataset[:, 0]] * self.movie_embeddings[dataset[:, 1]], axis=1) 
        errors = dataset[:, 2] - predictions
        rmse = np.sqrt(np.mean(errors ** 2))
        print('RMSE on {} dataset : {}'.format(mode, rmse))

if __name__ == "__main__":
    # for batch_size in [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]:
    for batch_size in [2**13]:
        model = MatrixFactorization(seed=0)
        model.evaluation('validation')
        start = time.time()
        model.train(epochs=1, batch_size=4096)
        print('Batch size :', batch_size, ', Took', time.time() - start, 'seconds.')
        model.evaluation('validation')
        model.evaluation('test')
        del model