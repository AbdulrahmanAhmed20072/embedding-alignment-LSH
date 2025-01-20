import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords, twitter_samples

from utils import (cosine_similarity, get_dict,
                   process_tweet)
from os import getcwd

en_embeddings_subset = pickle.load(open('en_embeddings.p','rb'))
fr_embeddings_subset = pickle.load(open('fr_embeddings.p','rb'))

# embedding size
len(en_embeddings_subset) , len(fr_embeddings_subset)

# embedding dim
en_embeddings_subset['the'].shape

# english to french translation
en_fr_train = get_dict('en-fr.train.txt')
en_fr_test = get_dict('en-fr.test.txt')

len(en_fr_train), len(en_fr_test)

# first 5 words in english to french data
{ i : en_fr_train[i] for i in list(en_fr_train.keys())[:5] }

def get_matrices(en_fr_data, en_embed, fr_embed):

  X, Y = [],[]

  for i, (en_word,fr_word) in enumerate(en_fr_data.items()):

    # ensure that words in embedding
    if en_word in  en_embed.keys() and fr_word in fr_embed.keys():
      X.append(en_embed[en_word])
      Y.append(fr_embed[fr_word])

  return np.vstack(X),np.vstack(Y)

X,Y = get_matrices(en_fr_train, en_embeddings_subset, fr_embeddings_subset)

X.shape , Y.shape

def compute_loss(X, Y, R):

  m = X.shape[0]

  # difference between XR and Y
  diff = np.dot(X,R) - Y

  # ferbenius norm
  diff_squared = diff ** 2
  sum_diff_squared = np.sum(diff_squared)
  loss = sum_diff_squared / m

  return loss

def compute_gradient(X, Y, R):

  m = X.shape[0]
  gradient = (2 / m) * np.dot(X.T, np.dot(X,R) - Y)

  return gradient

def align_embeddings(X, Y, train_steps=100, learning_rate=0.01):

  # training process for R matrics
  np.random.seed(129)
  R = np.random.rand(X.shape[1],X.shape[1])

  for epoch in range(train_steps):

    # every 100 step printloss
    if epoch % 100 == 0:
      loss = compute_loss(X, Y, R)
      print(f"loss at iteration {epoch} is: {loss:.4f}")

    grad = compute_gradient(X,Y,R)

    # update R
    R -= learning_rate * grad

  return R

np.random.seed(129)
m = 10
n = 5
X_ = np.random.rand(m, n)
Y_ = np.random.rand(m, n) * .1
R = align_embeddings(X_, Y_)

R_train = align_embeddings(X,Y,400,1)

def nearest_neighbor(v, candidates, k=1):

  # v is the selected vector, candidates is all other vectors in data

  similarity_l=[]

  for vec in candidates:

    cos = cosine_similarity(v, vec)
    similarity_l.append(cos)
  k_idx = np.argsort(similarity_l)

  return k_idx[-k:]

v = np.array([1, 0, 1])
candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
nearst_idx = nearest_neighbor(v, candidates, 1)
print("nearst neighbor index", nearst_idx)
print("nearst neighbor vector", candidates[nearst_idx])

def test_vocabulary(X, Y, R):

  n_true = 0
  pred = np.dot(X,R)

  for i in range(len(pred)):

    pred_idx = nearest_neighbor(pred[i],Y).item()
    # nearest_neighbor returns the vector idx, it should match the true vector idx "i"
    if pred_idx == i:
      n_true += 1

  return n_true / len(pred)

X_val, Y_val = get_matrices(en_fr_test, en_embeddings_subset,fr_embeddings_subset)

test_vocabulary(X_val, Y_val,R_train)

from nltk.corpus import twitter_samples,stopwords
nltk.download('twitter_samples')
nltk.download('stopwords')

pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

all_tweets = pos_tweets + neg_tweets

def get_document_embedding(tweet, en_embeddings):

  # empty vector for embedding
  doc_embed = np.zeros((300))
  processed_tweet = process_tweet(tweet)

  for word in processed_tweet:

    # sum all word embeddings for a specific sentence
    doc_embed += en_embeddings.get(word , 0)

  return doc_embed

custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"

tweet_embedding = get_document_embedding(custom_tweet, en_embeddings_subset)
tweet_embedding.shape

def get_document_vecs(all_docs, en_embeddings):

  embedding_matrix_l = []
  ind2embed_dict = {}

  for i, doc in enumerate(all_docs):

    doc_embed = get_document_embedding(doc,en_embeddings)
    embedding_matrix_l.append(doc_embed)
    ind2embed_dict[i] = doc_embed

  embedding_matrix = np.vstack(embedding_matrix_l)

  return embedding_matrix, ind2embed_dict

embedding_matrix, ind2embed_dict = get_document_vecs(all_tweets, en_embeddings_subset)

embedding_matrix.shape

# {k:v for k, v in list(ind2embed_dict.items())[:1]}

N_VECS = len(all_tweets)
N_DIMS = len(ind2embed_dict[1])

N_PLANES = 10
N_UNIVERSES = 25

# each plane has a dim 300 * 10
np.random.seed(0)
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES))
            for _ in range(N_UNIVERSES)]

def hash_value_of_vector(v, planes):

  proj = np.dot(v, planes)
  sign = np.sign(proj)>=0
  h = np.squeeze(sign)
  hash = 0

  for i in range(planes.shape[1]):

    # hashing equation
    hash += np.power(2,i) * h[i]

  return int(hash)

np.random.seed(32)
idx = 0
planes = planes_l[idx]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
print(f"{hash_value_of_vector(vec, planes)}")

def make_hash_table(vecs, planes):

  num_buckets = 2 ** planes.shape[1]

  hash_table = {i : [] for i in range(num_buckets)}
  id_table = {i : [] for i in range(num_buckets)}

  for i, vec in enumerate(vecs):

    h = hash_value_of_vector(vec, planes)
    hash_table[h].append(vec)
    id_table[h].append(i)

  return hash_table, id_table

np.random.seed(0)

planes = planes_l[0]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
print(planes.shape,'')

tmp_hash_table, tmp_id_table = make_hash_table(embedding_matrix, planes)

print(f"The hash table at key 0 has {len(tmp_hash_table[0])} document vectors")
print(f"The id table at key 0 has {len(tmp_id_table[0])}")
print(f"The first 5 document indices stored at key 0 of are {tmp_id_table[0][0:5]}")

hash_tables = []
id_tables = []

for i in range(N_UNIVERSES):

  plane = planes_l[i]
  hash_table, id_table = make_hash_table(embedding_matrix, plane)

  hash_tables.append(hash_table)
  id_tables.append(id_table)

def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=N_UNIVERSES):

  vecs_to_consider_l = list()
  ids_to_consider_l = list()
  ids_to_consider_set = set()

  for universe in range(num_universes_to_use):

    planes = planes_l[universe]

    hash_value = hash_value_of_vector(v,planes)

    hash_table = hash_tables[universe]
    doc_vectors_l = hash_table[hash_value]

    id_table = id_tables[universe]
    ids_to_consider = id_table[hash_value]

    ids_to_consider_l.append(ids_to_consider)

    if doc_id in ids_to_consider:
      ids_to_consider.remove(doc_id)

    for i, id in enumerate(ids_to_consider):

      if id not in ids_to_consider_set:

        vecs_to_consider_l.append(doc_vectors_l[i])

        ids_to_consider_l.append(id)

        ids_to_consider_set.add(id)

  vecs_to_consider_arr = np.array(vecs_to_consider_l)

  nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_arr, k=k)

  nearest_neighbor_ids = [ids_to_consider_l[i] for i in nearest_neighbor_idx_l]

  return nearest_neighbor_ids

doc_id = 0
v = embedding_matrix[doc_id]
approximate_knn(doc_id, v, planes_l, 1, 1)

