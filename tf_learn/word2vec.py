import collections
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 20
embedding_size = 2
num_sampled = 15  # Number of negative examples to sample.
sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]
words = " ".join(sentences).split()
print(words)
count = collections.Counter(words).most_common()
print(count[:5])

idx2word = [i[0] for i in count]
word2idx = {w: i for i, w in enumerate(idx2word)}
print(word2idx)
voc_size = len(word2idx)
print(voc_size)

data = [word2idx[word] for word in words]
print(data)

# Let's make a training data for window size 1 for simplicity
# ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
cbow_pairs = []

for i in range(1, len(data) - 1):
    cbow_pairs.append([[data[i-1], data[i+1]], data[i]])

# Let's make skip-gram pairs
# (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
skip_pairs = []
for c in cbow_pairs:
    skip_pairs.append([c[1], c[0][0]])
    skip_pairs.append([c[1], c[0][1]])

def generate_batch(size):
    assert size < len(skip_pairs)
    x_data = []
    y_data = []
    index = np.random.choice(range(len(skip_pairs)), size, replace=False)
    for i in index:
        x_data.append(skip_pairs[i][0])
        y_data.append(skip_pairs[i][1])
    return x_data, y_data

print ('Batches (x, y)', generate_batch(3))

train_inputs = tf.placeholder(tf.int32, [batch_size])
train_labels = tf.placeholder(tf.int32, [batch_size])
train_labels1 = tf.expand_dims(train_labels, axis=1)

embeddings = tf.Variable(tf.random_normal([voc_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

nce_weights = tf.Variable(tf.random_normal([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels1, embed, num_sampled, voc_size))
train_step = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(loss)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        batch_x, batch_y = generate_batch(batch_size)
        _, temp_loss = sess.run([train_step, loss], feed_dict={train_inputs: batch_x, train_labels: batch_y})
        if i % 10 == 0:
            print("Loss at ", i, temp_loss)
        trained_embeddings = embeddings.eval()

if trained_embeddings.shape[1] == 2:
    labels = idx2word[:30] # Show top 10 words
    for i, label in enumerate(labels):
        x, y = trained_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()
    plt.savefig("word2vec.png")