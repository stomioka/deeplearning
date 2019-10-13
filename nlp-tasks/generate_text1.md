# Generate Text

```python
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
```
```python
tokenizer = Tokenizer()

data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)
```
[OUT]
```out
{'and': 1, 'the': 2, 'a': 3, 'in': 4, 'all': 5, 'i': 6, 'for': 7, 'of': 8, 'lanigans': 9, 'ball': 10, 'were': 11, 'at': 12, 'to': 13, 'she': 14, 'stepped': 15, 'his': 16, 'girls': 17, 'as': 18, 'they': 19, 'til': 20, 'he': 21, 'again': 22, 'got': 23, 'boys': 24, 'round': 25, 'that': 26, 'her': 27, 'there': 28, 'three': 29, 'weeks': 30, 'up': 31, 'out': 32, 'him': 33, 'was': 34, 'spent': 35, 'learning': 36, 'new': 37, 'steps': 38, 'long': 39, 'away': 40, 'left': 41, 'friends': 42, 'relations': 43, 'when': 44, 'wall': 45, 'myself': 46, 'nice': 47, 'just': 48, 'dancing': 49, 'merry': 50, 'tipped': 51, 'me': 52, 'soon': 53, 'time': 54, 'old': 55, 'their': 56, 'them': 57, 'danced': 58, 'dublin': 59, 'an': 60, 'put': 61, 'leg': 62, 'miss': 63, 'fainted': 64, 'from': 65, 'town': 66, 'athy': 67, 'one': 68, 'jeremy': 69, 'lanigan': 70, 'battered': 71, 'hadnt': 72, 'pound': 73, 'father': 74, 'died': 75, 'made': 76, 'man': 77, 'farm': 78, 'ten': 79, 'acres': 80, 'ground': 81, 'gave': 82, 'grand': 83, 'party': 84, 'who': 85, 'didnt': 86, 'forget': 87, 'come': 88, 'if': 89, 'youll': 90, 'but': 91, 'listen': 92, 'ill': 93, 'make': 94, 'your': 95, 'eyes': 96, 'glisten': 97, 'rows': 98, 'ructions': 99, 'be': 100, 'sure': 101, 'free': 102, 'invitation': 103, 'might': 104, 'ask': 105, 'minute': 106, 'both': 107, 'bees': 108, 'cask': 109, 'judy': 110, 'odaly': 111, 'little': 112, 'milliner': 113, 'wink': 114, 'give': 115, 'call': 116, 'arrived': 117, 'with': 118, 'peggy': 119, 'mcgilligan': 120, 'lashings': 121, 'punch': 122, 'wine': 123, 'ladies': 124, 'potatoes': 125, 'cakes': 126, 'bacon': 127, 'tea': 128, 'nolans': 129, 'dolans': 130, 'ogradys': 131, 'courting': 132, 'songs': 133, 'went': 134, 'plenty': 135, 'water': 136, 'harp': 137, 'once': 138, 'sounded': 139, 'taras': 140, 'hall': 141, 'sweet': 142, 'nelly': 143, 'gray': 144, 'rat': 145, 'catchers': 146, 'daughter': 147, 'singing': 148, 'together': 149, 'doing': 150, 'kinds': 151, 'nonsensical': 152, 'polkas': 153, 'room': 154, 'whirligig': 155, 'julia': 156, 'we': 157, 'banished': 158, 'nonsense': 159, 'twist': 160, 'reel': 161, 'jig': 162, 'ach': 163, 'mavrone': 164, 'how': 165, 'mad': 166, 'youd': 167, 'think': 168, 'ceiling': 169, 'would': 170, 'fall': 171, 'brooks': 172, 'academy': 173, 'learn': 174, 'nothing': 175, 'hearty': 176, 'around': 177, 'couples': 178, 'groups': 179, 'accident': 180, 'happened': 181, 'young': 182, 'terrance': 183, 'mccarthy': 184, 'right': 185, 'through': 186, 'finnertys': 187, 'hoops': 188, 'poor': 189, 'creature': 190, 'cried': 191, 'meelia': 192, 'murther': 193, 'called': 194, 'brothers': 195, 'gathered': 196, 'carmody': 197, 'swore': 198, 'hed': 199, 'go': 200, 'no': 201, 'further': 202, 'had': 203, 'satisfaction': 204, 'midst': 205, 'row': 206, 'kerrigan': 207, 'cheeks': 208, 'same': 209, 'red': 210, 'rose': 211, 'some': 212, 'lads': 213, 'declared': 214, 'painted': 215, 'took': 216, 'small': 217, 'drop': 218, 'too': 219, 'much': 220, 'suppose': 221, 'sweetheart': 222, 'ned': 223, 'morgan': 224, 'so': 225, 'powerful': 226, 'able': 227, 'saw': 228, 'fair': 229, 'colleen': 230, 'stretched': 231, 'by': 232, 'tore': 233, 'under': 234, 'table': 235, 'smashed': 236, 'chaneys': 237, 'oh': 238, 'twas': 239, 'then': 240, 'runctions': 241, 'lick': 242, 'big': 243, 'phelim': 244, 'mchugh': 245, 'replied': 246, 'introduction': 247, 'kicked': 248, 'terrible': 249, 'hullabaloo': 250, 'casey': 251, 'piper': 252, 'near': 253, 'being': 254, 'strangled': 255, 'squeezed': 256, 'pipes': 257, 'bellows': 258, 'chanters': 259, 'ribbons': 260, 'entangled': 261, 'end': 262}
26
```
![](images/generate_text1-a76c0162.png)
```python
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```



Let's take a look at the index of the first sentence.
```python

print(tokenizer.word_index['in'])
print(tokenizer.word_index['the'])
print(tokenizer.word_index['town'])
print(tokenizer.word_index['of'])
print(tokenizer.word_index['athy'])
print(tokenizer.word_index['one'])
print(tokenizer.word_index['jeremy'])
print(tokenizer.word_index['lanigan'])
```
```
4
2
66
8
67
68
69
70
```
We made the last word as Y and used hot encoding to encode it. We could use other encoding method as well.
```python
print('       sentence: {}\\n
              padded sequence: {}\n\
              x: {}\\n
              label: {} \n\
              y: \n{}'.format(corpus[0], input_sequences[6], xs[6],labels[6],ys[6]))
```
```
       sentence: in the town of athy one jeremy lanigan
padded sequence: [ 0  0  0  4  2 66  8 67 68 69 70]
              x: [ 0  0  0  4  2 66  8 67 68 69]
          label: 70
              y: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

### Train the model
```python
model = Sequential()
  model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
  model.add(Bidirectional(LSTM(20)))
  model.add(Dense(total_words, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(xs, ys, epochs=500, verbose=1)
```


```python
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()
```

```python
plot_graphs(history, 'acc')
```

![](images/generate_text1-25c20613.png)


### prediction
```python
seed_text = "Laurence went to dublin"
next_words = 100

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)
```

```python
Laurence went to dublin round as plenty as plenty as red as the water water rose little rose so rose rose ladies youd hall youd eyes glisten hall right declared make make make glisten hall eyes glisten eyes glisten eyes glisten glisten glisten glisten youd hall youd eyes glisten hall right polkas hall right declared make make make make make make make make make glisten hall eyes glisten hall glisten glisten glisten youd eyes glisten hall glisten youd catchers catchers right hoops hoops hoops right make make make make make make glisten eyes glisten hall glisten glisten glisten glisten together how how how how
```

```python

```

```python

```
