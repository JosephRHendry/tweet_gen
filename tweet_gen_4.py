""" Machine Learning componant. Reads from a text of tweets and learns to create in that style.
Initially based on Keras LSTM example code """

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

import pandas as pd

# In[2]:


load_file = pd.read_excel('source.xlsx',sheet_name='Sheet1') # a spreadsheet of tweets
text = " "
load_data = []
#print(load_file['text'][1])

for x in load_file['text']:
    #print(x)
    load_data.append(x)
    
#for x in load_data:
#    print(x)

#print(load_data[5])   
n=0

print(load_data[4])

for x in range(len(load_data)):
    #print(x, " :", load_data[x])
    while n < len(load_data[x]):
        text += str(load_data[x][n])
        n += 1
    text += "😉" # This character is meant to be a special character
    # indicating the beginning and ending of tweets.
    n=0


    
#ext = load_data
#print(text[5])

chars = sorted(list(set(text)))
#chars = sorted(list(set(str(text))))
#print(chars)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#print(text)


# In[3]:



print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 50
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(125, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Dropout(.2))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.008)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.load_weights('tweet_gen_4.h5', by_name=True)
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    model.save_weights('tweet_gen_4.h5')
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

train = True

if train == True:
    model.fit(x, y, batch_size=128, epochs=2,
    callbacks=[print_callback])


# In[4]:


""" I need to build out several functions. First a way to sample
until a certain character. Also to sample from an abrtitrary chunk
of input. So it could act as an auto-complete.

I also want to add a sentiment analyzer function to the creation of
tweets, perhaps even on a word by word, or letter by letter? level.
"""

def get_input(text, style = 'f'):
    sentence = ''
    end_index = 0
    if style == 'f':
        start_index = random.randint(0, len(text) - maxlen - 1)
        sentence = text[start_index: start_index + maxlen]
        #print("yes it was in the f style")
    elif style == 'bt':
        while text[end_index] != "😉":
            end_index = random.randint(maxlen, len(text)-1)
            sentence = text[end_index-maxlen: end_index]
    return sentence
    
def gen_text(text):
    sentence = get_input(text, style='f')
    print("sentence is :", sentence)
    #start_index = random.randint(0, len(text) - maxlen - 1)
    #sentence = text[start_index: start_index + maxlen]
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        print("sentence gen :", sentence)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
        

def gen_tweet():
    start_index=5
    max_len = 340
    end_len = 2
    print()
    #print('----- Generating tweet after Epoch: %d' % epoch)

    while text[start_index] != "😉":
        #print("start index :",text[start_index])
        start_index = random.randint(0, len(text) - maxlen - 1)
        
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        # I'm going to have it use as the input seed my end tweet
        # char 😉 idea being that what's generated follows from that,
        # but, I can edit this easily
        # to end arbitrarily so that I can have it auto-complete
        # in trump style.
        
        while text[start_index+end_len] != "😉":
            #print("end char :", text[start_index+end_len])
            print("text :", text[start_index: start_index + end_len])
            end_len += 1
        
        
        sentence = text[start_index: start_index + end_len]
        
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(350):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            print(x_pred.shape())
            x_pred = x.pred.reshape(1, 40* len(chars))
            
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            if(next_char=='😉'):
                break

def gen_tweet_b(text, diversity = 0.5):
    sentence = ''
    end_index=2
    max_len = 40
    end_len = 2
    print()
    #print('----- Generating tweet after Epoch: %d' % epoch)

    #while text[end_index] != "😉":
        #print("start index :",text[start_index])
    #    end_index = random.randint(maxlen, len(text)-1)
    sentence = get_input(text, style = 'bt')
    #print("sentence is :", sentence)
    print(" ")
    
    generated = ''
        
    generated += sentence
    #print('----- Generating with seed: "' + sentence + '"')
    #sys.stdout.write(generated)
    #print("sentence gen :", sentence)

    for i in range(400):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
        if(next_char=='😉'):
            break

    """
    Old method of printing 4 different texts with different 'diversity'
    changed to above using one with a default input of 0.5
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        #print('----- diversity:', diversity)

        generated = ''
        
        generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        #print("sentence gen :", sentence)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            if(next_char=='😉'):
                break
    
    print("text[end_index] = :", text[end_index], "index = ", end_index)
    print("text indexed :", text[end_index-40: end_index])
    text_indexed = text[end_index-40: end_index]
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        # I'm going to have it use as the input seed my end tweet
        # char 😉 idea being that what's generated follows from that,
        # but, I can edit this easily
        # to end arbitrarily so that I can have it auto-complete
        # in trump style.

        while text[end_index-end_len] != "😉":
            #print("end char :", text[start_index+end_len])
            print("text :", text[end_index-end_len: end_index])
            end_len += 1
        
        sentence = text_indexed
        print("sentence :", sentence)
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        #print("indices_char :", indices_char)
        print("generated :", generated)
        

        for i in range(350):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            if i % 20 == 3:
                print("next index is :", next_index)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            if(next_char=='😉'):
                break
        
    """

                
def auto_full_tweet(start_tweet):
    max_len = 340
    print()
    #print('----- Generating tweet after Epoch: %d' % epoch)

    #while text[start_index] != "😉":
    #    start_index = random.randint(0, len(text) - maxlen - 1)
        
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        # this modified version of the above will complete a tweet
        # initially I will program this to the end, but maybe
        # change to word by word next so a user can streer the
        # tweet as it happens. Could do by letter. That seems
        # annoying. 
        
    #    while text[start_index:start_index+end_len] != "😉":
    #        end_len = random.randint(0, 350)
        sentence = start_tweet
        generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        for i in range(350):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            if(next_char=='😉'):
                break

def auto_word_tweet(start_tweet):
    max_len = 40
    print()
    #print('----- Generating tweet after Epoch: %d' % epoch)

    #while text[start_index] != "😉":
    #    start_index = random.randint(0, len(text) - maxlen - 1)
        
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        # this modified version of the above will complete a tweet
        # initially I will program this to the end, but maybe
        # change to word by word next so a user can streer the
        # tweet as it happens. Could do by letter. That seems
        # annoying. 
        
    #    while text[start_index:start_index+end_len] != "😉":
    #        end_len = random.randint(0, 350)
        sentence = start_tweet
        generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        for i in range(350):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            if(next_char == '😉' or ' '):
                break
                


# In[14]:


#print("gen text :")
#print(gen_text())

print("gen tweet :")
print(gen_tweet_b(text))

for x in range(0,10):
    print(" ")
    print("gen_tweet :", gen_tweet_b(text))
    print(" ")
#amtc("Hillary Clinton")
#print('gen_text :', gen_text())


# In[6]:


#print(text[5: i + maxlen])

print('gen_text :', gen_text(text))


# In[7]:


import sentiment_test as st

def amtc(tweet, mood_select = 'neg', variance=.3): #auto_mood_tweet_complete
        post = auto_word_tweet(tweet)
        post = str(post)
        mood = st.analyze_text(post)
        while(mood[mood_select] < variance):
            #print("mood :", mood[mood_select], "variance ", variance)
            post = auto_full_tweet(tweet)
            mood = st.analyze_text(post)
        print("Post :", post)
        return tweet


# In[ ]:





# In[8]:



""" This was my modified function so I could just generate without
training. It saves the output to a file"""

def gentext():
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            with open('garland_to_2.txt', 'a', encoding='utf-8') as file:
                file.write(next_char)
            sys.stdout.write(next_char)
            sys.stdout.flush()
        #model.save_weights('anthro.h5')


# In[ ]:




