1.Article A: A Unit Selection Methodology for Music Generation Using Deep NN
2.Article B: Deep Learning for Music
3.Article C: Music Generation by Deep Learning-Challenges and Directions
4.Article D: Music Generation with Markov Models
5.Article E: Music transcription modelling and composition using DL
6.Article F: Text-based LSTM networks for Automatic Music Composition

# Objective

- Article A: pass a musical Turing test/A subjective listening test was performed.
- Article B: find a meaningful way to represent notes in music as a vector/build interesting generative neural network architectures that effectively express the notions of harmony and melody
- Article E: create music transcription models that facilitate music composition, both within and outside particular conventions.
- Article F: the automatic generation of jazz chord progressions and rock music drum tracks.

# Representation

- Article A: unit/BOW like feature
- Article B: a "piano-roll" representation of midi files
- Article E: single character/continuous text file & transcription tokens/single complete transcriptions
- Article F: jazz chord/we used a binary representation of pitches


# Methods/Models

- Article A: autoencoder/LSTM
- Article B: LSTM-RNN
- Article E: RNNs and LSTM
- Article F: RNNs and LSTM


# Drawbacks/Challenges

- Article A: restricted to what is available in the unit library/the concatenation process may lead to "jumps" or "shifts"--sound unnatural and jarring
- Article B: incorporating the notion of musical aesthetic.
- Article E: space limitations
- Article F: One of the drawbacks of HMMs, however, is the ineciency of 1-of-K scheme of its hidden states/One drawback of word-based learning is the large number of states (or the size of vocabulary)


# Strategy

- Article A: multi-layer LSTM/note-level LSTM
- Article B: 2-layered LSTM-RNN/character level model
- Article E: char-rnn & folk-rnn & three hidden layers LSTM
- Article F: character- and word-based RNNs with LSTM units
