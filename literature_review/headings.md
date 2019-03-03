Article A: A Unit Selection Methodology for Music Generation Using Deep NN
Article B: Deep Learning for Music
Article C: Music Generation by Deep Learning-Challenges and Directions
Article D: Music Generation with Markov Models
Article E: Music transcription modelling and composition using DL
Article F: Text-based LSTM networks for Automatic Music Composition

# Objective

Article A: pass a musical Turing test/A subjective listening test was performed.
Article B: find a meaningful way to represent notes in music as a vector/build interesting generative neural network architectures that effectively express the notions of harmony and melody

# Representation

Article A: unit/BOW like feature
Article B: a "piano-roll" representation of midi files


# Methods/Models

Article A: autoencoder/LSTM
Article B: LSTM-RNN


# Drawbacks/Challenges

Article A: restricted to what is available in the unit library/the concatenation process may lead to "jumps" or "shifts"--sound unnatural and jarring
Article B: incorporating the notion of musical aesthetic.


# Strategy

Article A: multi-layer LSTM/note-level LSTM
Article B: 2-layered LSTM-RNN/character level model
