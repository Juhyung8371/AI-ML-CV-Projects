# Movie Review Sentiment Analysis With BERT

## Introduction

This project aims to learn how to classify movie reviews based on their sentiments from the text using the BERT model. 

From this exercise, I learned:
1. How to create training examples and targets for text generation.
2. What RNN is and how to build an RNN model for sequence generation using Keras.
3. How to create a text generator and evaluate the output.

The tutorial I followed is available at the [TensorFlow’s website](https://www.tensorflow.org/text/tutorials/text_generation)

## What is the BERT?

<img src="bert_model.png" height="350">

[[Image source]](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

Introduced in 2018 by Google, the Bidirectional Encoder Representations from Transformers (BERT) model is a pre-trained natural language processing (NLP) model based on transformer architecture. 

Here are some key concepts:

1. Bidirectional Context:

   Unlike earlier models that processed text in one direction (left-to-right or right-to-left), BERT considers both directions. This bidirectional context allows the model to understand the meaning of a word based on its surrounding words from both sides.

2. Transformer Architecture:

   BERT utilizes the transformer architecture, which employs self-attention mechanisms. It enables the model to efficiently capture relationships and dependencies between words by weighing the relative importance of words against each other. The key advantage of the transformer against RNNs, a traditional NLP model choice, is their ability to capture longer dependencies, thanks to the residual connection that keeps the learning gradient from vanishing. 
  
3. Pre-training: 

    BERT is pre-trained on a large corpus of text data (Wikipedia (~2.5B words) and Google’s BooksCorpus (~800M words))using the following techniques: 
    Masked Language Model (MLM):
    * The model learns how to fill in the blank by using the context in the sequence, like words and their positions. To be more specific, 15% of the words in a sequence are replaced with a [MASK] token.
    Next Sentence Prediction (NSP)
    * The model learns the relationship between sentences by classifying whether one sentence can follow another.

    The large amount of training data combined with the unsupervised pre-training helps BERT grasp the nuances and complexities of language. Various-sized BERT models are available for platforms with different computing power.

4. Fine-tuning:

    The pre-trained BERT model can be fine-tuned for more specific tasks like sentiment analyzer, spam detector, chatbot, etc. 

Check [this Huggin Face article](https://huggingface.co/blog/bert-101) and [this Medium article](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) about the BERT model as well. 

## Data and Model

I used the 50k movie reviews from the Internet Movie Database. This is divided into a 25k train set, a 25k test set, and a 5k validation set. Each data consists of a review (string) and label (0 for negative and 1 for positive). 

The pre-trained BERT model and its data preprocessor are loaded from the TensorFlow hub. For speed matter, I used a small BERT model with 4 layers, 512 hidden units, and 8 attention units.

<img src="model_architecture.png" height="350">

I used the Adam optimizer with a 3e-5 learning rate and trained the model for 10 epochs. 


## Result

<img src="training_validation.png" height="350">

The training is evaluated on BinaryCrossentropy and BinaryAccuracy since the task is a classification problem. As shown in the results, the model shows signs of overfitting after the second epoch - the training score increases, but the validation score decreases. 





