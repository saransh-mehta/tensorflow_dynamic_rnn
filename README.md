# tensorflow_dynamic_rnn
Implementation of dynamic rnn and vector embeddings to detect the intent (context) of the sentence
In this project, we have certain sentences in our dataset which are classified into certain categories.
Hence, we built a tensorflow dynamic Rnn, which creates Long-Short Term Memory network according to the size of the sentences, i.e, dynamic sized. This gives much more accurate and relevant results than padding every sentence to a static size. Further, tensorflow's embeddings lookup method has been used to project the words in vocabulary in 64- Dimensional space. 
