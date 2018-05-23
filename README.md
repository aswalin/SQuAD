# SQuAD
Building the QA system for Stanford Question Answering Datatset (https://rajpurkar.github.io/SQuAD-explorer/)

Please read this blog for details: https://towardsdatascience.com/building-a-question-answering-system-part-1-9388aadff507

The first file create_emb.ipynb takes care of creating a dictionary of sentence embedding for all the sentences and questions in the wikipedia articles of training dataset

The second file unsupervised.ipynb calculates the distance between sentence & questions basis Euclidean & Cosine similarity using sentence embeddings. It finally extracts the setence from each paragraph that has the minimum distance from the question. Currently, they are giving an accuracy of 45% & 63% respectively.

The last file treats this problem as supervised learning problem where I am fitting multinomial logistic regression, random forest and xgboost and create 20 features - (2 features represnts the cosine distance & euclidean for one sentence. I am limiting each para to 10 sentences). The target variable is the sentence ID having the correct answer. So I have 10 labels. This is currently giving an accuracy of 63%, 65% & 69% respectively.

Future Work: Use RNNs to get the exact answer
