def build_DF(N,processed_text,processed_bookname):

    DF = {}

    for i in range(N):
        tokens = processed_text[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

        tokens = processed_bookname[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
    for i in DF:
        DF[i] = len(DF[i])
    
    total_vocab_size = len(DF)
    total_vocab = [x for x in DF]
    
    return DF, total_vocab_size, total_vocab