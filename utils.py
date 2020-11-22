import pickle as pkl
import numpy as np
import json

import pandas as pd
import nltk
from konlpy.tag import Okt

okt = Okt()


def tokenize(doc):
    """
    input:
        doc: dtype=string, example) '아 더빙.. 진짜 짜증나네요 목소리'
    return:
        list: dtype=string, elements example) ['아/Exclamation', '더빙/Noun', '../Punctuation', '진짜/Noun', '짜증나다/Adjective', '목소리/Noun']
    """
    return ["/".join(token) for token in okt.pos(doc, norm=True, stem=True)]


def saveDocs(docs, filePath):
    # If docs has null value, exchange it to ''.
    if docs["document"].isnull().any():
        docs["document"] = docs["document"].fillna("")

    reviewDocs = [(tokenize(row[1]), row[2]) for row in docs.values]

    with open(filePath, "wb") as f:
        pkl.dump(reviewDocs, f)
        print("[%s] is saved." % (filePath))


def loadDocs(filePath):
    with open(filePath, "rb") as f:
        reviewDocs = pkl.load(f)
        return reviewDocs


if __name__ == "__main__":
    train_df = pd.read_csv("nsmc-master/ratings_train.txt", "\t")
    test_df = pd.read_csv("nsmc-master/ratings_test.txt", "\t")

    # saveDocs(train_df, "./trainDocs.pkl")
    # saveDocs(test_df, "./testDocs.pkl")

    trainDocs = loadDocs("./trainDocs.pkl")
    testDocs = loadDocs("./testDocs.pkl")

    print("print: trainDocs[0:10]")
    for idx in range(10):
        print(trainDocs[idx])

    print("print: testDocs[0:10]")
    for idx in range(10):
        print(testDocs[idx])

    # get all tokens in a list.
    trainTokens = [token for doc in trainDocs for token in doc[0]]

    text = nltk.Text(trainTokens, name="NMSC")

    # print number of all tokens.
    print("number of all tokens :", len(text.tokens))

    # print number of tokens excluding duplicates. (convert to tuple dtype)
    print("number of tokens excluding duplicates :", len(set(text.tokens)))

    # get TOP 10000 tokens with the highest frequency of output.
    topN = 10000
    print("get top %d words ..." % (topN), end=" ")
    topN_words = {token[0]: token[1] for token in text.vocab().most_common(topN)}
    print("done.")
    print("top %d words : " % (topN), topN_words)
    with open("top_%d_words.json" % (topN), "w", encoding="utf-8") as jsonF:
        json.dump(topN_words, jsonF, ensure_ascii=False)
