import matplotlib.pyplot as plt
import seaborn as sns 
import random
import pandas as pd
import warnings
import re 

from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class DataExplorer:
    def __init__(self, colors):
        self.colors = colors

    def subject_count(self, dataframe, n_most_common):
        print("-----EMAIL ARCHIVE SUBJECT INFORMATION-----", end='\n\n')
        subject_cntr = Counter(dataframe["Subject"].tolist())
        print(f"Total subjects: {len(subject_cntr)}", end='\n\n')

        print(f'{n_most_common} most common subjects:')
        for subject, count in subject_cntr.most_common(n_most_common):
            print(f"Subject: {subject}  |  count: {count}")

        print()
    

    # TEXT COLLECTION SHOULD PROBABLY BE RENAMED TO WORD COLLECTION, OR SOMETHING MORE FITTING
    def wg_most_commonly_used_words(self, text_collection, wg, n_tokens=20):
        tokens = text_collection[wg]

        most_common_tokens, counts = [], []

        cntr = Counter(tokens).most_common(n_tokens)

        for token, count in cntr:
            most_common_tokens.append(token)
            counts.append(count)

        plt.figure(figsize=(18, 5))
        plt.bar(most_common_tokens, 
                counts, 
                color=random.choices(self.colors, k=n_tokens))
        
        plt.label(f"Top {n_tokens} most commonly used words in {wg.upper()}")
        plt.savefig(f'plots/{wg}_{n_tokens}_most_common_words')


    def wg_wordcloud(self, text_collection, wg, max_words):
        tokens = ' '.join(text_collection[wg])
        wordcloud = WordCloud(max_font_size=50,
                              background_color='white',
                              max_words=max_words).generate(tokens)

        plt.title(f'{max_words}-word WordCloud for {wg.upper()} WG')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f'wordclouds/{wg}_{max_words}_words')

    
    def wg_body_len_dist(self, text_collection, wg=None, kde=False, bins=45, whole=False, limit=10000):
        warnings.filterwarnings("ignore")

        if whole:
            body_lengths = [len(body) 
                            for body_coll in text_collection.values()
                            for body in body_coll]
        else:
            body_lengths = [len(body) for body in text_collection[wg]]

        ax = sns.histplot(body_lengths,
                          kde=kde,
                          bins=bins,
                          edgecolor="black", 
                          linewidth=1)
        
        for i, patch in enumerate(ax.patches):
            ax.patches[i].set_facecolor(random.choices(self.colors, k=1)[0])

        
        ax.set(xlim=(1, limit))
        
        if whole:
            ax.set(title=f'Body lengths in whole collection')
            plt.savefig(f'distributions/whole_body_len_dist')

        else:
            ax.set(title=f'Body lengths in "{wg.upper()}" WG')
            plt.savefig(f'distributions/{wg}_body_len_dist')

    # I THINK THAT THE WAY THE WG ARRAY IS HANDLED IS WRONG
    # WHAT IF WGS PASSED IN ARE UNORDERED?
    def ngram_vectorizer(self, text_collection, vectorizer_type='count', wgs=[], top_n=20, ngram_range=(1,1), min_df=2, sent_lex=False):
        if sent_lex: 
            str_documents = [' '.join(word for word in doc) for doc in text_collection][:len(wgs)]
        else:
            documents = list(text_collection.values())[:len(wgs)]
            str_documents = [' '.join(word for word in doc) for doc in documents]

        if vectorizer_type == 'tf_idf':
            vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(ngram_range=ngram_range)

        vectors = vectorizer.fit_transform(str_documents)
        
        tdm = pd.DataFrame(vectors.todense().round(3))
        
        tdm.columns = vectorizer.get_feature_names_out()
        tdm = tdm.T
        tdm.columns = wgs

        if vectorizer_type == 'tf_idf':
            tdm['highest_score'] = tdm.max(axis=1)
            tdm = tdm.sort_values(by='highest_score', ascending=False)

        elif vectorizer_type == 'count':
            tdm['total_count'] = tdm.sum(axis=1)
            tdm = tdm.sort_values(by='total_count', ascending=False)
        
        print(tdm.head(top_n))
        

    def sent_lex_vectorizer(self, text_collection, wgs=[], lexicon_path='vader_lexicon.txt', pos_thres=2, neg_thresh=-2):
        sentiment_lexicon = {}
         
        with open(lexicon_path, 'r') as lexicon:
            lexicon_lines = [re.split('\t', line.strip()) for line in lexicon]
    
        for token, score, _, _ in lexicon_lines: 
            if float(score) > pos_thres:
                sentiment_lexicon[token] = 'pos'
            
            elif float(score) < neg_thresh: 
                sentiment_lexicon[token] = 'neg'

        sent_lex_docs = []
        for doc in text_collection.values():
            transformed_doc = []
            for word in doc:
                if word in sentiment_lexicon.keys():
                    transformed_doc.append(word + '_' + sentiment_lexicon[word])
            sent_lex_docs.append(transformed_doc)
        
        self.ngram_vectorizer(text_collection=sent_lex_docs,
                              vectorizer_type='count',
                              wgs=wgs,
                              ngram_range=(1,1),
                              min_df=1,
                              sent_lex=True)

    
    def keyword_concordance(self, text_collection, wgs, keywords, left_context, right_context, num_samples):
        for wg, bodies in text_collection.items():
            for desired_wg in wgs:
                num_samples_extracted = 0
                samples = []
                if wg == desired_wg:
                    for keyword in keywords:
                        for body in bodies:
                            body = body.split()
                            
                            for i, word in enumerate(body):
                                if keyword == word:
                                    if i >= left_context and (i + right_context) < len(body):
                                        left = body[i - left_context:i]
                                        rigt = body[i+1:i+right_context]  # might be wrong
                                        whole = left + word + right
                                        num_samples_extracted += 1
                                        samples.append(whole)
                                        if num_samples_extracted == num_samples:
                                            return samples







        
