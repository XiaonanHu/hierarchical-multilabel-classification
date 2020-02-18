from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np

def extract_features(patent_data, extractor):
    embedding = {}
    if extractor == "glove":
        # load model. It might take a while.
        glove_model = KeyedVectors.load_word2vec_format("../pretrained_model/glove/glove.twitter.27B/glove.twitter.27B.25d.word2vec.txt", binary=False)

        for patent_index in tqdm(patent_data.keys()):
            title = patent_data[patent_index]['title']
            abstract = patent_data[patent_index]['abstract']
            description = patent_data[patent_index]['description']
            # merge all titles
            #title = title[0] + ' ' + title[1] + ' ' + title[2]
            title = title[1]
            
            # get embeddings
            title_embedding = []
            abstract_embedding = []
            description_embedding = []
            for word in title.split(' '):
                if not word in glove_model.wv.vocab:
                    continue
                title_embedding.append(glove_model[word])
            for word in abstract.split(' '):
                if not word in glove_model.wv.vocab:
                    continue
                abstract_embedding.append(glove_model[word])
            for word in description.split(' '):
                if not word in glove_model.wv.vocab:
                    continue
                description_embedding.append(glove_model[word])

            title_embedding = np.array(title_embedding)
            abstract_embedding = np.array(abstract_embedding)
            description_embedding = np.array(description_embedding)

            embedding[patent_index] = {'title': title_embedding, 'abstract': abstract_embedding, 'description': description_embedding}

        return embedding

if __name__ == '__main__':
    from get_data import load_data_small
    import os
    labeled_patent_data, unlabeled_patent_data = load_data_small(100)
    print("number of patent:", len(labeled_patent_data))
    features = extract_features(labeled_patent_data, extractor = "glove")

    os.system("pause")