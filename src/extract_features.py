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

def get_patent_feature_vector(all_embeds):
    feature_vectors = {}
    for ID in all_embeds.keys():

        title = all_embeds[ID]['title']
        abstract = all_embeds[ID]['abstract']
        description = all_embeds[ID]['description']
        # title length set to 50
        if title.shape[0] < 50:
            title = np.vstack((title, np.zeros(50-title.shape[0], title.shape[1])))
        title = title[:50,:].flatten()
        title = title.reshape((1,len(title)))

        # abstract length set to 100
        if abstract.shape[0] < 100:
            abstract = np.vstack((abstract, np.zeros(100-abstract.shape[0], abstract.shape[1])))
        abstract = abtract[:100,:].flatten()
        abstract = abstract.reshape((1, len(abstract)))
        vec = np.hstack((vec, abstract))


        # description length set to 200
        if description.shape[0] < 200:
            description = np.vstack((description, np.zeros(200-description.shape[0], description.shape[1])))
        description = description[:200,:].flatten()
        description = description.reshape((1, len(description)))
        vec = np.hstack((vec, description))
        

        feature_vectors[ID] = vec


    return feature_vectors 




#if __name__ == '__main__':
from get_data import load_data_small
import os
labeled_patent_data, unlabeled_patent_data = load_data_small(1000)
print("number of patent:", len(labeled_patent_data))
features = extract_features(labeled_patent_data, extractor = "glove")
vectors = get_patent_feature_vector(features)
print('len(vectors)',len(vectors))
print('len(vectors[vectors.keys()[0])', len(vectors[vectors.keys()[0]]))

#os.system("pause")