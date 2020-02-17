# Hierarchical Multilabel Classification

After git clone https://github.com/XiaonanHu/Hierarchical_multilabel_classification.git, please create a data folder inside this directory and put Train.csv, Dev.csv, patent_TITLE.csv, patent_ABSTR.csv and patent_description.csv in the folder. 

- Data loader:  
	In get_data.py file within the src directory, you can run load_data_small function which get N number of patents. It outputs two lists labeled_patent_data and unlabeled_patent_data (roughly 7:3). 

- Data format description:  
     labeled_patent_data = {patent1_ID: patent1_value, patent2_ID: patent2_value, ...}    
     
     patent
     - key: patent_ID  
     - value : {'title':..., 'abstract':..., 'description':..., 'labels':\[label1, label2, ...\]}  
     (unlabeled patents do not have 'labels')

## extract embedding
**extract_features.py** is provided to extract embeddings for the patent.  

### Glove
Now the pretrained GloVe model [1] is used to extract embeddings from the title, abstract and description.

1. Please download **glove.twitter.27B.zip** [here](https://nlp.stanford.edu/projects/glove/), unzip the file, and put the **glove.twitter.27B** folder into the **pretrained_model/glove** folder. 
2. Run **pretrained_model/glove/glove2word2vec.py**.
3. **extract_features.py** is provided to extract embeddings for the patent.  
	*  Since only the pretrained model is used now, some words are not in the dictionary, and these words are simply ignored for now.
	* output format:  
		feature  
     		- key: patent_ID    
     		- value : {'title':..., 'abstract':..., 'description':..., 'labels':\[label1, label2, ...\]} 
	* e.g. `feature = extract_features(unlabeled_patent_data, extractor = "glove")`  
	`feature['1234567']['title'].shape` is (10, 25), where 10 is the number of words, and 25 is the length of embedding for each word.
	
	
Reference:  
[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf)
