# Hierarchical_multilabel_classification

After git clone https://github.com/XiaonanHu/Hierarchical_multilabel_classification.git, please create a data folder inside this directory and put Train.csv, Dev.csv, patent_TITLE.csv, patent_ABSTR.csv and patent_description.csv in the folder

- Data loader:  
In get_data.py file within the src directory, you can run load_data_small function which get N number of patents.
It outputs two lists labeled_patent_data and unlabeled_patent_data (roughly 7:3). 

- Data format description:  
     labeled_patent_data = \[patent_1, patent_2, ...\]   
     patent = {ID : {'title':..., 'abstract':..., 'description':..., 'labels':\[...\]}}  
     (unlabeled patents does not have 'labels')



