# Custom NER Analysis Spacy 

_This project demonstrates how to build custom NER model in spacy._

 ##### Python library:

- spacy-streamlit
- spacy


##### Installation steps:

1. Clone the repository
2. Create virtual enviornment

> virtualenv -p python3.6 venv_spacy_ner
>
> .venv_spacy_ner/bin/activate

3. Install dependencies/packages:

> pip install -r requirements.txt

4. Run the following command:

> 1 convert tsv file to json format.   
- python tsv_convertor.py

> 2 convert json file to spacy file format.
- python json_convertor.py

> 3. Train the model
- python ner_train.py 
