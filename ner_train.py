from __future__ import unicode_literals, print_function
import pickle
import random
from pathlib import Path
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding


def test_custom_ner_model(nlp):
    """
    This method test model.
    """
    test_text = 'India is the great country!'
    doc = nlp(test_text)
    print("\n\n NER FOR -->  '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)
    return test_text


def save_custom_model(nlp, output_dir, new_model_name ):
    """
    This method saves the custom model.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("\n\n Saved model to", output_dir)


def load_saved_model(output_dir, test_text):
    """
    This method load and test the saved model.
    """
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    for ent in doc2.ents:
        print(f'\n {ent.text} --> {ent.label_}')


# New entity labels
LABEL = ['I-geo', 'B-geo', 'I-art', 'B-art', 'B-tim', 'B-nat', 'B-eve', 'O', 'I-per', 'I-tim', 'I-nat', 'I-eve', 'B-per', 'I-org', 'B-gpe', 'B-org', 'I-gpe']


def train_custom_ner_model(model=None, new_model_name='custom_ner_model', output_dir=None, n_iter=10, TRAIN_DATA=None):
    """
    Setting up the pipeline and entity recognizer, and training the new model.
    """
    if model is not None:
        nlp = spacy.load(model)  # load user defined spacy model
        print("\n Model Vocab Language --> '%s'" % model)
    else:
        nlp = spacy.blank('en')
        print("\n Created default 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # Disable other models
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                for text, annotation in batch:
                    example = Example.from_dict(nlp.make_doc(text), annotation)
                    nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)
            print('Losses', losses)

    test_sent = test_custom_ner_model(nlp)

    save_custom_model(nlp, output_dir, new_model_name)

    load_saved_model(output_dir, test_sent)


if __name__ == '__main__':

    with open('input/training_ds.spacy', 'rb') as fp:
        TRAIN_DATA = pickle.load(fp)

    train_custom_ner_model(output_dir='output/', TRAIN_DATA=TRAIN_DATA)