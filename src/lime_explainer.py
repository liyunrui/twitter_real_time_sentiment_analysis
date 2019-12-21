import argparse
import spacy
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from config import Configs


METHODS = {
    'transformer': {
        'class': "TransformerExplainer",
        'file': "models/transformer"
    },
    'BERT': {
        'class': "BertExplainer",
        'file': "models/transformer"
    }
}

class BertExplainer:
    """Class to explain classification results of the Bert fine-tune model.
       Assumes that we already have a trained Bert model with which to make predictions.
       
       Notice:
       	Our goal is to see how the words that affect the classifier the most.
    """
    def __init__(self):
        from infer import Inferer

        self.inf = Inferer(Configs.opt)

    def predict(self, texts):
        """
        texts: list of n-gram of sentence, list of str

        Return lisft softmax probabilities, list of list of flaot
        """
        t_probs = self.inf.evaluate(
            texts,
            input_cols = Configs.opt.inputs_cols
            )
        return t_probs

def tokenizer(text):
    "Tokenize input string using a spaCy pipeline"
    nlp = spacy.blank('en')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))  # Very basic NLP pipeline in spaCy
    doc = nlp(text)
    tokenized_text = ' '.join(token.text for token in doc)
    return tokenized_text

def explainer_class(method):
    "Instantiate class using its string name"
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_

def explainer(method: str,
              text: str,
              num_samples: int,
              num_classes: int,
              ):
    """Run LIME explainer on provided classifier"""

    #model = explainer_class(method)
    model = BertExplainer()
    # def predictor(texts):
    #     print ("input of predictor", len(texts)) # list of n-gram of sentence, list of str
    #     probs = []  # Process each text and get softmax probabilities
    #     for text in tqdm(texts):
    #         probs.append([0.2, 0.5, 0.3])
    #     return np.array(probs)
    predictor = model.predict
    #print (predictor)
    # Create a LimeTextExplainer
    explainer = LimeTextExplainer(
        # Specify split option
        split_expression=lambda x: x.split(),
        # Our classifer uses bigrams or contextual ordering to classify text
        # Hence, order matters, and we cannot use bag of words.
        bow=False,
        # Specify class names for this case
        #class_names=[1, 2, 3, 4, 5]
        class_names = num_classes
    )

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        text,
        classifier_fn=predictor,
        top_labels=1,
        num_features=20,
        num_samples=num_samples,
    )
    return exp

def main(samples):
    # Get list of available methods:
    method_list = [method for method in METHODS.keys()]
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, nargs='+', help="Enter one or more methods \
                        (Choose from following: {})".format(", ".join(method_list)),
                        required=True)
    parser.add_argument('-n', '--num_samples', type=int, help="Number of samples for explainer \
                        instance", default=500)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=140, type=int)
    parser.add_argument('--num_classes', default=["negative","neutral","positive"], type=list)

    args = parser.parse_args()

    for method in args.method:
        if method not in METHODS.keys():
            parser.error("Please choose from the below existing methods! \n{}".format(", ".join(method_list)))
        #path_to_file = METHODS[method]['file']
        # Run explainer function
        print("Method: {}".format(method.upper()))
        for i, text in enumerate(samples):
            #text = tokenizer(text)  # Tokenize text using spaCy before explaining
            print("Generating LIME explanation for example {}: {} ".format(i+1, text))
            exp = explainer(method, text, args.num_samples, args.num_classes)
            # list of tuple, words and his weight effecting the model
            print ("Extract keywords effecting the emotion of the sentence : {}", exp.as_list(label = 0))
            # Output to HTML
            output_filename = Path(__file__).parent / "{}-explanation-{}.html".format(i+1, method)
            print (output_filename)
            exp.save_to_file(output_filename)

if __name__ == "__main__":
    # Evaluation text
    samples = [
        "It's not horrible, just horribly mediocre."
        # "The cast is uniformly excellent ... but the film itself is merely mildly charming.",
    ]
    main(samples)