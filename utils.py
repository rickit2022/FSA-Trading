import pandas as pd
import numpy as np
import psutil
import os

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree


class InputFeatures(object):
    """
    A single set of features for the data.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.agree = agree

def chunks(l, n):
    """
    Simple utility function to split a list into fixed-length chunks.
    Parameters
    ----------
    l: list
        given list
    n: int
        length of the sequence
    """
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode='classification'):
    """
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.

    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokenizer: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'

    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                          len(tokens) - (3 * max_seq_length // 4) + 1:]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding


        token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if mode == 'classification':
            label_id = label_map[example.label]
        elif mode == 'regression':
            label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

        agree = example.agree
        mapagree = {'0.5': 1, '0.66': 2, '0.75': 3, '1.0': 4}
        try:
            agree = mapagree[agree]
        except:
            agree = 0

        if ex_index < 1:
            pass
            # logger.info("*** Example ***")
            # logger.info("guid: %s" % (example.guid))
            # logger.info("tokens: %s" % " ".join(
            #     [str(x) for x in tokens]))
            # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            # logger.info(
            #     "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          agree=agree))
    return features

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]

#------------------------------------------------------------------------------------------------------------------------------#

def loadFrame(input_path, format, keys = None, chunksize = 1000):
    filepath = input_path
    if keys:
        for key in keys:
            filepath = filepath + "_" + key

    if os.path.exists(filepath + "." + format) == True:
        filepath = filepath + "." + format
    
    chunk= False
    if format == "csv":
        if os.path.getsize(filepath) / (1024 ** 3) > 1.0:
            df = pd.read_csv(filepath, encoding = 'utf-8', chunksize=chunksize)
            chunk = True
        else:
            df = pd.read_csv(filepath, encoding = 'utf-8')

    elif format == "json":
        if os.path.getsize(filepath) / (1024 ** 3) > 1.0:
            df = pd.read_json(filepath, lines=True, chunksize=chunksize)
            chunk = True
        else:
            df = pd.read_json(filepath, lines=True)
    return df,chunk

def saveFrame(df, output_path,output_format = "csv", keys= None,index = False, writerheader = True, overwrite = False, full_path = False):
    filepath = output_path
    mode = 'w'

    if overwrite == False:
        #if file exists and overwrite flag not set, only appends to eof, otherwise overwrites
        mode = 'a'
    
    if not os.path.exists(filepath):
        if full_path:
            #if full path is specified i.e., data/history/new.csv, parse just the directory 
            dir_path = os.path.dirname(filepath)
        else:
            dir_path = filepath
        
        os.makedirs(dir_path, exist_ok=True)

    if keys:
        #if keys are specified, save the file with the keys as the filename
        keystring = f"{'_'.join(keys)}.{output_format}"
        filepath = filepath + "/" + keystring
    else:
        if not full_path:
            filepath = filepath + "." + output_format
    
    if output_format == "csv":
        df.to_csv(filepath, mode=mode, header=writerheader,index = index, encoding='utf-8')
    elif output_format == "xlsx":
        df.to_excel(filepath, index=index) 

def organiseFrame(df):
    pass

def setCores(noCores):
    """
    Use to set the number of cores to be used by the process/program, prevents 100% cpu utilisation. 
    Useful if you run on a potato pc like me.
    """
    cores = []
    for i in range(0, noCores):
        cores.append(i)
    
    psutil.Process(os.getpid()).cpu_affinity(cores)

def clean(dirs):
    for dir in  dirs:
        for root, dirs, files in os.walk(dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))