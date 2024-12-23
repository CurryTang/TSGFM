import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)
from torchmetrics import AveragePrecision, AUROC
import numpy as np
from torch_geometric.utils import (to_scipy_sparse_matrix, scatter, )
from tqdm.autonotebook import trange
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
# from llm2vec import LLM2Vec
ENCODER_DIM_DICT = {"ST": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120, 'minilm':384, 'random': 128, 'tfidf': 300, "e5_mistral": 4096, "l2v": 4096, "mpnet": 768}

def generate_tfidf_pytorch(texts, hidden_dimension=300):
    """
    Generates TF-IDF vectors with a specified hidden dimension and returns a PyTorch tensor.

    Args:
        texts (list): A list of text documents.
        hidden_dimension (int, optional): The desired dimensionality of the TF-IDF vectors. 
                                          Defaults to 2.

    Returns:
        torch.Tensor: A PyTorch tensor containing the TF-IDF vectors.
    """

    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=hidden_dimension)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # Convert to PyTorch tensor
    tfidf_tensor = torch.tensor(tfidf_reduced, dtype=torch.float32)

    return tfidf_tensor


class SentenceEncoder:
    def __init__(self, name, root="cache_data/model", batch_size=1, multi_gpu=False):
        self.name = name
        self.root = root
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = None
        self.tokenizer = None

    def get_model(self):
        if self.name == "ST":
            self.model = SentenceTransformer("multi-qa-distilbert-cos-v1", device=self.device, cache_folder=self.root, )
            self.encode = self.ST_encode
        
        elif self.name == 'minilm':
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device, cache_folder=self.root)
            self.encode = self.ST_encode
        
        elif self.name == 'mpnet':
            self.model = SentenceTransformer("all-mpnet-base-v2", device=self.device, cache_folder=self.root)
            self.encode = self.ST_encode

        elif self.name == "llama2_7b":
            # model_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'llama-2-7b')
            model_name = "meta-llama/Llama-2-7b-hf"
            model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=self.root)
            self.model = model.to(self.device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=self.root)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = 'right'
            self.tokenizer = tokenizer
            self.encode = self.llama_encode

        elif self.name == "llama2_13b":
            # model_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'llama-2-7b')
            model_name = "meta-llama/Llama-2-13b-hf"
            model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=self.root)
            self.model = model.to(self.device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=self.root)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = 'right'
            self.tokenizer = tokenizer
            self.encode = self.llama_encode

        elif self.name == "e5":
            model_name = "intfloat/e5-large-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.root)
            model = AutoModel.from_pretrained(model_name, cache_dir=self.root)
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.encode = self.e5_encode
        
        elif self.name == 'e5_mistral':
            self.model = SentenceTransformer("intfloat/e5-mistral-7b-instruct", device=self.device, cache_folder=self.root)
            self.model.max_seq_length = 4096
            self.encode = self.prompt_ST_encode
        
        # elif self.name == 'l2v':
        #     # Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", cache_dir=self.root
        #     )
        #     config = AutoConfig.from_pretrained(
        #         "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True, cache_dir=self.root
        #     )
        #     model = AutoModel.from_pretrained(
        #         "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        #         trust_remote_code=True,
        #         config=config,
        #         torch_dtype=torch.bfloat16,
        #         device_map="cuda" if torch.cuda.is_available() else "cpu",
        #         cache_dir=self.root
        #     )
        #     model = PeftModel.from_pretrained(
        #         model,
        #         "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        #         cache_dir=self.root
        #     )
        #     model = model.merge_and_unload()  # This can take several minutes on cpu

        #     # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
        #     model = PeftModel.from_pretrained(
        #         model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
        #         cache_dir=self.root
        #     )

        #     # Wrapper for encoding and pooling operations
        #     l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        #     self.model = l2v
        #     self.encode = self.l2v_encode
        elif self.name == "roberta":
            self.model = SentenceTransformer("sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                device=self.device, cache_folder=self.root, )
            self.encode = self.ST_encode
        elif self.name == "tfidf":
            self.encode = self.tfidf_encode
        elif self.name == "random":
            self.encode = self.random_encode
        else:
            raise ValueError(f"Unknown language model: {self.name}.")

    def encode(self, texts, to_tensor=True):
        raise NotImplementedError("Not define llm encoder yet.")

    def ST_encode(self, texts, to_tensor=True):
        if self.multi_gpu:
            # Start the multi-process pool on all available CUDA devices
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=self.batch_size, )
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                convert_to_tensor=to_tensor, convert_to_numpy=not to_tensor, )
        return embeddings
    
    def prompt_ST_encode(self, texts, to_tensor=True):
        if self.multi_gpu:
            # Start the multi-process pool on all available CUDA devices
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=self.batch_size, )
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                convert_to_tensor=to_tensor, convert_to_numpy=not to_tensor, prompt = "encode the text attributes of this graph node")
        return embeddings
    
    def random_encode(self, texts, to_tensor=True):
        embeddings = torch.normal(0, 1, size=(len(texts), ENCODER_DIM_DICT['random']))
        return embeddings
    
    def tfidf_encode(self, texts, to_tensor=True):
        embeddings = generate_tfidf_pytorch(texts, hidden_dimension=ENCODER_DIM_DICT['tfidf'])
        return F.normalize(embeddings)

    def l2v_encode(self, texts, to_tensor=True):
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                if not isinstance(sentences_batch, list):
                    sentences_batch = sentences_batch.tolist()
                d_reps = self.model.encode(sentences_batch)
                all_embeddings.append(d_reps)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def llama_encode(self, texts, to_tensor=True):

        # Add EOS token for padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                if not isinstance(sentences_batch, list):
                    sentences_batch = sentences_batch.tolist()
                input_ids = self.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).input_ids.to(self.device)
                transformer_output = self.model(input_ids, return_dict=True, output_hidden_states=True)["hidden_states"]
                # No gradients on word_embeddings
                word_embeddings = transformer_output[-1].detach()
                sentence_embeddings = word_embeddings.mean(dim=1)
                all_embeddings.append(sentence_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def e5_encode(self, texts, to_tensor=True):
        def average_pool(last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                if not isinstance(sentences_batch, list):
                    sentences_batch = sentences_batch.tolist()
                batch_dict = self.tokenizer(sentences_batch, padding="longest", truncation=True, return_tensors='pt')
                for item, value in batch_dict.items():
                    batch_dict[item] = value.to(self.device)
                outputs = self.model(**batch_dict)
                embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = embeddings.detach()
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(all_embeddings.size())
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        if self.tokenizer is not None:
            self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()


def binary_single_auc_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    # if len(score.unique()) == 1:
    # print(output[:20])
    label = batch.bin_labels[batch.true_nodes_mask]
    # print(score)
    # print(label)
    return func.update(score, label.view(-1, batch.num_classes[0]))


def flat_auc(func, output, batch):
    return func(torch.sigmoid(output).view(-1), batch.bin_labels[batch.true_nodes_mask].view(-1))


def binary_apr_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(len(batch), -1))


def binary_auc_multi_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(-1, batch.num_classes[0]))


def label_apr_func(func, output, batch):
    score = torch.sigmoid(output)
    return func.update(score, batch.y)


def flat_label_func(func, output, batch):
    labels = batch.y.view(-1)
    valid_ind = labels == labels
    return func(output.view(-1)[valid_ind], labels[valid_ind])


def classification_single_func(func, output, batch):
    label = batch.bin_labels[batch.true_nodes_mask].view(-1, batch.num_classes[0])
    output = output.view(-1, batch.num_classes[0])
    return func(output, torch.argmax(label, dim=-1))


class MultiApr(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AveragePrecision(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


class MultiAuc(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AUROC(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


def scipy_rwpe(data, walk_length):
    row, col = data.edge_index
    N = data.num_nodes

    value = data.edge_weight
    if value is None:
        value = torch.ones(data.num_edges, device=row.device)
    value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
    value = 1.0 / value
    adj = to_scipy_sparse_matrix(data.edge_index, edge_attr=value, num_nodes=data.num_nodes)

    out = adj
    pe_list = [out.diagonal()]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(out.diagonal())
    pe = torch.tensor(np.stack(pe_list, axis=-1))

    return pe


def get_available_devices():
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_label_texts(labels):
    label_texts = [None] * int(len(labels) * 2)
    for entry in labels:
        label_texts[labels[entry][0]] = (
                "prompt node. molecule property description. " + "The molecule is effective to the following assay. " +
                labels[entry][1][0][:-41])
        label_texts[labels[entry][0] + len(labels)] = (
                "prompt node. molecule property description. " + "The molecule is not effective to the following "
                                                                 "assay. " +
                labels[entry][1][0][:-41])
    return label_texts


def set_mask(data, name, index, dtype=torch.bool):
    mask = torch.zeros(data.num_nodes, dtype=dtype)
    mask[index] = True
    setattr(data, name, mask)


def get_mask(data, i = 0):
    """
        Given different types of mask format, return the first seed
    """
    if hasattr(data, 'train_mask'):
        if isinstance(data.train_mask, torch.Tensor):
            return data.train_mask, data.val_mask, data.test_mask
        else:
            if i < len(data.train_mask):
                return data.train_mask[i], data.val_mask[i], data.test_mask[i]
            else:
                return data.train_mask[0], data.val_mask[0], data.test_mask[0]
    elif hasattr(data, 'train_masks'):
        if i < len(data.train_masks):
            return data.train_masks[i], data.val_masks[i], data.test_masks[i]
        else:
            return data.train_masks[0], data.val_masks[0], data.test_masks[0]

    