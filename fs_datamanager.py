import json
import os.path
from abc import abstractmethod
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class SimpleFSManager:
    def __init__(self, class_ind, data_ind, k_shot, q_query, n_way, min_k_shot=None, min_n_way=None):
        self.class_ind = class_ind
        self.data_ind = data_ind
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_way = n_way
        self.min_n_way = min_n_way
        self.min_k_shot = min_k_shot

    def get_few_shot_idx(self):
        if self.min_n_way is not None:
            n_way = np.random.permutation(np.arange(self.min_n_way, self.n_way))[0]
        else:
            n_way = self.n_way
        if self.min_k_shot is not None:
            k_shot = np.random.permutation(np.arange(self.min_k_shot, self.k_shot))[0]
        else:
            k_shot = self.k_shot

        target_classes_ind = np.random.permutation(len(self.class_ind))[:n_way]
        target_classes = self.class_ind[target_classes_ind]
        samples = []
        for idx in target_classes_ind:
            samples.append(np.random.choice(self.data_ind[idx], k_shot + self.q_query))
        return np.array(samples), target_classes


# def low_label_rate_manager(data_path, data_name):
#     saved_path = os.path.join(data_path, data_name, "low_label_rate.pt") 
#     original_data_path = os.path.join(data_path, data_name, "geometric_data_processed.pt")

class LowRateLabelManager(SimpleFSManager):
    def __init__(self, class_ind, data_ind, k_shot, q_query, n_way, min_k_shot=None, min_n_way=None, test_idx = None, labels = None):
        self.class_ind = class_ind
        self.data_ind = data_ind
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_way = n_way
        self.min_n_way = min_n_way
        self.min_k_shot = min_k_shot
        self.test_idx = test_idx
        self.labels = labels
        self.total_labels = sorted(list(set(labels)))
        super(LowRateLabelManager, self).__init__(self.class_ind, self.data_ind, self.k_shot, self.q_query, self.n_way, self.min_k_shot, self.min_n_way)

    def get_few_shot_idx(self, mode='train', idx = -1):
        n_way = self.n_way
        k_shot = self.k_shot
        if mode == 'train':
            return super().get_few_shot_idx()
        else:
            cidx = self.class_ind.index(self.labels[idx])
            target_classes_ind = random_choice_with_first(list(range(self.class_ind)), size=n_way, first_element=cidx)
            target_classes = self.class_ind[target_classes_ind]
            samples = []
            for idx in target_classes_ind:
                if idx == cidx:
                    sel = []
                    sel.append(idx)
                    sel.extend(np.random.choice(self.data_ind[idx], k_shot).tolist())
                    sel = np.array(sel)
                    samples.append(sel)
                else:
                    samples.append(np.random.choice(self.data_ind[idx], k_shot + self.q_query))
            return np.array(samples), target_classes, cidx

def random_choice_with_first(a, size=None, replace=True, p=None, first_element=None):
    """
    Randomly choose elements from an array, with the option to specify the first element.

    Parameters:
    a : 1-D array-like or int
        If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a was np.arange(a).
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement. Default is True.
    p : 1-D array-like, optional
        The probabilities associated with each entry in a. If not given, the sample assumes a uniform distribution over all entries in a.
    first_element : Any, optional
        The element that should be selected first. If None, the first element is chosen randomly.

    Returns:
    samples : single item or ndarray
        The generated random samples.
    """
    
    # Handle the first element
    if first_element is not None:
        # Ensure the first element exists in the array
        if first_element not in a:
            raise ValueError("first_element must be in the array a")

        # Remove the first element to avoid duplication if sampling without replacement
        if not replace:
            a = [x for x in a if x != first_element]

        # Draw the remaining samples
        remaining_samples = np.random.choice(a, size=(size - 1) if size else None, replace=replace, p=p)
        
        # Combine the first element with the remaining samples
        samples = np.concatenate(([first_element], remaining_samples))
    else:
        # Standard random choice if no first element is specified
        samples = np.random.choice(a, size=size, replace=replace, p=p)
    
    return samples

class DataManager:
    @abstractmethod
    def get_data_loader(self, mode):
        pass


class FewShotDataManager(DataManager):
    """
    Return dataloader for train/val/test node idx.
    Example:
        data_manager = FewShotDataManager(g, params)
        train_dataloader = data_manager.get_dataloader(0)
        val_dataloader = data_manager.get_dataloader(1)
        test_dataloader = data_manager.get_dataloader(2)
        next(iter(train_dataloader)).shape: (n_way, k_shot + q_query)
    """

    def __init__(
            self,
            data,
            n_way,
            k_shot,
            q_query,
            class_split_ratio=None,
            class_split_lst=None,
            num_workers=0,
    ):
        super(FewShotDataManager, self).__init__()
        data.y = data.y.squeeze()
        self.n_way = n_way
        self.num_workers = num_workers
        self.dataset = FewShotDataset(
            data,
            k_shot + q_query,
            class_split_ratio,
            num_workers=self.num_workers,
            class_split_lst=class_split_lst,
        )
        # self.split = self.dataset.split

    def get_data_loader(self, mode):
        # mode: 0->train, 1->val, 2->test
        class_list = self.dataset.__getclass__(mode)
        sampler = EpisodeBatchSampler(self.n_way, class_list)
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        data_loader = DataLoader(self.dataset, **data_loader_params)
        return data_loader

    def get_dataset(self):
        return self.dataset


class FewShotDataset(Dataset):
    def __init__(
            self, data, batch_size, class_split_ratio, num_workers, class_split_lst
    ):
        self.data = data
        self.batch_size = batch_size
        self.class_split_ratio = class_split_ratio
        self.num_workers = num_workers
        self.class_split_lst = class_split_lst

        self.unique_label = torch.unique(self.data.y)

        self.cls_split_lst = self.class_split()
        self.cls_dataloader = self.create_subdataloader()
        # self.split = self.get_split_index()

    def class_split(self):
        """
        Split class for train/val/test in meta learning setting.
        Save as list: [[train_class_index], [val_class_index], [test_class_index], [all_class_index]]
        """
        if self.class_split_lst is not None:
            cls_split_lst = self.class_split_lst + [
                [[cls for sublist in self.class_split_lst for cls in sublist]]
            ]
            assert len(cls_split_lst) == 4

        elif self.class_split_ratio is not None:
            # create list according to class_split_ratio and save
            label = self.unique_label.cpu().detach()
            valid_label_mask = torch.where(label >= 0)
            label = label[valid_label_mask]
            selected_labels = []
            if hasattr(self.data, "non_cs_labels"):
                print("Ignore overlapped labels in mag240m with ogbn-arxiv.")
                label = torch.tensor(self.data.non_cs_labels)
                selected_labels = self.data.cs_labels
            # randomly shuffle
            label = label.index_select(0, torch.randperm(label.shape[0]))
            train_class, val_class, test_class = torch.split(
                label, self.class_split_ratio
            )
            cls_split_lst = [
                train_class.tolist(),
                val_class.tolist(),
                test_class.tolist(),
                label.tolist(),
                selected_labels,
            ]
        return cls_split_lst

    def label_to_index(self):
        """
        Generate a dictionary mapping labels to index list
        :return: dictionary: {label: [list of index]}
        """
        label = self.unique_label
        label2index = {}
        remove_label_list = []
        for i in label:
            idx = torch.nonzero(self.data.y == i)
            if idx.shape[0] < self.batch_size * 2:
                remove_label_list.append(int(i))
                label2index[int(i)] = None
            else:
                label2index[int(i)] = idx.squeeze()
        if len(remove_label_list) > 0:
            print(f"Remove invalid labels {len(remove_label_list)}.")
        self.invalid_labels = set(remove_label_list)

        return label2index, label

    def create_subdataloader(self):
        """
        :return: list of subdataloaders for each class i
        """
        label2index, label = self.label_to_index()
        cls_dataloader = []
        cls_dataloader_params = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        for c in label:
            idx = label2index[int(c)]
            if idx is None:
                cls_dataloader.append(False)
            else:
                cls_dataset = ClassDataset(label2index[int(c)])
                cls_dataloader.append(
                    DataLoader(cls_dataset, **cls_dataloader_params)
                )

        return cls_dataloader

    def get_split_index(self):
        """
        :return: dictionary that contains the node index for each split
        """
        label2index, label = self.label_to_index()
        cls_split_lst = self.cls_split_lst
        split = {"train": [], "valid": [], "test": []}

        exclude_labels = []
        for c in label:
            if c in cls_split_lst[0]:
                split["train"].extend(
                    [int(idx) for idx in label2index[int(c)]]
                )
            elif c in cls_split_lst[1]:
                split["valid"].extend(
                    [int(idx) for idx in label2index[int(c)]]
                )
            elif c in cls_split_lst[2]:
                split["test"].extend([int(idx) for idx in label2index[int(c)]])
            else:
                exclude_labels.append(c)

        # print("Ignore labels: " % exclude_labels)

        return split

    def __getitem__(self, class_index):
        node_id = next(iter(self.cls_dataloader[class_index]))
        return next(iter(self.cls_dataloader[class_index])), class_index

    def __len__(self):
        # mode = 0 -> train; 1 -> validation; 2 -> test
        return len(self.unique_label)

    def __getclass__(self, mode):
        # return available classes under current mode (train/val/test)
        # print(self.cls_split_lst)
        class_list = self.cls_split_lst[mode]
        valid_labels = [
            label for label in class_list if label not in self.invalid_labels
        ]
        return valid_labels


class EpisodeBatchSampler(object):
    def __init__(self, n_way, class_list):
        # TODO: change value of episode to some variables
        self.episode = 1
        self.n_way = n_way
        self.class_list = class_list

    def __len__(self):
        return self.episode

    def __iter__(self):
        for i in range(self.episode):
            batch_class = []
            # Don't change task_num for OFA
            task_num = 1
            for j in range(task_num):
                batch_class.append(
                    np.random.choice(
                        self.class_list, self.n_way, replace=False
                    )
                )
            yield np.concatenate(batch_class)


class ClassDataset(Dataset):
    def __init__(self, label_index):
        self.label_index = label_index

    def __getitem__(self, i):
        return self.label_index[i]

    def __len__(self):
        return self.label_index.shape[0]
