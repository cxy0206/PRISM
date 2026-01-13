import random
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np

class CLDataset_withoutCluster(Dataset):
    def __init__(self, data_list, threshold=None, num_positives=1, num_negatives=3, distance_func=None, parameter="Polarity"):
        self.data_list = data_list
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.distance_func = distance_func if distance_func is not None else self._hansen_p_distance
        self.parameter = parameter
        self.distance_matrix = self._precompute_distance_matrix()
        self.threshold = threshold if threshold is not None else self._calculate_default_threshold()

    def _hansen_p_distance(self, sample1, sample2, parameter):
        p1, p2 = sample1.Polarity[0], sample2.Polarity[0]
        d1, d2 = sample1.Dispersion[0], sample2.Dispersion[0]
        h1, h2 = sample1.HydrogenBonding[0], sample2.HydrogenBonding[0]

        if parameter == "Polarity":
            return np.abs(p1 - p2)
        elif parameter == "Dispersion":
            return np.abs(d1 - d2)
        elif parameter == "HydrogenBonding":
            return np.abs(h1 - h2)
        else:
            raise ValueError("Invalid combination of parameter.")

    def _precompute_distance_matrix(self):
        num_samples = len(self.data_list)
        distance_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    distance_matrix[i, j] = self.distance_func(self.data_list[i], self.data_list[j], parameter=self.parameter)
        return distance_matrix

    def _calculate_default_threshold(self):
        thresholds = []
        for i in range(len(self.data_list)):
            distances = self.distance_matrix[i]
            sorted_distances = np.sort(distances[distances > 0])
            if len(sorted_distances) < self.num_positives:
                thresholds.append(sorted_distances[-1])
            else:
                thresholds.append(sorted_distances[self.num_positives - 1])
        return np.max(thresholds)

    def _get_positive_samples(self, anchor_idx):
        positive_indices = [j for j in range(len(self.data_list)) if anchor_idx != j and self.distance_matrix[anchor_idx, j] < self.threshold]
        if len(positive_indices) == 0:
            positive_indices = random.sample(range(len(self.data_list)), self.num_positives)
        return random.sample(positive_indices, min(self.num_positives, len(positive_indices)))

    def _get_negative_samples(self, anchor_idx):
        negative_indices = [j for j in range(len(self.data_list)) if anchor_idx != j and self.distance_matrix[anchor_idx, j] >= self.threshold]
        if len(negative_indices) == 0:
            negative_indices = random.sample(range(len(self.data_list)), self.num_negatives)
        return random.sample(negative_indices, min(self.num_negatives, len(negative_indices)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        anchor_sample = self.data_list[idx]
        positive_indices = self._get_positive_samples(idx)
        positive_samples = [self.data_list[i] for i in positive_indices]
        negative_indices = self._get_negative_samples(idx)
        negative_samples = [self.data_list[i] for i in negative_indices]
        return anchor_sample, positive_samples, negative_samples

class CLDataset_withoutCluster_withoutAnchor(Dataset):
    def __init__(self, data_list, threshold=None, num_positive_pairs=10, num_negative_pairs=30, distance_func=None, parameter="Polarity"):
        self.data_list = data_list
        self.num_positive_pairs = num_positive_pairs
        self.num_negative_pairs = num_negative_pairs
        self.distance_func = distance_func if distance_func is not None else self._hansen_p_distance
        self.parameter = parameter
        self.distance_matrix = self._precompute_distance_matrix()
        self.threshold = threshold if threshold is not None else self._calculate_default_threshold()
        self.pair_dict = self._generate_all_pairs()

    def _hansen_p_distance(self, sample1, sample2, parameter):
        p1, p2 = sample1.Polarity[0], sample2.Polarity[0]
        d1, d2 = sample1.Dispersion[0], sample2.Dispersion[0]
        h1, h2 = sample1.HydrogenBonding[0], sample2.HydrogenBonding[0]

        if parameter == "Polarity":
            return np.abs(p1 - p2)
        elif parameter == "Dispersion":
            return np.abs(d1 - d2)
        elif parameter == "HydrogenBonding":
            return np.abs(h1 - h2)
        else:
            raise ValueError("Invalid combination of parameter.")

    def _precompute_distance_matrix(self):
        num_samples = len(self.data_list)
        distance_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    distance_matrix[i, j] = self.distance_func(self.data_list[i], self.data_list[j], parameter=self.parameter)
        return distance_matrix

    def _calculate_default_threshold(self):
        distances = self.distance_matrix[np.triu_indices(len(self.data_list), k=1)]
        return np.median(distances)

    def _generate_all_pairs(self):
        pair_dict = defaultdict(lambda: {'positives': [], 'negatives': []})
        for idx in range(len(self.data_list)):
            positive_indices = [j for j in range(len(self.data_list)) if j != idx and self.distance_matrix[idx, j] < self.threshold]
            negative_indices = [j for j in range(len(self.data_list)) if j != idx and self.distance_matrix[idx, j] >= self.threshold]

            if len(positive_indices) == 0:
                positive_indices = random.sample(range(len(self.data_list)), self.num_positive_pairs)

            if len(negative_indices) == 0:
                negative_indices = random.sample(range(len(self.data_list)), self.num_negative_pairs)

            pair_dict[idx]['positives'] = random.sample(positive_indices, min(self.num_positive_pairs, len(positive_indices)))
            pair_dict[idx]['negatives'] = random.sample(negative_indices, min(self.num_negative_pairs, len(negative_indices)))

        return pair_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_sample = self.data_list[idx]
        positive_indices = self.pair_dict[idx]['positives']
        negative_indices = self.pair_dict[idx]['negatives']
        positive_pairs = [(data_sample, self.data_list[p_idx]) for p_idx in positive_indices]
        negative_pairs = [(data_sample, self.data_list[n_idx]) for n_idx in negative_indices]
        return positive_pairs, negative_pairs

class ContrastiveLearningDataset(Dataset):
    def __init__(self, data_list, threshold=None, 
                 num_positives=1, num_negatives=3, 
                 num_positive_pairs=5, num_negative_pairs=10, 
                 num_pairs_per_anchor=1, distance_func=None, if_anchor=True, parameter="Polarity"):
        self.data_list = data_list
        self.if_anchor = if_anchor
        self.parameter = parameter

        if self.if_anchor:
            self.dataset = CLDataset_withoutCluster(
                data_list, 
                threshold=threshold, 
                num_positives=num_positives, 
                num_negatives=num_negatives, 
                distance_func=distance_func,
                parameter=parameter
            )
        else:
            self.dataset = CLDataset_withoutCluster_withoutAnchor(
                data_list, 
                threshold=threshold, 
                num_positive_pairs=num_positive_pairs, 
                num_negative_pairs=num_negative_pairs, 
                distance_func=distance_func,
                parameter=parameter
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
