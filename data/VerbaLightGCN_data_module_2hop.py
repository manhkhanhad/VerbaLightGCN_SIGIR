from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import networkx as nx
from scipy.sparse import coo_matrix
from collections import Counter

# Set up logger for this module
import logging
logger = logging.getLogger(__name__)

class VerbaLightGCNDataset(Dataset):
    """Dataset for recommendation data."""

    def __init__(
        self,
        data: pd.DataFrame,
        item_attributes: Dict[int, str],
        graph: Any,
        mode: str = "train",
        seed: int = 42,
        ranking_mode: bool = False,
        num_negative_sample: int = 19,
        valid_item: List[int] = [],
    ):
        """Initialize the dataset.

        Args:
            data: DataFrame containing interaction data
            item_attributes: Dictionary mapping item IDs to their attributes
            graph: NetworkX graph of user-item interactions
            mode: Dataset mode (train, val, or test)
        """
        self.data = data
        self.item_attributes = item_attributes
        self.graph = graph
        self.mode = mode
        self.ranking_mode = ranking_mode
        if self.ranking_mode:
            self.test_columns = ['item_1', 'item_2', 'label', 'neighbor_item', 
                      'neighbor_user_1', 'neighbor_user_2', 'user_2hop_node', 'item_1_2hop_node', 'item_2_2hop_node', 'candidate', 'pos_index']
        else:
            self.test_columns = ['item_1', 'item_2', 'label', 'neighbor_item',
                      'neighbor_user_1', 'neighbor_user_2', 'user_2hop_node', 'item_1_2hop_node', 'item_2_2hop_node']
        self.num_negative_sample = num_negative_sample
        self.valid_item = valid_item
        logger.info(f"Valid item: {len(self.valid_item)}")

        self.rng = np.random.RandomState(seed)
        if mode != 'train':
            self.data = self.sample_test_data()
        else:
            self.data = self.data.explode('train')
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node in the graph.

        Args:
            node_id: Node ID in the graph

        Returns:
            List of neighbor node IDs
        """
        try:
            return list(self.graph.neighbors(node_id))
        except:
            logger.info(f"Node {node_id} not in graph")
            return []

    def get_2hop_neighbors(self, node_ids: List[str]) -> List[str]:
        """Get 2-hop neighbors of a node in the graph.
        Input is: list of 1-hop neighbors
        Output is: list of 2-hop neighbors, select top 10 most common neighbors
        """
        neighbors = []
        for node_id in node_ids:
            neighbors.extend(self.get_neighbors(node_id))
        if len(neighbors) == 0:
            return []
        neighbors_count = Counter(neighbors)
        return [neighbor for neighbor, count in neighbors_count.most_common(10)]


    
    def _generate_sample(self, sample: pd.Series, is_random_item: bool = True) -> Tuple[int, int, int, List[int], List[int], List[int]]:
        """Generate a sample for training or evaluation.

        Args:
            sample: Input data sample
            is_random_item: Whether to randomly select items

        Returns:
            Tuple containing (item_1, item_2, label, neighbor_item, neighbor_user_1, neighbor_user_2)
        """
        user_id = sample['user_id']
        
        if is_random_item:
            if 'candidate' in sample: # Handle the transition from old data format to new data format
                valid_candidate = [i for i in list(set(sample['candidate']) - set(sample['item_id'])) if i in self.valid_item]
            else:
                valid_candidate = list(set(self.valid_item) - set(sample['item_id']))
            if self.mode == 'train':
                # pos_item = self.rng.choice([i for i in sample['train'] if i in self.valid_item], 1)[0]
                pos_item = sample['train']
                neg_item = np.random.choice(valid_candidate, 1)[0]
            elif self.mode == 'test':
                pos_item = sample['test']
                neg_item = self.rng.choice(valid_candidate, 1)[0]
            else:  # val mode
                pos_item = sample['val']
                neg_item = self.rng.choice(valid_candidate, 1)[0]
                # neg_item = np.random.choice(valid_candidate, 1)[0]
        else:
            if sample['label'] == 1:
                neg_item, pos_item = sample['item_2'], sample['item_1']
            elif sample['label'] == 2:
                neg_item, pos_item = sample['item_1'], sample['item_2']
            else:
                raise ValueError("Label should be 1 or 2.")

        # Get neighbor users for positive and negative items
        neighbor_user_pos_item = [int(i.strip("user_id")) for i in self.get_neighbors("item_id_" + str(pos_item))]
        neighbor_user_pos_item = [i for i in neighbor_user_pos_item if i != user_id]
        if len(neighbor_user_pos_item) > 20:
            neighbor_user_pos_item = np.random.choice(neighbor_user_pos_item, 20, replace=False)
            
        neighbor_user_neg_item = [int(i.strip("user_id")) for i in self.get_neighbors("item_id_" + str(neg_item))]
        neighbor_user_neg_item = [i for i in neighbor_user_neg_item if i != user_id]
        if len(neighbor_user_neg_item) > 20:
            neighbor_user_neg_item = np.random.choice(neighbor_user_neg_item, 20, replace=False)

        # Get neighbor items for the user
        user_test_sample = set([sample['test'], sample['val'], pos_item])
        neighbor_item = [int(i.strip("item_id")) for i in self.get_neighbors("user_id_" + str(user_id)) if int(i.strip("item_id")) not in user_test_sample]
        neighbor_item = [i for i in neighbor_item if i != pos_item]  # remove pos item from historical items
        if len(neighbor_item) > 20:
            neighbor_item = np.random.choice(neighbor_item, 20, replace=False)

        # Randomly assign items to item_1 and item_2
        if self.rng.uniform(0, 1) < 0.5:
            label = 2
            item_1 = neg_item
            item_2 = pos_item
            neighbor_user_1 = neighbor_user_neg_item
            neighbor_user_2 = neighbor_user_pos_item
        else:
            label = 1
            item_1 = pos_item
            item_2 = neg_item
            neighbor_user_1 = neighbor_user_pos_item
            neighbor_user_2 = neighbor_user_neg_item

        #### Add multihop data
        user_2hop_node = [int(j.strip("user_id")) for j in self.get_2hop_neighbors(['item_id_' + str(i) for i in neighbor_item]) if j != 'user_id_' + str(user_id)]
        item_1_2hop_node = [int(j.strip("item_id")) for j in self.get_2hop_neighbors(['user_id_' + str(i) for i in neighbor_user_1]) if j != 'item_id_' + str(item_1)]
        item_2_2hop_node = [int(j.strip("item_id")) for j in self.get_2hop_neighbors(['user_id_' + str(i) for i in neighbor_user_2]) if j != 'item_id_' + str(item_2)]
        if len(user_2hop_node) == 0: 
            user_2hop_node = [0]
        if len(item_1_2hop_node) == 0:
            item_1_2hop_node = [0]
        if len(item_2_2hop_node) == 0:
            item_2_2hop_node = [0]
        ####
        
        # Base return values for all modes
        base_return = (item_1, item_2, label, neighbor_item, neighbor_user_1, neighbor_user_2, user_2hop_node, item_1_2hop_node, item_2_2hop_node)
        
        # Return base values for train mode or non-ranking evaluation
        if self.mode == "train" or not self.ranking_mode:
            return base_return
        # Add ranking data for evaluation with ranking mode
        if 'candidate' in sample:
            negative_list = [i for i in sample['candidate'] if i != pos_item]
        else:
            negative_list = self.rng.choice(valid_candidate, self.num_negative_sample, replace=False)
        candidate = [pos_item] + list(negative_list)
        self.rng.shuffle(candidate)    
        pos_index = candidate.index(pos_item) + 1  # Candidate index starts from 1
        return (*base_return, candidate, pos_index)

    def sample_test_data(self) -> pd.DataFrame:
        """Sample test data."""
        data = self.data.copy()
        generated_samples = data.apply(lambda x: pd.Series(self._generate_sample(x)), axis=1)
        data[self.test_columns] = generated_samples
        return data

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing sample data
        """
        sample = self.data.iloc[idx]
        user_id = sample['user_id']
        if self.mode == "train":
            item_1, item_2, label, neighbor_item, neighbor_user_1, neighbor_user_2, user_2hop_node, item_1_2hop_node, item_2_2hop_node = self._generate_sample(sample, is_random_item=True)
            return {
                "user_id": user_id,
                "item_1": item_1,
                "item_2": item_2,
                "label": label,
                "neighbor_item": neighbor_item,
                "neighbor_user_1": neighbor_user_1,
                "neighbor_user_2": neighbor_user_2,
                "user_2hop_node": user_2hop_node,
                "item_1_2hop_node": item_1_2hop_node,
                "item_2_2hop_node": item_2_2hop_node,
            }
        else:
            return sample.to_dict()
        
    def set_data(self, data: pd.DataFrame):
        self.data = data

def recommendation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for recommendation data.
    
    Args:
        batch: List of dictionaries containing sample data
        
    Returns:
        Dictionary containing batched tensors
    """
    # Initialize lists to store batched data
    user_ids = []
    item_1s = []
    item_2s = []
    labels = []
    neighbor_items = []
    neighbor_users_1 = []
    neighbor_users_2 = []
    user_2hop_nodes = []
    item_1_2hop_nodes = []
    item_2_2hop_nodes = []
    if 'candidate' in batch[0]: # Ranking mode
        candidates = []
        pos_indices = []
    
    # Get max lengths for padding
    max_neighbor_items = max(len(sample['neighbor_item']) for sample in batch)
    max_neighbor_users = max(
        max(len(sample['neighbor_user_1']), len(sample['neighbor_user_2'])) 
        for sample in batch
    )
    max_2hop_nodes = max(
        max(len(sample['user_2hop_node']), len(sample['item_1_2hop_node']), len(sample['item_2_2hop_node']))
        for sample in batch
    )
    
    # Process each sample in the batch
    for sample in batch:
        user_ids.append(sample['user_id'])
        item_1s.append(sample['item_1'])
        item_2s.append(sample['item_2'])
        labels.append(sample['label'])
        
        # Pad neighbor items
        neighbor_item = sample['neighbor_item']
        if len(neighbor_item) < max_neighbor_items:
            neighbor_item = neighbor_item + [0] * (max_neighbor_items - len(neighbor_item))
        neighbor_items.append(neighbor_item)
        
        # Pad neighbor users for item 1
        neighbor_user_1 = sample['neighbor_user_1']
        if len(neighbor_user_1) < max_neighbor_users:
            neighbor_user_1 = neighbor_user_1 + [0] * (max_neighbor_users - len(neighbor_user_1))
        neighbor_users_1.append(neighbor_user_1)
        
        # Pad neighbor users for item 2
        neighbor_user_2 = sample['neighbor_user_2']
        if len(neighbor_user_2) < max_neighbor_users:
            neighbor_user_2 = neighbor_user_2 + [0] * (max_neighbor_users - len(neighbor_user_2))
        neighbor_users_2.append(neighbor_user_2)

        # Pad user 2hop nodes
        user_2hop_node = sample['user_2hop_node']
        if len(user_2hop_node) < max_2hop_nodes:
            user_2hop_node = user_2hop_node + [0] * (max_2hop_nodes - len(user_2hop_node))
        user_2hop_nodes.append(user_2hop_node)

        # Pad item 1 2hop nodes
        item_1_2hop_node = sample['item_1_2hop_node']
        if len(item_1_2hop_node) < max_2hop_nodes:
            item_1_2hop_node = item_1_2hop_node + [0] * (max_2hop_nodes - len(item_1_2hop_node))
        item_1_2hop_nodes.append(item_1_2hop_node)

        # Pad item 2 2hop nodes
        item_2_2hop_node = sample['item_2_2hop_node']
        if len(item_2_2hop_node) < max_2hop_nodes:
            item_2_2hop_node = item_2_2hop_node + [0] * (max_2hop_nodes - len(item_2_2hop_node))
        item_2_2hop_nodes.append(item_2_2hop_node)

        if 'candidate' in sample: # Ranking mode
            candidates.append(sample['candidate'])
            pos_indices.append(sample['pos_index'])
    
    # Convert to tensors
    res = {
        'user_id': torch.tensor(user_ids),
        'item_1': torch.tensor(item_1s),
        'item_2': torch.tensor(item_2s),
        'label': torch.tensor(labels),
        'neighbor_item': torch.tensor(neighbor_items),
        'neighbor_user_1': torch.from_numpy(np.array(neighbor_users_1)),
        'neighbor_user_2': torch.from_numpy(np.array(neighbor_users_2)),
        'user_2hop_node': torch.from_numpy(np.array(user_2hop_nodes)),
        'item_1_2hop_node': torch.from_numpy(np.array(item_1_2hop_nodes)),
        'item_2_2hop_node': torch.from_numpy(np.array(item_2_2hop_nodes)),
    }
    if 'candidate' in batch[0]:
        res['candidate'] = torch.tensor(candidates)
        res['pos_index'] = torch.tensor(pos_indices)
    return res



class VerbaLightGCNDataModule(pl.LightningDataModule):
    """Data module for EM alignment data."""

    def __init__(
        self,
        interaction_path: str,
        item_info_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        ranking_mode: bool = False,
        **kwargs: Any,
    ):
        """Initialize the data module.

        Args:
            interaction_path: Path to interaction data
            item_info_path: Path to item information
            user_profile_path: Path to user profiles
            item_profile_path: Path to item profiles
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
            val_split: Validation split ratio
            test_split: Test split ratio
            **kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.interaction_path = interaction_path
        self.item_info_path = item_info_path
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ranking_mode = ranking_mode
        self.prepare_data()
        self.setup()


    def prepare_data(self) -> None:
        """Prepare data for training."""
        # Load interaction data
        self.interaction_df = pd.read_parquet(self.interaction_path)
        if 'full_interacted_item' in self.interaction_df.columns:
            self.interaction_df = self.interaction_df.drop(columns='item_id').rename(columns={'full_interacted_item': 'item_id'})
        self.user_num = self.interaction_df.user_id.nunique() + 1
        self.item_num = self.interaction_df.item_id.explode('item_id').nunique() + 1

        self.all_item = list(set(self.interaction_df.item_id.explode('item_id').tolist()))
        self.all_user = list(set(self.interaction_df.user_id.tolist()))

        #TODO: remove hardcode here
        if "kw" in self.item_info_path:
            with open(self.item_info_path, 'r') as f:
                item_attributes = json.load(f)
            self.item_attributes = {}
            for k,v in item_attributes.items():
                self.item_attributes[int(k)] = json.dumps({"Title": v['Title'] if 'Title' in v else v['item_name'], "Item key characteristics": v['feature']})
        elif "desciption" in self.item_info_path:
            with open(self.item_info_path, 'r') as f:
                item_description = json.load(f)
            self.item_attributes = {}
            for item_id in list(item_description['title'].keys()):
                attr = {
                    "Title": item_description['title'][item_id].replace("&amp;", "").replace("&quot;", "").replace("&apos;", "").replace("&lt;", "").replace("&gt;", ""),
                    "description": item_description['description'][item_id].replace("&amp;", "").replace("&quot;", "").replace("&apos;", "").replace("&lt;", "").replace("&gt;", ""),
                }
                if item_description['brand'][item_id] != '':
                    attr['brand'] = item_description['brand'][item_id]
                item_id = int(item_id)
                self.item_attributes[item_id] = json.dumps(attr)

        # Preparing data
        self.generate_graph()
        self.interaction_df = self.interaction_df[self.interaction_df['train'].map(lambda x: len([i for i in x if i in self.valid_item])) > 0]
        self.sparse_matrix = self._create_sparse_matrix()

    def get_user_item_num(self) -> Tuple[int, int]:
        """Get number of users and items."""
        return self.user_num, self.item_num
    
    def generate_graph(self):
        pair_interaction = self.interaction_df.copy()
        pair_interaction = pair_interaction[['user_id','item_id']].explode('item_id')
        self.valid_item = set(pair_interaction.item_id.tolist()).intersection(set(self.item_attributes.keys()))
        self.valid_user = set(pair_interaction.user_id.tolist())
        pair_interaction['user_id'] = pair_interaction['user_id'].apply(lambda x: "user_id_" + str(x))
        pair_interaction['item_id'] = pair_interaction['item_id'].apply(lambda x: "item_id_" + str(x))
        pair_interaction = pair_interaction.drop_duplicates()
        self.graph = nx.Graph()
        self.graph.add_edges_from(zip(pair_interaction['user_id'], pair_interaction['item_id']))


        # ADD all node
        for item in self.all_item:
            if item not in self.valid_item:
                logger.info(f"Adding item {item} to graph")
                self.graph.add_node("item_id_" + str(item))
        for user in self.all_user:
            if user not in self.valid_user:
                logger.info(f"Adding user {user} to graph")
                self.graph.add_node("user_id_" + str(user))
        logger.info(f"Graph item node coverage: {len(self.valid_item) / self.item_num}")
        logger.info(f"Graph user node coverage: {len(self.valid_user) / self.user_num}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing.

        Args:
            stage: Current stage (fit, validate, test, or predict)
        """
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = VerbaLightGCNDataset(
                self.interaction_df,
                self.item_attributes,
                self.graph,
                mode="train",
                seed=self.seed,
                ranking_mode=self.ranking_mode,
                valid_item=self.valid_item,
            )

            self.val_dataset = VerbaLightGCNDataset(
                    self.interaction_df.query('val != 0'),
                    self.item_attributes,
                    self.graph,
                    mode="val",
                    seed=self.seed,
                    ranking_mode=self.ranking_mode,
                    valid_item=self.valid_item,
                )
        
        if stage == "test" or stage is None:
            self.test_dataset = VerbaLightGCNDataset(
                self.interaction_df.query('test != 0'),
                self.item_attributes,
                self.graph,
                mode="test",
                seed=self.seed,
                ranking_mode=True,
                valid_item=self.valid_item,
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader.

        Returns:
            Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=recommendation_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader.

        Returns:
            Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=recommendation_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader.

        Returns:
            Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=recommendation_collate_fn,
        )

    def _create_sparse_matrix(self, form: str = "coo") -> Any:
        """Create sparse matrix from training data.
        
        Args:
            form: Matrix format ("coo" or "csr")
            
        Returns:
            Sparse matrix in specified format
        """
        train_data = self.interaction_df[['user_id', 'train']].explode('train')
        src = train_data['user_id'].values
        tgt = train_data['train'].values
        data = np.ones(len(train_data))
        mat = coo_matrix(
            (data, (src, tgt)), shape=(self.user_num, self.item_num)
        )

        if form == "coo":
            return mat
        elif form == "csr":
            return mat.tocsr()

# if __name__ == "__main__":
#     data_module = RecommendationDataModule(
#         interaction_path="/home/khanh/data/LLM_GCN/dataset/Toys_small/interaction_sequence_w_candidate.parquet",
#         item_info_path="/home/khanh/data/LLM_GCN/dataset/Toys_small/item_desciption.json",
#     )
#     data_module.prepare_data()
#     data_module.setup()