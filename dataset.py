from data.chemmol.gen_data import MolOFADataset
from utils import SentenceEncoder
from torch_geometric.loader import DataLoader
from subgcon.efficient_dataset import NodeSubgraphDataset, LinkSubgraphDataset, GraphSubgraphDataset

# encoder = SentenceEncoder("minilm", root=".", batch_size=256)
# test = MolOFADataset(name = 'tox21', encoder=encoder, root="./cache_data_minilm",load_text=True)
# ll = DataLoader(test, batch_size=2, shuffle=False)

# aa = next(iter(ll))
# import ipdb; ipdb.set_trace()

test_node = 