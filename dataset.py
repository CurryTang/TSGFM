from data.chemmol.gen_data import MolOFADataset
from utils import SentenceEncoder


encoder = SentenceEncoder("minilm", root=".", batch_size=256)
test = MolOFADataset(name = 'tox21', encoder=encoder, root=".",load_text=True)
import ipdb; ipdb.set_trace()
