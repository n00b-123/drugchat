import os
import json
import pickle
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Data, Batch


class MultimodalDataset(Dataset):
    def __init__(self, datapath, use_image=True, use_graph=False, image_size=224, is_train=False) -> None:
        super().__init__()
        self.use_image = use_image
        self.use_graph = use_graph
        jsonpath = os.path.join(datapath, "smiles_img_qa.json")
        print(f"Using {jsonpath=}")
        with open(jsonpath, "rt") as f:
            meta = json.load(f)

        if use_graph:
            with open(os.path.join(datapath, "graph_smi.pkl"), "rb") as f:
                graphs = pickle.load(f)
        
        self.images = {}
        self.data = []
        self.graphs = {}
        for idx, rec in meta.items():
            if use_image:
                img_file = 'img_{}.png'.format(idx)
                image_path = os.path.join(datapath, img_file)
                # Removed image processing using torchvision
                self.images[idx] = image_path  # Store the image path
            smi, qa = rec
            if use_graph:
                g = graphs[smi]["graph"]
                graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
                self.graphs[idx] = graph
            qa = [(idx, qa_pair) for qa_pair in qa]
            self.data.extend(qa)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx, qa_pair = self.data[index]
        out = {"question": qa_pair[0], "text_input": str(qa_pair[1])}
        if self.use_image:
            img_path = self.images[idx]  # Retrieve the image path instead of the image itself
            out.update({"img": img_path})  # You can later process it as needed
        if self.use_graph:
            out.update({"graph": self.graphs[idx]})
        return out
    
    @staticmethod
    def collater(samples):
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        out = {"question": qq, "text_input": aa}
        if "img" in samples[0]:
            img_paths = [x["img"] for x in samples]  # Retrieve image paths
            out.update({"image": img_paths})  # Use image paths for now
        if "graph" in samples[0]:
            g = Batch.from_data_list([x["graph"] for x in samples])
            out.update({"graph": g})
        return out
