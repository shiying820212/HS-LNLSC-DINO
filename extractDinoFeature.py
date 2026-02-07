import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import os

pca = PCA(n_components=128, whiten=True)

vit_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


def generate_patch_coords(img_size, patch_size=16):
    grid = img_size // patch_size
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    xs = (xs + 0.5) * patch_size
    ys = (ys + 0.5) * patch_size
    return xs.flatten(), ys.flatten()


class DINOPatchExtractor:
    def __init__(self,
                 layer_id=8,              
                 device="cuda"):
        self.device = device
    
        #1. 使用官方 DINO vits8
        self.model = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vitb16'
        ).to(device).eval()

        self.layer_id = layer_id

        #2. 从模型结构中读取信息
        self.embed_dim = self.model.embed_dim        # 384
        self.patch_size = self.model.patch_embed.patch_size  # 8

        self._features = None

        #3. 在 Transformer block 上 hook
        self.model.blocks[layer_id].register_forward_hook(self._hook)


    def _hook(self, module, inp, out):
        self._features = out

    @torch.no_grad()
    def extract(self, img_tensor):
        _ = self.model(img_tensor)
        # x = self._features[:, 1:, :]   # 去 CLS
        x = self._features[:, 0, :]   # [CLS]
        x = x.squeeze(0)               # [N, D]
        return x

def extract_image_to_mat(
    img_path,
    save_path,
    extractor,
    device='cuda'
):
    # 1. load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = vit_transform(img).unsqueeze(0).to(device)

    # 2. extract ViT patch features
    patch_features =extractor.extract(img_tensor)   # vit[196, 768] dino[196,384]
    #pca降维
    vit_feat=patch_features.cpu().numpy()
    

    patch_features = vit_feat / (np.linalg.norm(vit_feat, axis=0, keepdims=True) + 1e-6)  #cls

    # L2 normalize
    # vit_feat = pca.fit_transform(vit_feat)      #patch
    # patch_features = vit_feat / (np.linalg.norm(vit_feat, axis=1, keepdims=True) + 1e-6)  #patch

    # 3. patch coordinates
    xs, ys = generate_patch_coords(
        img_tensor.shape[2],
        extractor.patch_size
    )

    # 4. construct feaSet (MATLAB compatible)
    feaSet = {
        # "feaArr": patch_features.cpu().numpy().astype(np.double).T,  # N × D 最后转置成128*196 
        "feaArr": patch_features.astype(np.double).T,  # N × D 最后转置成128*196
        "x": ys.reshape(-1, 1).astype(np.double),
        "y": xs.reshape(-1, 1).astype(np.double),
        "width": np.array(img_tensor.shape[2], dtype=np.double),
        "height": np.array(img_tensor.shape[2], dtype=np.double),
    }

    sio.savemat(save_path, {"feaSetVit": feaSet})     #cls
    # sio.savemat(save_path, {"feaSetVitPatch": feaSet})      #patch




def process_dataset(img_root, save_root,layer=4, device='cuda'):
    extractorDino = DINOPatchExtractor(
        layer_id=layer,
        device="cuda"
    )

    for cls_name in os.listdir(img_root):
        cls_path = os.path.join(img_root, cls_name)
        if not os.path.isdir(cls_path):
            continue

        save_cls_path = os.path.join(save_root, cls_name)
        os.makedirs(save_cls_path, exist_ok=True)

        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(cls_path, fname)
            mat_name = os.path.splitext(fname)[0] + '.mat'
            save_path = os.path.join(save_cls_path, mat_name)

            print(f'Processing {img_path}')
            extract_image_to_mat(img_path, save_path, extractorDino, device)


if __name__ == "__main__":

    layers=[12]
    for lay in layers:
        # img_root = "D:/NLLSC/image/Corel10"
        # save_root = "D:/NLLSC/dataVitCls2/Corel10"

        # img_root = 'D:/NLLSC/imageMID/MID'                  
        # save_root = 'D:/NLLSC/dataMIDVit/MID'                 
        

        # img_root = 'D:/NLLSC/imageSeaShips/SeaShips'                  
        # save_root = 'D:/NLLSC/dataSeaShipsVitCls#/SeaShips'.replace('#',str(lay))    #768*1             
        # print(save_root)

        # img_root = 'D:/NLLSC/imageScene15/Scene15'                  
        # save_root = 'D:/NLLSC/dataScene15VitCls#/Scene15'.replace('#',str(lay))    #768*1             
        # print(save_root)

        # img_root = 'D:/NLLSC/imageSeaShips/SeaShips'                  
        # save_root = 'D:/NLLSC/dataSeaShipsVitPatch#/SeaShips'.replace('#',str(lay))    #768*1             
        # print(save_root)

        # img_root = 'D:/NLLSC/imageScene15/Scene15'                  
        # save_root = 'D:/NLLSC/dataScene15VitPatch#/Scene15'.replace('#',str(lay))    #768*1             
        # print(save_root)

        # img_root = 'D:/NLLSC/imageSeaShipClip/SeaShipsClip'                
        # save_root = 'D:/NLLSC/dataSeaShipClipVitCls#/SeaShipsClip'.replace('#',str(lay))                   


        # img_root = 'D:/NLLSC/imageSMDClip/SMDClip'                  
        # save_root = 'D:/NLLSC/dataSMDClipVitCls#/SMDClip'.replace('#',str(lay))


        img_root = 'D:/NLLSC/imageCorel10/Corel10'                  
        save_root = 'D:/NLLSC/dataCorel10VitCls#/Corel10'.replace('#',str(lay))                  
    
        # img_root = 'D:/NLLSC/imageCaltech101/Caltech101'                  
        # save_root = 'D:/NLLSC/dataCaltech101VitCls#/Caltech101'.replace('#',str(lay)) 
        
        # img_root = 'D:/NLLSC/imageCaltech256/Caltech256'                  
        # save_root = 'D:/NLLSC/dataCaltech256VitCls#/Caltech256'.replace('#',str(lay)) 

        process_dataset(img_root, save_root,layer=lay-1, device='cuda')
