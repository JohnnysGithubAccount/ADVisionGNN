from GVAE.ADVisionGNN.model.layers import *


class GraphVariationalAutoencoder(nn.Module):
    def __init__(self,
                 mode: str = "train",
                 memory_bank_path: str = "../model/results/model/memory_bank.index",
                 sub_type: str = "normal"):
        super(GraphVariationalAutoencoder, self).__init__()

        self.memory_bank_path = memory_bank_path
        self.memory_bank = list()
        self.mode = mode
        self.sub_type = sub_type

        # blocks = [2, 2, 6, 2]
        blocks = [2, 4, 2]
        self.n_blocks = sum(blocks)
        k = 9
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        channels = [48, 96, 240, 384]
        channels = [96, 240, 384]
        act = 'gelu'
        drop_path = 0.0
        reduce_ratios = [4, 2, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))

        self.encoder = GraphEncoder(
            num_block=blocks,
            hidden_channels=channels,
            reduce_ratios=reduce_ratios,
            dpr=dpr
        )

        self.decoder = CNNDecoder(
            hidden_channels=[384, 240, 96, 3],
            kernel_sizes=[4, 4, 4, 4],
            list_strides=[2, 2, 2, 2],
            paddings=[1, 1, 1, 1]
        )

        self.model_init()

        self.features = []
        def hook(module, inputs, outputs):
            self.features.append(outputs)

        self.encoder.blocks[4][1].fc2[1].register_forward_hook(hook)
        self.encoder.blocks[7][1].fc2[1].register_forward_hook(hook)

        self.avg = torch.nn.AvgPool2d(3, stride=1)

        self.y_score= list()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def fill_memory_bank(self):
        self.memory_bank = subsampling(self.memory_bank)
        save_to_faiss(
            self.memory_bank,
            self.memory_bank_path
        )

    def forward(self, inputs, is_saved: bool = False):
        segm_map = None
        embedding_layer = self.stem(inputs) + self.pos_embed
        B, C, H, W = embedding_layer.shape

        self.features = []
        encoder_output = self.encoder(embedding_layer)

        fmap_size = self.features[0].shape[-2]
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)  # Merge the resized feature maps
        patch = patch.reshape(patch.shape[1], -1).T  # Create a column tensor

        if self.mode == "train":
            sub_patch = subsampling(patch) if self.sub_type == "normal" else corset_subsampling(patch)
            print("Sub patch", sub_patch.shape)
            sub_patch = sub_patch.detach().cpu().numpy()
            save_to_faiss(sub_patch, path=self.memory_bank_path)

        memory_bank = load_faiss_to_tensor(self.memory_bank_path).cuda()
        print("Patch shape, memory bank shape:", patch.shape, memory_bank.shape)
        distances = torch.cdist(
            patch,
            memory_bank
        )
        print("distance", distances.shape)
        dist_score, dist_score_idxs = torch.min(distances, dim=1)
        s_star = torch.max(dist_score)

        if self.mode == "test":
            segm_map = dist_score.view(B, 1, 28, 28)
            segm_map = F.interpolate(segm_map, size=(224, 224), mode='bilinear', align_corners=False)

        x_reconstruct = self.decoder(encoder_output)
        if self.mode == 'test' and segm_map is not None:
            return x_reconstruct, segm_map, s_star
        return x_reconstruct, encoder_output, s_star





def main():
    pass


if __name__ == "__main__":
    main()
