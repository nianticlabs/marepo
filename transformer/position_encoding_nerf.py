'''
MIT License

Copyright (c) 2022 Active Vision Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Code modified from DFNet: Enhance Absolute Pose Regression with Direct Feature Matching (https://arxiv.org/abs/2204.00559)
'''

import torch

# Input Feature Positional Encoding based on NeRF Paper
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.N_freqs = 0
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        self.N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq,
                                              steps=self.N_freqs)  # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=self.N_freqs)

        for freq in freq_bands:  # 10 iters for 3D location, 4 iters for 2D direction
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # inputs [N, 3]
        if self.kwargs['max_freq_log2'] != 0:
            ret = torch.cat([fn(inputs) for fn in self.embed_fns], -1)  # cos, sin embedding # ret.shape [N, 33]
        else:
            ret = inputs
        return ret

def get_NeRF_embedder(multires):
    '''
    multires: log2 of max freq for positional encoding (3D location)
    '''

    # for experiments
    # NeRF paper default
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim, embedder_obj  # 63 for pos