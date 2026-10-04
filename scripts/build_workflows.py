"""Generate readable native UI workflows and matching API prompts from one graph."""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen-image-2.1-UC-Q8_0.gguf'
ENCODER = 'qwen3vl_8b_int8_convrot.safetensors'
VAE = 'qwen_image_2.1_vae_bf16.safetensors'


class Graph:
    def __init__(self):
        self.nodes, self.links, self.api = [], [], {}

    def node(self, id, type, title, pos, size, widgets, named, inputs, outputs):
        node = dict(id=id, type=type, title=title, pos=pos, size=size, flags={},
                    order=len(self.nodes), mode=0,
                    inputs=[dict(name=name, type=kind, link=None) for name, kind in inputs],
                    outputs=[dict(name=name, type=kind, links=[], slot_index=i)
                             for i, (name, kind) in enumerate(outputs)],
                    properties={'Node name for S&R': type}, widgets_values=widgets,
                    widgets_values_named=named.copy())
        self.nodes.append(node)
        self.api[str(id)] = {'class_type': type, 'inputs': named.copy(), '_meta': {'title': title}}
        # UI-only widget; it is not part of KSampler's API.
        self.api[str(id)]['inputs'].pop('control_after_generate', None)
        return node

    def connect(self, source, slot, target, name):
        index = next(i for i, item in enumerate(target['inputs']) if item['name'] == name)
        kind = source['outputs'][slot]['type']
        assert kind == target['inputs'][index]['type']
        link_id = len(self.links) + 1
        self.links.append([link_id, source['id'], slot, target['id'], index, kind])
        source['outputs'][slot]['links'].append(link_id)
        target['inputs'][index]['link'] = link_id
        self.api[str(target['id'])]['inputs'][name] = [str(source['id']), slot]

    def workflow(self, name):
        return dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL, 'qwen-imagegen/' + name)), revision=0,
                    last_node_id=max(n['id'] for n in self.nodes), last_link_id=len(self.links),
                    nodes=self.nodes, links=self.links, groups=[], config={},
                    extra={'ds': {'scale': 0.65, 'offset': [40, 70]}}, version=0.4)


def build(reference=False):
    graph = Graph()
    node, connect = graph.node, graph.connect
    model = node(1, 'UnetLoaderGGUF', 'Q8 diffusion model', [0, 0], [360, 80], [MODEL],
                 {'unet_name': MODEL}, [], [('MODEL', 'MODEL')])
    clip = node(2, 'CLIPLoader', 'Text + vision encoder on CPU', [0, 150], [360, 130],
                [ENCODER, 'qwen_image', 'cpu'], {'clip_name': ENCODER, 'type': 'qwen_image', 'device': 'cpu'},
                [], [('CLIP', 'CLIP')])
    vae = node(3, 'VAELoader', 'Qwen Image 2.1 VAE', [0, 350], [360, 80], [VAE],
               {'vae_name': VAE}, [], [('VAE', 'VAE')])
    prompt = ('Keep the same subject and facial identity as the reference image. '
              'Place the subject in a sunlit garden with soft natural light. Preserve realistic detail.' if reference else
              'An editorial photograph of a ceramic teapot on a walnut table beside a window. '
              'Soft morning light, subtle steam, detailed glaze, natural colors, crisp focus on the teapot.')
    inputs = [('clip', 'CLIP')]
    if reference:
        inputs += [('images.image_1', 'IMAGE'), ('vae', 'VAE')]
    encode = node(4, 'TextEncodeQwenImage21', 'Write your prompt / edit instruction here', [450, 150], [460, 380],
                  [prompt, '', 1024], {'prompt': prompt, 'negative_prompt': '', 'resolution': 1024},
                  inputs, [('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('latent', 'LATENT')])
    connect(clip, 0, encode, 'clip')
    cache = node(5, 'QwenImage21Cache', 'Lossless prefix cache in system RAM', [450, 0], [360, 110],
                 ['cpu', 'default'], {'device': 'cpu', 'dtype': 'default'}, [('model', 'MODEL')], [('MODEL', 'MODEL')])
    connect(model, 0, cache, 'model')
    if reference:
        source = node(6, 'LoadImage', 'Upload your reference image', [0, 500], [360, 340],
                      ['', 'image'], {'image': ''}, [], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')])
        connect(source, 0, encode, 'images.image_1')
        connect(vae, 0, encode, 'vae')
        latent, latent_slot = encode, 2
    else:
        latent = node(6, 'EmptyLatentImage', 'Output size: start at 1024 x 1024', [450, 600], [360, 140],
                      [1024, 1024, 1], {'width': 1024, 'height': 1024, 'batch_size': 1}, [], [('LATENT', 'LATENT')])
        latent_slot = 0
    sampler = node(7, 'KSampler', 'Quality: 40 steps, CFG 1', [1000, 150], [320, 300],
                   [42, 'randomize', 40, 1.0, 'euler', 'simple', 1.0],
                   {'seed': 42, 'control_after_generate': 'randomize', 'steps': 40, 'cfg': 1.0,
                    'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0},
                   [('model', 'MODEL'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('latent_image', 'LATENT')],
                   [('LATENT', 'LATENT')])
    for source, slot, name in [(cache, 0, 'model'), (encode, 0, 'positive'),
                                (encode, 1, 'negative'), (latent, latent_slot, 'latent_image')]:
        connect(source, slot, sampler, name)
    decode = node(8, 'VAEDecodeTiled', 'Tiled decode to save VRAM', [1400, 150], [300, 180],
                  [512, 64, 64, 8], {'tile_size': 512, 'overlap': 64, 'temporal_size': 64, 'temporal_overlap': 8},
                  [('samples', 'LATENT'), ('vae', 'VAE')], [('IMAGE', 'IMAGE')])
    connect(sampler, 0, decode, 'samples')
    connect(vae, 0, decode, 'vae')
    prefix = 'Qwen21/reference' if reference else 'Qwen21/text'
    save = node(9, 'SaveImage', 'Save PNG with workflow metadata', [1780, 150], [400, 420],
                [prefix], {'filename_prefix': prefix}, [('images', 'IMAGE')], [])
    connect(decode, 0, save, 'images')
    note = ('Upload an image in LoadImage before running. Describe the changes and what to preserve. '
            'The first reference sets output aspect ratio; resolution sets approximate square pixel budget. '
            'Start at 1024; lower to 768 if memory is tight.' if reference else
            'Edit the prompt and output dimensions, then Run. Start at 1024x1024; keep dimensions divisible by 32. '
            'Increase to 1536 or 2048 only after verifying available memory.')
    note += '\n\n40 steps / Euler / simple / CFG 1 / denoise 1 / batch 1. Negative prompt has no effect at CFG 1. '
    note += 'CPU encoding can take time, especially with references. Q8 is preserved; tiled VAE decode may introduce subtle seams.'
    node(10, 'MarkdownNote', 'How to use', [1000, 600], [660, 230], [note], {}, [], [])
    graph.api.pop('10')
    return graph


def main():
    output = ROOT / 'workflows'
    (output / 'api').mkdir(parents=True, exist_ok=True)
    for name, reference in [('qwen21_q8_text_to_image', False), ('qwen21_q8_reference_image', True)]:
        graph = build(reference)
        (output / (name + '.json')).write_text(json.dumps(graph.workflow(name), indent=2) + '\n')
        (output / 'api' / (name + '.json')).write_text(json.dumps(graph.api, indent=2) + '\n')


if __name__ == '__main__':
    main()
