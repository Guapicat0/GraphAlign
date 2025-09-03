import argparse
import json
import random
from pathlib import Path
import os, sys
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
import util.misc as utils
import wandb
from collections import OrderedDict


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('-c', '--config_file', type=str, default='config/GroundingDINO_SwinT_OGC_ovdr.py',
                        help='Path to the configuration file')
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')

    # dataset parameters


    # test parameters
    parser.add_argument('--output_dir', type=str, default='./logs',
                        help='Directory to save output files')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', type=str, default='./weights/vg-ovdr-swint-mega-best.pth',
                        help='Path to the model weights for resuming training or evaluation')
    #-------------------------
    # results.png contains all nodes and edg,but to
    # generate some high confidence nodes or edgs,we set the threshold:
    #--------------------------
    # sgg threshold
    parser.add_argument('--node_model', type=int, default=1,
                        help='node_model=1   ->top_n select the nodes}'
                             'node_model=2   ->threshold select the nodes')
    parser.add_argument('--top_n', type=int, default=6,
                        help='when using node_model =1,select the high confidence nodes')
    parser.add_argument('--bbox_threshold', type=int, default=0.25,
                        help='when using node_model =2,select the bbox which is higher then detection logit')

    parser.add_argument('--edge_model', type=int, default=1,
                        help='node_model=1   ->top_n select the edges}'
                             'node_model=2   ->threshold select the edges')
    parser.add_argument('--relation_top_n', type=int, default=12,
                        help='when using edge_model =1,select the high confidence edges ')
    parser.add_argument('--relation_threshold', type=int, default=0.20,
                        help='when using edge_model =2,select the relation which is higher then relation logit')


    # dataset predict model
    parser.add_argument('--model', type=int, default=2,
                        help='model=1   ->input : image_path}'
                             'model=2   ->input : dataset_path')
    parser.add_argument('--image_path', type=str, default="/home/ippl7/LFY/code/TEXT2IMAGE/ssg/968.png",
                        help='when using model =1,select the image_path to do sgg')
    parser.add_argument('--dataset', type=str, default="/home/ippl7/LFY/code/datasets/ft/AIGCIQA2023/dataset_seed3",
                        help='when using model =2,dataset path, local dataset to generate scene graph in json format')


    # generation button
    parser.add_argument('--det', type=bool, default=False,
                        help='whether to generate detection image-bbox')
    parser.add_argument('--output_dir_det', type=str, default="/home/ippl7/LFY/code/datasets/ft/SG-RichHF-18K/det2",
                        help='the path of detection image')

    parser.add_argument('--sgg', type=bool, default=False,
                        help='whether to generate sgg in json format')
    parser.add_argument('--output_dir_sgg', type=str, default="/home/ippl7/LFY/code/datasets/ft/SG-RichHF-18K/det",
                        help='the path of sgg in json format')

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict, "ERROR: modelname:{} not in models:{}".format(args.modelname, MODULE_BUILD_FUNCS._module_dict)

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

import torch
import torchvision.transforms as transforms
from PIL import Image


# 定义 NestedTensor 类
class NestedTensor(object):
    def __init__(self, tensors, mask, device):
        self.tensors = tensors
        self.mask = mask
        self.device = device
        self.shape = {'tensors': tensors.shape, 'mask': mask.shape}

    @property
    def tensor(self):
        return self.tensors

# 预处理图像函数
import datasets.transforms as T
def preprocess_image(image_path, device='cuda:0'):
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    # config the params for data aug
    max_size = 1333
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # 调整图像大小
    transform = T.Compose([
        T.RandomResize([max(scales)], max_size=max_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image,None)[0].unsqueeze(0).to(device)  # 添加批次维度并移动到指定设备

    # 创建掩码
    mask = torch.zeros((1, image_tensor.size(2),image_tensor.size(3)), dtype=torch.bool, device=device)  # 默认所有像素都不是填充的

    return NestedTensor(image_tensor, mask, device)

def results2scenegraph(results,name2classes,name2predicates):
    # 创建逆映射字典
    image_path =args.image_path
    class2name = {v: k for k, v in name2classes.items()}
    predicate2name = {v: k for k, v in name2predicates.items()}

    # 假设 results.png 列表中只有一个元素，或者我们需要处理所有元素
    for result in results:
        graph = result.get('graph')
        pred_boxes_class = graph['pred_boxes_class']
        all_node_pairs = graph['all_node_pairs']
        all_relation = graph['all_relation']
        pred_boxes_score = graph['pred_boxes_score']
        pred_boxes = graph['pred_boxes']

    # 找出 pred_boxes_score > 0.3 的 pred_boxes_class
    if args.node_model == 1:
        valid_indices = torch.argsort(pred_boxes_score, descending=True)[:args.top_n]

    elif args.node_model == 2:
        valid_indices = torch.where(pred_boxes_score > args.bbox_threshold)[0]

    valid_classes = pred_boxes_class[valid_indices]
    valid_boxes = pred_boxes[valid_indices]
    # 将 pred_boxes_class 中的数字转换为类别名称
    pred_boxes_class_names = [class2name[class_id.item()] for class_id in valid_classes]

    # scene graph generation in json,output:logs/json

    # 初始化场景图信息
    scene_graph = {
        'nodes': [],
        'relations': []
    }

    # 构建每个节点的信息
    for idx, (class_name, box) in enumerate(zip(pred_boxes_class_names, valid_boxes)):
        node_info = {
            'index': int(valid_indices[idx].item()),
            'name': class_name,
            'location': box.tolist()
        }
        scene_graph['nodes'].append(node_info)

    # 构建节点之间的关系信息
    node_pairs_with_relations = []
    for i, (x, y) in enumerate(all_node_pairs):
        if x in valid_indices and y in valid_indices:
            name1 = class2name[pred_boxes_class[x].item()]
            name2 = class2name[pred_boxes_class[y].item()]
            # 找到 all_relation 中第 i 行的最大值及其索引
            max_value, max_index = all_relation[i].max(dim=0)

            if args.edge_model==1:
                relation_name = predicate2name[max_index.item()]
                node_pairs_with_relations.append({
                    'subject_name': name1,
                    'object_name': name2,
                    'relation_name': relation_name,
                    'subject_index1': int(x.item()),
                    'object_index2': int(y.item()),
                    'relation_score': max_value.item()
                })


            if args.edge_model==2:
                if max_value.item() > args.relation_threshold:
                    relation_name = predicate2name[max_index.item()]
                    node_pairs_with_relations.append({
                        'subject_name': name1,
                        'object_name': name2,
                        'relation_name': relation_name,
                        'subject_index1': int(x.item()),
                        'object_index2': int(y.item())
                    })

    if args.edge_model == 1:
        node_pairs_with_relations = sorted(node_pairs_with_relations, key=lambda x: x['relation_score'], reverse=True)[:args.relation_top_n]
    scene_graph['relations'] = node_pairs_with_relations

    # detection image ,output:logs/det
    if args.det:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        input_path = Path(image_path)
        input_filename = input_path.name
        output_detection_path = Path(args.output_dir_det) / input_filename
        (Path(args.output_dir_det)).mkdir(parents=True, exist_ok=True)

        # 绘制图像
        image = Image.open(image_path)
        plt.imshow(image)
        ax = plt.gca()

        # 为不同类别创建颜色映射
        unique_classes = list(set(pred_boxes_class_names))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        class_to_color = dict(zip(unique_classes, colors))

        # 绘制检测框
        for box, class_name in zip(valid_boxes, pred_boxes_class_names):
            x1, y1, x2, y2 = box.tolist()
            # 获取该类别对应的颜色
            color = class_to_color[class_name]
            # 绘制边界框,不显示标签文本
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False,
                                 edgecolor=color,
                                 linewidth=2)
            ax.add_patch(rect)

        plt.axis('off')
        plt.savefig(output_detection_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    if args.sgg:
        # 保存在logs/json
        filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(filename)[0]
        (Path(args.output_dir_sgg)).mkdir(parents=True, exist_ok=True)
        json_path = Path(args.output_dir_sgg)/(filename_without_ext + ".json")
        with open(json_path , 'w') as f:
            json.dump(scene_graph , f, indent=4)

    '''
    # 将 all_node_pairs 中的节点对转换为 (name1, name2, relation_name) 形式
    node_pairs_with_relations = []
    for i, (x, y) in enumerate(all_node_pairs):
        if x in valid_indices and y in valid_indices:
            name1 = class2name[pred_boxes_class[x].item()]
            name2 = class2name[pred_boxes_class[y].item()]
            # 找到 all_relation 中第 i 行的最大值及其索引
            max_value, max_index = all_relation[i].max(dim=0)
            if max_value.item() > 0.2:
                relation_name = predicate2name[max_index.item()]
                node_pairs_with_relations.append((name1, name2, relation_name))
    '''




    return scene_graph


def main(args):


    print("Loading config file from {}".format(args.config_file))

    #time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)


    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))


    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'),  color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))

    save_json_path = os.path.join(args.output_dir, "config_args_all.json")
    with open(save_json_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(save_json_path))

    logger.info("args: " + str(args) + '\n')


    # model

    device = torch.device(args.device)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    model_without_ddp = model
    if utils.get_rank() == 0:
        print("PostProcessors:", postprocessors)



    rln_proj = getattr(model, "rln_proj", None)
    rln_classifier = getattr(model, "rln_classifier", None)
    rln_freq_bias = getattr(model, "rln_freq_bias", None)

    if utils.get_rank() == 0:
        logger.info("rank:{}, rln_proj:{}, rln_classifier:{}, \
                     rln_freq_bias:{}".format(utils.get_rank(), \
                        rln_proj, rln_classifier, rln_freq_bias,
                        ))

    try:
        if utils.get_rank() == 0 and args.frozen_backbone:
            logger.info("Frozen backbone!")
    except:
        pass

    # 定义类别和谓词
    VG150_OBJ_CATEGORIES = 'airplane. animal. arm. bag. banana. basket. beach. bear. bed. bench. bike. bird. board. boat. book. boot. bottle. bowl. box. boy. branch. building. bus. cabinet. cap. car. cat. chair. child. clock. coat. counter. cow. cup. curtain. desk. dog. door. drawer. ear. elephant. engine. eye. face. fence. finger. flag. flower. food. fork. fruit. giraffe. girl. glass. glove. guy. hair. hand. handle. hat. head. helmet. hill. horse. house. jacket. jean. kid. kite. lady. lamp. laptop. leaf. leg. letter. light. logo. man. men. motorcycle. mountain. mouth. neck. nose. number. orange. pant. paper. paw. people. person. phone. pillow. pizza. plane. plant. plate. player. pole. post. pot. racket. railing. rock. roof. room. screen. seat. sheep. shelf. shirt. shoe. short. sidewalk. sign. sink. skateboard. ski. skier. sneaker. snow. sock. stand. street. surfboard. table. tail. tie. tile. tire. toilet. towel. tower. track. train. tree. truck. trunk. umbrella. vase. vegetable. vehicle. wave. wheel. window. windshield. wing. wire. woman. zebra.'

    VG150_PREDICATES = 'above. across. against. along. and. at. attached to. behind. belonging to. between. carrying. covered in. covering. eating. flying in. for. from. growing on. hanging from. has. holding. in. in front of. laying on. looking at. lying on. made of. mounted on. near. of. on. on back of. over. painted on. parked on. part of. playing. riding. says. sitting on. standing on. to. under. using. walking in. walking on. watching. wearing. wears. with.'
    # 定义 name2classes 和 name2predicates
    name2classes = OrderedDict([
        ('airplane', 1), ('animal', 2), ('arm', 3), ('bag', 4), ('banana', 5), ('basket', 6), ('beach', 7), ('bear', 8),
        ('bed', 9), ('bench', 10),
        ('bike', 11), ('bird', 12), ('board', 13), ('boat', 14), ('book', 15), ('boot', 16), ('bottle', 17),
        ('bowl', 18), ('box', 19), ('boy', 20),
        ('branch', 21), ('building', 22), ('bus', 23), ('cabinet', 24), ('cap', 25), ('car', 26), ('cat', 27),
        ('chair', 28), ('child', 29), ('clock', 30),
        ('coat', 31), ('counter', 32), ('cow', 33), ('cup', 34), ('curtain', 35), ('desk', 36), ('dog', 37),
        ('door', 38), ('drawer', 39), ('ear', 40),
        ('elephant', 41), ('engine', 42), ('eye', 43), ('face', 44), ('fence', 45), ('finger', 46), ('flag', 47),
        ('flower', 48), ('food', 49), ('fork', 50),
        ('fruit', 51), ('giraffe', 52), ('girl', 53), ('glass', 54), ('glove', 55), ('guy', 56), ('hair', 57),
        ('hand', 58), ('handle', 59), ('hat', 60),
        ('head', 61), ('helmet', 62), ('hill', 63), ('horse', 64), ('house', 65), ('jacket', 66), ('jean', 67),
        ('kid', 68), ('kite', 69), ('lady', 70),
        ('lamp', 71), ('laptop', 72), ('leaf', 73), ('leg', 74), ('letter', 75), ('light', 76), ('logo', 77),
        ('man', 78), ('men', 79), ('motorcycle', 80),
        ('mountain', 81), ('mouth', 82), ('neck', 83), ('nose', 84), ('number', 85), ('orange', 86), ('pant', 87),
        ('paper', 88), ('paw', 89), ('people', 90),
        ('person', 91), ('phone', 92), ('pillow', 93), ('pizza', 94), ('plane', 95), ('plant', 96), ('plate', 97),
        ('player', 98), ('pole', 99), ('post', 100),
        ('pot', 101), ('racket', 102), ('railing', 103), ('rock', 104), ('roof', 105), ('room', 106), ('screen', 107),
        ('seat', 108), ('sheep', 109), ('shelf', 110),
        ('shirt', 111), ('shoe', 112), ('short', 113), ('sidewalk', 114), ('sign', 115), ('sink', 116),
        ('skateboard', 117), ('ski', 118), ('skier', 119), ('sneaker', 120),
        ('snow', 121), ('sock', 122), ('stand', 123), ('street', 124), ('surfboard', 125), ('table', 126),
        ('tail', 127), ('tie', 128), ('tile', 129), ('tire', 130),
        ('toilet', 131), ('towel', 132), ('tower', 133), ('track', 134), ('train', 135), ('tree', 136), ('truck', 137),
        ('trunk', 138), ('umbrella', 139), ('vase', 140),
        ('vegetable', 141), ('vehicle', 142), ('wave', 143), ('wheel', 144), ('window', 145), ('windshield', 146),
        ('wing', 147), ('wire', 148), ('woman', 149), ('zebra', 150)
    ])
    name2predicates = {
        '[UNK]': 0, 'above': 1, 'across': 2, 'against': 3, 'along': 4, 'and': 5, 'at': 6, 'attached to': 7, 'behind': 8,
        'belonging to': 9,
        'between': 10, 'carrying': 11, 'covered in': 12, 'covering': 13, 'eating': 14, 'flying in': 15, 'for': 16,
        'from': 17, 'growing on': 18, 'hanging from': 19,
        'has': 20, 'holding': 21, 'in': 22, 'in front of': 23, 'laying on': 24, 'looking at': 25, 'lying on': 26,
        'made of': 27, 'mounted on': 28, 'near': 29,
        'of': 30, 'on': 31, 'on back of': 32, 'over': 33, 'painted on': 34, 'parked on': 35, 'part of': 36,
        'playing': 37, 'riding': 38, 'says': 39,
        'sitting on': 40, 'standing on': 41, 'to': 42, 'under': 43, 'using': 44, 'walking in': 45, 'walking on': 46,
        'watching': 47, 'wearing': 48, 'wears': 49, 'with': 50
    }

    # 创建 kw 字典
    kw = {
        "captions": [VG150_OBJ_CATEGORIES],
        "rel_captions": [VG150_PREDICATES]
    }

    model = model.cuda()

    checkpoint = torch.load(args.resume, map_location='cpu')
    missing, unexpected = model_without_ddp.load_state_dict(
        utils.clean_state_dict(checkpoint['model']), strict=False)
    model.eval()

    postprocessors['bbox'].rln_proj = rln_proj
    postprocessors['bbox'].rln_classifier = rln_classifier
    postprocessors['bbox'].rln_freq_bias = rln_freq_bias

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        postprocessors['bbox'].name2classes = name2classes
        do_sgg = getattr(postprocessors['bbox'], 'do_sgg', False)
        if do_sgg:
            postprocessors['bbox'].name2predicates = name2predicates




    # predict per image
    if args.model == 1:

        image_path = args.image_path
        target_image = Image.open(image_path).convert('RGB')
        width, height = target_image.size
        orig_target_sizes = torch.tensor([[height, width]], device=device)
        processed_image = preprocess_image(image_path,device=device)
        with torch.no_grad():
            outputs = model(processed_image, targets=None,**kw)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        # model results.png to scene graph generation
        scene_graph = results2scenegraph(results,name2classes,name2predicates)

    # predict local dataset
    if args.model == 2:
        from tqdm import tqdm
        files = [f for f in os.listdir(args.dataset) if f.endswith('.json')]
        for file in files:
            file_path = os.path.join(args.dataset, file)
            # 读取原始 JSON 文件
            with open(file_path, 'r') as f:
                data = json.load(f)

            new_data = []
            for entry in tqdm(data, desc=f"SGG,Processing {file}"):
                img_path = entry['img_path']
                # 设置 args.image_path
                current_directory = os.getcwd()
                root_directory = os.path.dirname(os.path.dirname(current_directory))
                img_path=img_path.lstrip('.')
                # 规范化路径
                relative_img_path = img_path.lstrip('/')
                img_path = os.path.join(root_directory, relative_img_path)
                args.image_path = img_path
                # 调用 main 函数生成场景图
                image_path = args.image_path
                target_image = Image.open(image_path).convert('RGB')
                width, height = target_image.size
                orig_target_sizes = torch.tensor([[height, width]], device=device)
                processed_image = preprocess_image(image_path, device=device)
                with torch.no_grad():
                    outputs = model(processed_image, targets=None, **kw)
                    results = postprocessors['bbox'](outputs, orig_target_sizes)

                # model results.png to scene graph generation
                scene_graph = results2scenegraph(results, name2classes, name2predicates)
                # 将场景图数据合并到原始条目中
                entry['scene_graph'] = scene_graph
                # 添加到新的数据列表
                new_data.append(entry)

            # 保存新的 JSON 文件
            new_file_path = os.path.join(args.dataset, f"new_{file}")
            with open(new_file_path, 'w') as f:
                json.dump(new_data, f, indent=4)

            print(f"Processed and saved {new_file_path}")



#已经知道valid_boxes={Tensor：（5，4）}，
    return

#





if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

