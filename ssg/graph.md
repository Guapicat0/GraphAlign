For visual graph, you can download the model checkpoint:
### 📥 Checkpoints (OvD+R-SGG)

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@20/50/100 (Joint)</th>
      <th>R@20/50/100 (Novel Object)</th>
      <th>R@20/50/100 (Novel Relation)</th>
      <th>Checkpoint</th>
      <th>Config</th>
      <th>Pre-trained Checkpoint</th>
      <th>Pre-trained Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Swin-T</td>
      <td>10.02 / 13.50 / 16.37</td>
      <td>10.56 / 14.32 / 17.48</td>
      <td>7.09 / 9.19 / 11.18</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swint.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovdr.py</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-pretrain-coco-swint.pth"><s>link</s></a></td>
      <td>config/GroundingDINO_SwinT_OGC_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-B</td>
      <td>12.37 / 17.14 / 21.03</td>
      <td>12.63 / 17.58 / 21.70</td>
      <td>10.56 / 14.62 / 18.22</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swinb.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovdr.py</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-pretrain-coco-swinb.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-T (pretrained on MegaSG)</td>
      <td>10.67 / 15.15 / 18.82</td>
      <td>8.22 / 12.49 / 16.29</td>
      <td>9.62 / 13.68 / 17.19</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swint-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovdr.py</td>
      <td><s>link</s></td>
      <td>config/GroundingDINO_SwinT_OGC_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-B (pretrained on MegaSG)</td>
      <td>12.54 / 17.84 / 21.95</td>
      <td>10.29 / 15.66 / 19.84</td>
      <td>12.21 / 17.15 / 21.05</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swinb-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovdr.py</td>
      <td><s>link</s></td>
      <td>config/GroundingDINO_SwinB_pretrain.py</td>
    </tr>
  </tbody>
</table>

When preparing your dataset, each sample should be represented as a JSON object containing the image path, prompt, score, and associated graphs.

# Graph Format

## Example Structure

```json
{
  "img_path": "../datasets/AGIQA-3K/image/AttnGAN_normal_000.jpg",
  "prompt": "statue of a man",
  "gt_score": 0.0908,
  "scene_graph": {
    "nodes": [
      {
        "index": 0,
        "name": "elephant",
        "location": [0.37, 3.08, 309.71, 482.43]
      },
      {
        "index": 1,
        "name": "hat",
        "location": [162.06, 386.57, 385.69, 511.04]
      }
    ],
    "relations": [
      {
        "subject_name": "elephant",
        "object_name": "hat",
        "relation_name": "has",
        "subject_index1": 0,
        "object_index2": 1,
        "relation_score": 0.207
      }
    ]
  },
  "GT_graph": {
    "nodes": [
      { "index": 0, "name": "statue" },
      { "index": 1, "name": "man" }
    ],
    "relations": [
      {
        "subject_name": "statue",
        "object_name": "man",
        "relation_name": "created_by",
        "subject_index1": 0,
        "object_index2": 1
      }
    ]
  },
  "semantic_graph": [
    {
      "gt_index": 1,
      "gt_name": "man",
      "node_index": 1,
      "node_name": "hat",
      "similarity": 1.0,
      "relation_name": "RelatedTo",
      "relation_confidence": 2.93
    }
  ]
}
```

## Notes
- **`img_path`**: Path to the image in your dataset.  
- **`prompt`**: The text prompt associated with the image.  
- **`gt_score`**: Ground truth score or quality score for evaluation.  
- **`scene_graph`**: Visual graph with `nodes` (objects + bounding boxes) and `relations` (object-object relations).  
- **`GT_graph`**: Ground-truth graph annotations for evaluation.  
- **`semantic_graph`**: Mapping between predicted and ground-truth nodes with similarity scores.  

