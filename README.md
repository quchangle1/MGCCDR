# MGCCDR
The implementation for SIGIR 2025: Bridging Short Videos and Streamers with Multi-Graph Contrastive Learning for Live Streaming Recommendation.

## How to run the code
To train MGCCDR on Doubanbook dataset with GPU 0, simply run:
 > python train.py -g 0 -m MGCCDR -d Doubanbook

## Environment

Our experimental environment is shown below:

```
numpy version: 1.24.4
pandas version: 2.0.3
torch version: 2.4.1
```

## Citation

If you find our code or work useful for your research, please cite our work.

```
@inproceedings{qu2025bridging,
  title={Bridging Short Videos and Streamers with Multi-Graph Contrastive Learning for Live Streaming Recommendation},
  author={Qu, Changle and Zhao, Liqin and Niu, Yanan and Zhang, Xiao and Xu, Jun},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2059--2069},
  year={2025}
}
```
