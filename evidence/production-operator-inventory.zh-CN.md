# UniAD 生产算子与 Tensor 生命周期清单

## 证据边界

本清单由真实 nuScenes mini 六相机样本的 PyTorch oracle 追踪生成。`cuda_elapsed_ms` 是嵌套 module CUDA Event 跨度，会重复计时；由于当前机器 CUPTI 返回 `CUPTI_ERROR_INVALID_DEVICE`，它不是 kernel 时间，也不可相加后当作端到端延迟。清单证明调用、shape、dtype、布局和 autograd 保存关系，不证明纯 CUDA 图等价或数据集精度。

## 总览

- 推理 module 调用：2230
- 训练 module 调用：5974
- 唯一 operator/shape/layout 签名：710
- 算子族：64
- checkpoint keys：2459
- 有观测消费者的 checkpoint keys：2451
- saved tensor 匹配生命周期：5226
- saved tensor 事件估算峰值：14.220 GiB （不含参数、workspace、allocator 碎片）

## 阶段调用

| 阶段 | 推理调用 | 训练调用 | 推理输出 GiB（累计流量） | 训练输出 GiB（累计流量） |
|---|---:|---:|---:|---:|
| bev_encoder | 157 | 1099 | 8.319 | 58.081 |
| image_backbone | 378 | 2268 | 38.847 | 271.926 |
| image_neck | 15 | 90 | 0.878 | 6.143 |
| mapping | 633 | 714 | 9.065 | 7.415 |
| motion | 311 | 322 | 0.163 | 0.192 |
| occupancy | 349 | 353 | 2.955 | 2.963 |
| planning | 59 | 63 | 0.382 | 0.382 |
| root | 1 | 1 | 0.008 | 0.000 |
| support | 2 | 43 | 0.000 | 0.000 |
| tracking | 325 | 1021 | 0.512 | 1.705 |

## 算子族

| 算子族 | 唯一签名数 |
|---|---:|
| BEVFormerEncoder | 1 |
| BEVFormerLayer | 2 |
| BaseTransformerLayer | 2 |
| BevFeatureSlicer | 1 |
| Block | 4 |
| Bottleneck | 19 |
| CVT_Decoder | 1 |
| CVT_DecoderBlock | 2 |
| CollisionLoss | 1 |
| DetectionTransformerDecoder | 3 |
| DetrTransformerDecoderLayer | 9 |
| DetrTransformerEncoder | 1 |
| DiceLoss | 2 |
| DiceLossWithMasks | 2 |
| Dropout2d | 3 |
| Embedding | 1 |
| FFN | 9 |
| FPN | 2 |
| FieryBinarySegmentationLoss | 2 |
| FocalLoss | 6 |
| GIoULoss | 2 |
| GridMask | 2 |
| Identity | 12 |
| IntentionInteraction | 2 |
| L1Loss | 4 |
| LearnedPositionalEncoding | 1 |
| LogSoftmax | 2 |
| MLP | 4 |
| MapInteraction | 2 |
| MemoryBank | 1 |
| Mlp | 3 |
| MotionHead | 2 |
| MotionTransformerDecoder | 2 |
| OccHead | 2 |
| PansegformerHead | 1 |
| PlanningHeadSingleMode | 2 |
| PlanningLoss | 1 |
| QueryInteractionModule | 1 |
| ResLayer | 8 |
| ResNet | 2 |
| SegMaskHead | 4 |
| Sequential | 82 |
| SinePositionalEncoding | 1 |
| TrackAgentInteraction | 2 |
| TrajLoss | 1 |
| TransformerDecoder | 1 |
| TransformerDecoderLayer | 5 |
| TransformerEncoderLayer | 2 |
| Unflatten | 2 |
| UniAD | 2 |
| Upsample | 3 |
| UpsamplingAdd | 1 |
| activation | 69 |
| attention | 15 |
| conv2d | 70 |
| conv2d_epilogue | 15 |
| convolution | 1 |
| deformable_attention_or_conv | 18 |
| dropout | 46 |
| layer_norm | 32 |
| linear | 144 |
| multihead_attention | 30 |
| normalization | 30 |
| pooling | 5 |

## 机器可读细节

同目录 `production-operator-inventory.json` 包含每个签名的完整输入/输出 shape、stride、dtype、layout、字节数、梯度需求、源码位置、module 集合、训练/推理调用数和 CUDA Event 分布；还包含最长/最大的 saved tensor 生命周期以及逐 checkpoint key 的运行时消费者。
