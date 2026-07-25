# UniAD.c：从像素到规划，一条可检查的路径

交互文章完整拆解 UniAD 的 planning-oriented 设计：六路图像如何进入时序
BEV，TrackFormer 与 MapFormer 如何产生可继续传递的 query，MotionFormer
如何建模 agent–agent / agent–map / agent–goal 交互，OccFormer 如何将稀疏
对象未来重新写回稠密风险场，以及 Planner 如何把 SDC query、导航命令和
occupancy 收束为六个 ego waypoints。

仓库同时提供一个可运行的 C11 合成垂直切片。它把生产图的表示流缩小为固定
容量：`6×3×8×8` 相机平面、`8×8×16` 时序 BEV、64 个候选到稳定 top-8
tracks、`3×4` 多模态未来、三帧 occupancy 和六点 ego plan。它验证的是接口、
时序状态和结果契约，不是发布权重的数值复现。

正确性不能跳级：容器测试证明输入安全；算子 fixtures 与两帧 PyTorch oracle
证明合成图等价；二者都不能证明 production checkpoint 等价或 nuScenes 任务
精度。完整图解、production / tiny profile 对照、operator ledger 与 canonical
结果可视化请打开 [交互文章](index.html)。
