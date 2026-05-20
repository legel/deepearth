![DeepEarth logo](https://github.com/legel/deepearth/blob/main/docs/deepearth_logo.png)

# DeepEarth：行星科学与可持续发展的 AI 基础模型

DeepEarth 是一个**自监督、多模态、时空 GeoAI 模型**，面向全球环境智能与优化。

DeepEarth 通过联合重建多模态掩码数据来学习（如上图所示）。它使用一种新颖的空间-时间位置编码器 **Earth4D**，专为地球观测数据设计。

![Earth4D 时空编码器](https://github.com/legel/deepearth/blob/main/docs/earth4d_spacetime_encoder.png)

---

## 最新动态

- **2026年3月7日** — **论文上线 arXiv。** [《具有4D时空嵌入的自监督多模态世界模型》](https://arxiv.org/abs/2603.07039) 经过 [2026 世界建模研讨会](https://world-model-mila.github.io/) 同行评审，现已在 arXiv 发布。

- **2026年1月28日** — **世界建模研讨会海报展示。** [Lance Legel](https://www.linkedin.com/in/legel/) 和 [Qin Huang](https://news.asu.edu/b/20250512-asu-phd-student-tackles-climate-change-and-extreme-weather) 在 [2026 世界建模研讨会](https://world-model-mila.github.io/) 展示了 DeepEarth。

- **2025年12月22日** — **10 倍加速。** [Brandon Voelker](https://www.egr.uh.edu/news/202410/space-ground-%E2%80%93-phd-student-voelker-leads-team-transforming-remote-sensing-based) 在小批量实验后，[Lance Legel](https://www.linkedin.com/in/legel/) 将小批量处理速度提升了 10 倍。

- **2025年12月19日** — **超算奖。** 美国能源部 NERSC 授予 DeepEarth 团队 2026 年超算访问权限。

- **2025年12月2日** — **顶会演讲。** 被 [2026 世界建模研讨会](https://world-model-mila.github.io/) 接收，与 [Yoshua Bengio](https://yoshuabengio.org/) 和 [Yann LeCun](http://yann.lecun.com/) 主旨演讲同台。

- **2025年11月17日** — **99% 参数缩减，4× 加速。** [Earth4D](https://github.com/legel/deepearth/tree/main/encoders/xyzt) 结合[学习哈希探针](https://arxiv.org/abs/2312.17241)，仅用 5M 参数即达到惊人精度。

- **2025年11月16日** — **时空编码器误差降低 23%。** [Lance Legel](https://www.linkedin.com/in/legel/) 和 [Qin Huang](https://news.asu.edu/b/20250512-asu-phd-student-tackles-climate-change-and-extreme-weather) 在生态预测基准上实现最先进的 R²。

- **2025年10月29日** — **预测火灾风险。** 通过 NSF 的 [地理空间理解研究所](http://i-guide.io/) 展示植被活体含水量模拟。

---

## 核心创新

### 深度贝叶斯模拟
DeepEarth 是一个深度神经网络，学习回答经典的贝叶斯问题，例如："当变量 **α** 在时空上变化时，在给定所有可用证据的情况下，变量 **β** 最可能如何变化？"

### 最大化行星似然
遵循 Google DeepMind 的[数学证明](https://proceedings.mlr.press/v37/germain15.html)，DeepEarth 学习跨时空的真实世界数据的**最可能统计模型**。它跨 (_x_, _y_, _z_, _t_, _energy_) 度量进行学习。

### 收敛式科学建模
大量 DeepEarth 模型可针对不同科学领域进行训练：每个模型只需输入特定领域的数据集，即可自动学习深度归纳先验。

### 物理模拟器 + 基础模型
DeepEarth 模型作为跨时空物理模拟器训练（例如从历史数据预测火灾风险）。模拟器也可微调用于特定应用，类似于从 _GPT_ 到 _ChatGPT_ 的演进。

### 深层时空流形
爱因斯坦相对论的伟大启示之一是**空间**和**时间**并非独立变量。Earth4D 扩展了 NVIDIA 的[3D 多分辨率哈希编码](https://nvlabs.github.io/instant-ngp/)，以学习时空分布。

---

## 快速开始

```python
from deepearth.encoders.xyzt.earth4d import Earth4D

# 初始化世界模型
world_model = Earth4D()

# 输入 (纬度, 经度, 海拔, 时间) → 输出 192 维时空特征
embeddings = world_model(
    (51.9976, -0.7416, 110, "1941-06-01 09:00 GMT"),    # 布莱切利园
    (40.4433, -79.9436, 270, "1985-01-15 10:00 ET"),    # 卡内基梅隆
    (46.2330, 6.0557, 430, "1989-03-12 10:00 CET"),     # CERN
    (45.5308, -73.6128, 63, "2026-02-04 11:00 ET"),     # Mila, 魁北克
)
# embeddings.shape: [4, 192] — 可训练的时空特征
```

## 安装

```bash
git clone https://github.com/legel/deepearth.git
cd deepearth
pip install torch ninja
cd encoders/xyzt/hashencoder && python setup.py build_ext --inplace
```

> **注意：** 哈希编码器需要 CUDA Toolkit 编译 GPU 内核。

---

## 应用场景

- 🌲 **植被含水量预测** — 野火风险评估（5M 参数超越 500M+ 参数通用模型）
- 🌡️ **气候变化模拟** — 时空环境变量预测
- 🛰️ **地球观测分析** — 多模态卫星数据融合
- 📊 **生态预测** — Caravan 基准上的最先进性能

---

## 引用

```bibtex
@article{legel2026deepearth,
  title={Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding},
  author={Legel, Lance and Huang, Qin and Voelker, Brandon},
  journal={arXiv preprint arXiv:2603.07039},
  year={2026}
}
```

## 贡献

欢迎协作者。联系：lance@ecodash.ai

---

## 许可证

查看 [LICENSE](LICENSE) 文件。
