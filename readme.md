## 基于CIFAR10 MAE的实现

由于可用资源有限，本项目仅在 cifar10 上测试模型。主要想重现这样的结果：**使用 MAE 预训练 ViT 可以比直接使用标签进行监督学习训练获得更好的结果**。这应该是**自我监督学习比监督学习更有效的数据**的证据。

主要遵循论文中的实现细节。但是，由于 Cifar10 和 ImageNet 的区别，做了一些修改：

- 使用 vit-tiny 而不是 vit-base。
- 由于 Cifar10 只有 50k 训练数据，将 pretraining epoch 从 400 增加到 2000，将 warmup epoch 从 40 增加到 200。我们注意到，在 2000 epoch 之后损失仍在减少。
- 使用Hugging Face accelerate,引入混合精度训练，加速训练，减少现存
- 使用Auto augment ,减轻过拟合




### Run

首先进行预训练

```python
# pretrained with mae
python mae_trainer.py --yaml_path $YOUR_YAML_PATH
```

训练未用MAE的分类器，也就是从头开始训练分类器

```
# train classifier from scratch
python mae_finetune_trainer.py
```

利用训练好的MAE的encoder作为输入，构建的分类模型作为分类器

```python
# train classifier from pretrained model
python mae_trainer.py --pretrained_model_path $YOUR_PRETRAIN_MAE_PATH --yaml_path $YOUR_YAML_PATH
```


### Result

|Model| Validation Acc |
|-----|----------------|
|ViT-T w/o pretrain| 86.13          |
|ViT-T w/  pretrain| **95.77**      |



### Reference

该项目主要基于 https://github.com/IcarusWizard/MAE 进行了一些翻译和注释，同时还加入了其他的一些功能。

- https://github.com/IcarusWizard/MAE
- https://github.com/facebookresearch/mae