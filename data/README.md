数据不入 Git。首选数据已具备时整理为 manifest.csv；否则可执行：

    python scripts/download_pascal.py --output data

脚本使用 UIUC 官方网站图片目录类别，固定取第一条英文描述。
数据源：https://vision.cs.uiuc.edu/pascal-sentences/
描述遵循官方页面原始授权说明；图片遵循原始数据条款，勿改称自有数据。

自有 Wikipedia / Pascal 版本必须提供真实单类别标签：

```csv
sample_id,image_path,text,label
000001,images/000001.jpg,"A dog playing on grass",dog
```

image_path 相对 manifest.csv 所在目录。CSV 以 UTF-8 保存；逗号、换行请按 CSV 标准引用。
一图只保留一条描述；sample_id 唯一；多标签 VOC 数据必须先确定并记录单类别规则，不能直接混用。
验证时会剔除空字段、坏图、内容完全相同的重复图片，记录 splits/rejected.csv。
需人工抽查图片和文本是否配对，程序无法验证语义标注本身正确。
