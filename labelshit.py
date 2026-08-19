from torch.utils.data import Subset
from collections import Counter
import numpy as np

def build_two_disjoint_label_shift_subsets(dataset, dist_a, size_a, dist_b, size_b, seed=0):
    """
    从同一个数据集里构造两个不重叠的 label-shift 子集
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(dist_a)

    dist_a = np.array(dist_a, dtype=float)
    dist_b = np.array(dist_b, dtype=float)
    dist_a = dist_a / dist_a.sum()
    dist_b = dist_b / dist_b.sum()

    count_a = (dist_a * size_a).astype(int)
    count_b = (dist_b * size_b).astype(int)

    # 补齐取整误差
    for i in range(size_a - count_a.sum()):
        count_a[i % num_classes] += 1
    for i in range(size_b - count_b.sum()):
        count_b[i % num_classes] += 1

    indices_a = []
    indices_b = []

    for c in range(num_classes):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)

        needed = count_a[c] + count_b[c]
        if needed > len(cls_idx):
            raise ValueError(
                f"类 {c} 总共需要 {needed} 个样本，但只有 {len(cls_idx)} 个。"
            )

        idx_a = cls_idx[:count_a[c]]
        idx_b = cls_idx[count_a[c]:count_a[c] + count_b[c]]

        indices_a.extend(idx_a.tolist())
        indices_b.extend(idx_b.tolist())

    rng.shuffle(indices_a)
    rng.shuffle(indices_b)

    return Subset(dataset, indices_a), Subset(dataset, indices_b)

def show_class_distribution(subset, full_dataset):
    labels = [full_dataset.targets[i] for i in subset.indices]
    counter = Counter(labels)
    total = len(labels)

    for c in range(10):
        cnt = counter[c]
        print(f"class {c}: {cnt} ({cnt/total:.4f})")



def label_shift_subsets(dataset, dist_a, size_a, seed=0):
    """
    从同一个数据集里构造两个不重叠的 label-shift 子集
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(dist_a)

    dist_a = np.array(dist_a, dtype=float)
    dist_a = dist_a / dist_a.sum()

    count_a = (dist_a * size_a).astype(int)

    # 补齐取整误差
    for i in range(size_a - count_a.sum()):
        count_a[i % num_classes] += 1

    indices_a = []

    for c in range(num_classes):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)

        needed = count_a[c]
        if needed > len(cls_idx):
            raise ValueError(
                f"类 {c} 总共需要 {needed} 个样本，但只有 {len(cls_idx)} 个。"
            )

        idx_a = cls_idx[:count_a[c]]
        indices_a.extend(idx_a.tolist())

    rng.shuffle(indices_a)

    return Subset(dataset, indices_a)