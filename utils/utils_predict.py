from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr


def compute_score(score,gt_score):
    # 计算 SROCC
    srocc, _ = spearmanr(score.squeeze(), gt_score.squeeze())

    # 计算 PLCC
    plcc = pearsonr(score.squeeze(), gt_score.squeeze())

    # 计算 KROCC
    krocc, _ = kendalltau(score.squeeze(), gt_score.squeeze())

    # 打印结果
    print("KROCC:", krocc)
    print("SROCC:", srocc)
    print("PLCC:", plcc[0])

    return {
        "srocc": srocc,
        "plcc": plcc[0],
        "krocc": krocc
    }