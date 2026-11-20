"""声纹漂移分界：纯计算，不接触数据库。

输入是按班次顺序排列的、每个班次内已排序的频谱质心数组。每个分段以
该段全部样本的中位数为基准，成本为该段样本到中位数的绝对差之和；每道
分界再加固定罚分。求总成本最低的分法，每段至少 ``min_shifts`` 个班次。

同分时逐个比较分界位置，取靠前（字典序更小）的一组；若一组分界是另一
组的前缀，分界更少的一组视为更靠前。

实现只保存「当前段」的一个合并数组与长度 O(班次数) 的成本表，因此
不会枚举全部 2^(k-1) 种分法，也不会缓存 O(k^2) 个候选区间。
"""
import numpy as np

MIN_SHIFTS = 3
ABS_TOL = 1e-9
REL_TOL = 1e-9


class DriftError(ValueError):
    """输入无法构成满足业务约束的分界。"""


def _tolerance(a, b):
    return ABS_TOL + REL_TOL * max(abs(a), abs(b))


def _close(a, b):
    return abs(a - b) <= _tolerance(a, b)


def merge_sorted(left, right):
    """把两个升序数组合并为新的升序数组，O(len(left)+len(right))。"""
    if left.size == 0:
        return np.array(right, dtype=np.float64, copy=True)
    if right.size == 0:
        return left
    positions = np.searchsorted(left, right, side='right') + np.arange(right.size)
    merged = np.empty(left.size + right.size, dtype=np.float64)
    free = np.ones(merged.size, dtype=bool)
    free[positions] = False
    merged[positions] = right
    merged[free] = left
    return merged


def segment_stat(sorted_values):
    """返回（到本段中位数的绝对差之和, 中位数）。

    偶数个样本时中位数取中间两个值的平均，与统计口径一致。
    """
    n = sorted_values.size
    if n == 0:
        return 0.0, None
    if n % 2:
        median = float(sorted_values[n // 2])
    else:
        median = float((sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0)
    deviation = float(np.abs(sorted_values - median).sum())
    return deviation, median


def compute_suffix_costs(blocks, penalty, min_shifts=MIN_SHIFTS, state=None, on_checkpoint=None):
    """F[i]：第 i 个班次起到末尾的最优总成本（含其后的分界罚分）。

    每个 i 从后往前增量合并，仅保留一个当前区间数组；一旦当前段的
    纯偏差已经超过已知最优总成本，更长的段只会更大，可以安全停止。
    同时记录整段 [i, k-1] 的偏差/中位数，供重建阶段使用。
    """
    k = len(blocks)
    if state is None:
        best_cost = [None] * k
        full_stat = [None] * k
        start = k - 1
    else:
        best_cost = list(state['costs'])
        full_stat = [None if x is None else tuple(x) for x in state['full']]
        start = int(state['next'])
        if len(best_cost) != k or len(full_stat) != k:
            raise DriftError('续跑状态与班次数不一致')
    for i in range(start, -1, -1):
        if k - i < min_shifts:
            continue
        values = np.empty(0, dtype=np.float64)
        best = None
        for e in range(i, k):
            values = merge_sorted(values, blocks[e])
            if e - i + 1 < min_shifts:
                continue
            deviation, median = segment_stat(values)
            if e == k - 1:
                full_stat[i] = (deviation, median, int(values.size))
            if best is not None and deviation > best + _tolerance(deviation, best):
                break
            if e < k - 1:
                if best_cost[e + 1] is None:
                    continue
                total = deviation + penalty + best_cost[e + 1]
            else:
                total = deviation
            if best is None or total < best - _tolerance(total, best):
                best = total
        best_cost[i] = best
        if on_checkpoint is not None:
            on_checkpoint(i, best, full_stat[i])
    return best_cost, full_stat


def solve(blocks, penalty, min_shifts=MIN_SHIFTS):
    """求最低总成本的分界。

    参数
    ----
    blocks: 按班次顺序排列、每项为该班次内升序频谱质心数组（非空）。
    penalty: 每道分界的固定罚分（非负有限数）。
    min_shifts: 每段至少包含的班次数，默认 3。

    返回字典包含 total、boundaries、segments 等；不返回任何候选区间。
    """
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise DriftError('罚分必须是非负有限数值')
    penalty = float(penalty)
    if not np.isfinite(penalty) or penalty < 0:
        raise DriftError('罚分必须是非负有限数值')
    if isinstance(min_shifts, bool) or int(min_shifts) != min_shifts:
        raise DriftError('每段班次数必须为整数')
    min_shifts = int(min_shifts)
    if min_shifts < 2:
        raise DriftError('每段至少需要 2 个班次')

    prepared = []
    for index, block in enumerate(blocks):
        array = np.sort(np.asarray(block, dtype=np.float64), kind='stable')
        if array.size == 0:
            raise DriftError(f'第 {index + 1} 个班次没有可参与计算的样本')
        if not np.all(np.isfinite(array)):
            raise DriftError(f'第 {index + 1} 个班次包含非有限频谱质心')
        prepared.append(array)
    k = len(prepared)
    if k < min_shifts:
        raise DriftError(f'班次数 {k} 不足以按每段至少 {min_shifts} 班切分')

    best_cost, full_stat = compute_suffix_costs(prepared, penalty, min_shifts)
    if best_cost[0] is None:
        raise DriftError(f'班次数 {k} 不足以按每段至少 {min_shifts} 班切分')

    return reconstruct(prepared, penalty, min_shifts, best_cost, full_stat)


def _segment_values(blocks, start, end):
    values = np.empty(0, dtype=np.float64)
    for e in range(start, end + 1):
        values = merge_sorted(values, blocks[e])
    return values


def reconstruct(blocks, penalty, min_shifts, best_cost, full_stat=None):
    """由后缀成本表重建分界；同分时取分界靠前（字典序更小）的一组。

    每到一个段首，先比较「到此为止」与最优成本：相同则不设分界（更少的
    分界是任何后继分界表的前缀，视为更靠前）；否则取能让剩余部分达到最优
    的最小分界位置。整段统计按需现算，不依赖缓存，保证续跑结果一致。
    """
    k = len(blocks)
    prepared = blocks
    if best_cost[0] is None:
        raise DriftError(f'班次数 {k} 不足以按每段至少 {min_shifts} 班切分')
    boundaries = []
    segments = []
    start = 0
    while start < k:
        whole = full_stat[start] if full_stat is not None else None
        if whole is not None and _close(whole[0], best_cost[start]):
            # 到此为止即是同分中分界最少的一组，不再向后切分。
            segments.append({
                'start': start, 'end': k - 1,
                'shift_count': k - start,
                'sample_count': int(whole[2]),
                'median': whole[1],
                'deviation': whole[0],
            })
            break
        values = np.empty(0, dtype=np.float64)
        chosen = None
        for e in range(start, k - 1):
            values = merge_sorted(values, prepared[e])
            if e - start + 1 < min_shifts or best_cost[e + 1] is None:
                continue
            deviation, median = segment_stat(values)
            if _close(deviation + penalty + best_cost[e + 1], best_cost[start]):
                chosen = (e, deviation, median, int(values.size))
                break
            if deviation > best_cost[start] + _tolerance(deviation, best_cost[start]):
                break
        if chosen is None:
            raise DriftError('无法重建满足约束的分界，请核对输入')
        e, deviation, median, count = chosen
        segments.append({
            'start': start, 'end': e,
            'shift_count': e - start + 1,
            'sample_count': count,
            'median': median,
            'deviation': deviation,
        })
        boundaries.append(e)
        start = e + 1

    return {
        'total': float(best_cost[0]),
        'deviation_total': float(sum(s['deviation'] for s in segments)),
        'penalty_total': float(penalty * len(boundaries)),
        'boundaries': boundaries,
        'segments': segments,
        'penalty': penalty,
        'min_shifts': min_shifts,
        'shift_total': k,
        'sample_total': int(sum(b.size for b in prepared)),
    }
