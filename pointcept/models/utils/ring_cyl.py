import math
import torch


@torch.inference_mode()
def ring_cyl_rank_encode(
    grid_coord: torch.Tensor,
    batch: torch.Tensor = None,
    gr: int = 20,
    gz: int = 40,
    theta_offset_deg: float = 0.0,
    center_mode: str = "bbox",
) -> torch.Tensor:
    """
    Ring Cylindrical rank encoding

    输入:
        grid_coord: (N, 3), int tensor
        batch:      (N,), optional
        gr:         半径层宽（单位: grid cell）
        gz:         高度层厚（单位: grid cell）
        theta_offset_deg: 角度起始偏移
        center_mode:
            - "bbox"
            - "mean"
            - "origin"

    输出:
        code: (N,), long tensor
              每个点对应排序名次 rank = 0,1,2,...,N-1

    排序规则:
        1) 先按 rq（半径层）
        2) 再按 zq（高度层）
        3) 最后在同一 (rq, zq) 内按 theta 排序
    """
    assert grid_coord.dim() == 2 and grid_coord.size(1) == 3
    assert gr > 0 and gz > 0

    device = grid_coord.device
    n = grid_coord.size(0)

    if batch is None:
        batch = torch.zeros(n, device=device, dtype=torch.long)
    else:
        batch = batch.long()

    theta_offset = theta_offset_deg / 180.0 * math.pi

    code = torch.empty(n, device=device, dtype=torch.long)
    unique_batch = torch.unique(batch, sorted=True)

    base = 0
    for b in unique_batch:
        mask = batch == b
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        xyz = grid_coord[idx].to(torch.float32)  # (M, 3)

        # 1) 选中心
        if center_mode == "bbox":
            xy_min = xyz[:, :2].min(dim=0).values
            xy_max = xyz[:, :2].max(dim=0).values
            center_xy = 0.5 * (xy_min + xy_max)
        elif center_mode == "mean":
            center_xy = xyz[:, :2].mean(dim=0)
        elif center_mode == "origin":
            center_xy = torch.zeros(2, device=device, dtype=xyz.dtype)
        else:
            raise ValueError(
                f"Unsupported center_mode={center_mode}, "
                f"choose from ['bbox', 'mean', 'origin']"
            )

        # 2) 圆柱坐标
        dx = xyz[:, 0] - center_xy[0]
        dy = xyz[:, 1] - center_xy[1]

        r = torch.sqrt(dx * dx + dy * dy)

        theta = torch.atan2(dy, dx)                    # [-pi, pi]
        theta = torch.remainder(theta, 2.0 * math.pi) # [0, 2pi)
        theta = torch.remainder(theta + theta_offset, 2.0 * math.pi)

        z = xyz[:, 2]
        z0 = z.min()

        # 3) 只离散 r 和 z
        rq = torch.floor(r / float(gr)).long()
        zq = torch.floor((z - z0) / float(gz)).long()
 
        # 4) 排序: rq -> zq -> theta
        local = torch.arange(idx.numel(), device=device)

        # 先按 theta
        perm = torch.argsort(theta[local], stable=True)
        local = local[perm]

        # 再按 zq
        perm = torch.argsort(zq[local], stable=True)
        local = local[perm]

        # 最后按 rq
        perm = torch.argsort(rq[local], stable=True)
        local = local[perm]

        sorted_idx = idx[local]

        # 5) 生成全局连续 rank code
        code[sorted_idx] = torch.arange(
            base, base + sorted_idx.numel(), device=device, dtype=torch.long
        )
        base += sorted_idx.numel()

    return code