"""出售小红: 藏品仓库普通视图里扫描全部红色藏品, 按单件价格逐件出售。

交互模型移植自参考实现 (按键精灵 即可落槌.txt 的命名藏品出售): 仓库普通视图
(不点出售圆钮)下点击网格里的藏品卡片弹出详情面板, 面板里有价格与「出售」
按钮; 参考实现在同一点连点两次(第二次为确认), 这里第一次点击后先找「确认」
OCR 命中框, 找不到再补点同位置, 与出价确认的既有模式一致。

红色藏品靠卡片底色识别: 卡框颜色 2026-10-28 用户仓库截图标定为
RGB(105,9,34) 且高度一致(见 RED_CARD_COLOR_RANGE 注释)。扫描从网格顶部
开始, 每屏: 颜色掩码 → 连通域聚类取卡片中心 → 逐件点开比价; 出售成功后
列表会回流, 重新扫当前屏; 跳过(保留)的件按「详情面板读到的藏品名」记名
去重 —— 同名藏品同价, 跳过一次后同名件不再点开。

遍历屏数是固定的, 不做视觉「到底」判定: 仓库大量空格子时滚动前后画面几乎
不变, 模板匹配会误判「没动」而提前收尾 (2026-10-28 实测); 到底后的空滚
一屏约 1 秒, 代价可接受。扫描自带独立预算 (RED_SCAN_BUDGET), 不吃轮次
deadline, 预算耗尽按已完成件数收尾。

模块函数的第一个参数 task 是 AutoBidAuctionTask 实例, 允许访问面由
contracts.AuctionSellOps 窄协议声明 (本模块复用出售域的协议, 额外用到的
frame 已在协议内声明)。
"""

import cv2
import numpy as np
from ok import Box, TaskDisabledException

from src.tasks.auction import interaction as auction_interaction
from src.tasks.auction.contracts import AuctionSellOps
from src.tasks.auction.layout import (
    POS_ITEM_DETAIL_CLOSE,
    POS_ITEM_SELL,
    POS_WAREHOUSE_GRID_SCROLL,
    RE_CONFIRM,
    RE_ITEM_SELL,
    RE_POPUP_CLOSE_HINT,
    AuctionBoxes,
)
from src.utils import clock
from src.utils import image_utils as iu

# 红色藏品卡片主色: 2026-10-28 用户仓库截图标定, 卡框颜色高度一致
# RGB(105,9,34) (截图 22.5% 像素落在 +-12 内), 每通道取宽容差; 背景(灰/棕,
# R~42-77 且 r-g 小)、空格子(近黑)、滚动条都在范围外。物品图标自身的红色
# (如宝石)亮于上限或与卡框连通成同一簇, 都不影响卡片级聚类。
RED_CARD_COLOR_RANGE = {
    "r": (80, 140),
    "g": (0, 45),
    "b": (0, 70),
}
# 连通域面积达到网格面积的这个比例才算一张红色卡片: 过滤零星高亮与描边;
# 另有绝对下限, 防超小网格把真卡片滤掉。
RED_CLUSTER_MIN_FRACTION = 0.0015
RED_CLUSTER_MIN_AREA = 600

# 滚动参数: 一次滚动 6 格滚轮(负值向下, 与参考实现 MouseWheel -6 同向),
# 滚后沉降等待。「到底」用滚动条滑块位置判定 (scrollbar_thumb_ratio):
# 滑块底部贴住轨道底部即收尾; 固定屏数 (RED_SCAN_MAX_PASSES) 只作滑块
# 读不出时的兜底 —— 视觉模板判定空格子误报的教训见 2026-10-28 记录。
RED_SCROLL_WHEEL = -6
RED_SCROLL_SETTLE = 0.6
RED_SCAN_MAX_PASSES = 15
# 滑块底部位置比例达到该值判定到底 (1.0 = 完全贴底, 留端帽厚度余量)。
RED_SCROLLBAR_BOTTOM_RATIO = 0.93
# 滚动条判定的像素阈值: 亮行判定 (>该灰度)、轨道最小高度与滑块最小高度,
# 端帽约 9px、滑块约 130px (2026-10-28 实测), 阈值按比例放宽。
SCROLLBAR_BRIGHT_THRESHOLD = 110
SCROLLBAR_TRACK_MIN_HEIGHT = 60
SCROLLBAR_THUMB_MIN_HEIGHT = 25
# 回顶固定上滚屏数: 同样不用视觉判定, 从任何深度都能保证到顶。
RED_SCROLL_TOP_PASSES = 15

# 整次扫描的独立预算与单件处理步进等待; 预算耗尽按已完成件数收尾,
# 下一轮开仓时继续扫。300 秒按「每件约 8-10 秒」覆盖三十件上下的清仓。
RED_SCAN_BUDGET = 300.0
DETAIL_POPUP_TIMEOUT = 3.0
DETAIL_STEP_SLEEP = 0.5
ITEM_PRICE_TIMEOUT = 3.0
# 出售成功后「获得物品」提示条的等待上限: 2026-10-28 实测提示条会挡住网格
# 交互(模态), 必须点空白关掉才能继续扫; 等不到(未弹出)就跳过不阻塞。
GAIN_POPUP_TIMEOUT = 3.0

# 跳过判定半径: 同屏内两次聚类中心落在网格高度的该比例内视为同一件;
# 过大会把相邻卡片并成一件。绝对下限 20 像素防超小网格。
SKIP_RADIUS_FRACTION = 0.03
SKIP_RADIUS_MIN = 20.0

# 藏品位置指纹参数: 横向按 30px 桶化(聚类质心抖动约 +-2px), 纵向绝对位置
# 允许 +-0.04 视口高(约 30px@1080p)的滚动条量化误差。指纹命中的候选直接
# 跳过, 不再点开详情 —— 保留过的高价物品不会被重复选中 (用户 2026-10-30
# 反馈: 名字去重仍会重复点开)。
RED_IDENTITY_X_BUCKET = 30
RED_IDENTITY_TOLERANCE = 0.04

# 单次扫描的兜底处理件数上限: 防详情面板识别失灵时的死循环; 正常仓库
# (200 格上限)即使全红也到不了这个数。
RED_SCAN_MAX_ITEMS = 200


def red_item_clusters(frame, box: Box) -> list[tuple[int, int]]:
    """返回网格区内红色卡片的屏幕像素坐标中心列表, 按从上到下、从左到右排序。

    frame 是框架的整屏 BGR 帧, box 是已换算的屏幕 Box (boxes.warehouse_grid)。
    纯计算无 IO, 合成帧可直接单测; 连通域面积低于阈值(比例与绝对值取大,
    见 RED_CLUSTER_MIN_*)的当作噪声丢弃。
    """
    cropped = frame[box.y : box.y + box.height, box.x : box.x + box.width]
    if cropped.size == 0:
        return []
    mask = iu.create_color_mask(cropped, RED_CARD_COLOR_RANGE, to_bgr=False)
    min_area = max(int(mask.size * RED_CLUSTER_MIN_FRACTION), RED_CLUSTER_MIN_AREA)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    clusters = []
    for index in range(1, count):  # 0 号是背景
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = centroids[index]
        clusters.append((int(box.x + cx), int(box.y + cy)))
    clusters.sort(key=lambda point: (point[1], point[0]))
    return clusters


def scrollbar_geometry(frame, box: Box) -> tuple[int, int, int] | None:
    """读取滚动条几何: (滑块顶, 滑块高, 轨道高), 单位像素(框内, 轨道顶为 0)。

    滑块是暗轨道上的浅灰竖条(2026-10-28 截图实测 RGB~185), 轨道两端各有一
    小段亮端帽; 全部亮行的极值即轨道范围, 最长的亮段是滑块。找不到轨道或
    滑块(弹窗遮挡/不在仓库界面)返回 None, 调用方按「读不出」回退固定屏数。
    纯计算无 IO, 合成帧可直接单测。
    """
    strip = frame[box.y : box.y + box.height, box.x : box.x + box.width]
    if strip.size == 0:
        return None
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    bright = gray.max(axis=1) > SCROLLBAR_BRIGHT_THRESHOLD
    rows = np.flatnonzero(bright)
    if rows.size < SCROLLBAR_TRACK_MIN_HEIGHT:
        return None
    track_top, track_bottom = int(rows[0]), int(rows[-1])
    track_height = track_bottom - track_top + 1
    if track_height < SCROLLBAR_TRACK_MIN_HEIGHT:
        return None
    # 最长连续亮段 = 滑块 (端帽是两小段, 滑块明显更长)。
    best_top = best_bottom = -1
    run_start = None
    for y in range(track_top, track_bottom + 1):
        if bright[y] and run_start is None:
            run_start = y
        elif not bright[y] and run_start is not None:
            if y - 1 - run_start > best_bottom - best_top:
                best_top, best_bottom = run_start, y - 1
            run_start = None
    if run_start is not None and track_bottom - run_start > best_bottom - best_top:
        best_top, best_bottom = run_start, track_bottom
    if best_bottom - best_top + 1 < SCROLLBAR_THUMB_MIN_HEIGHT:
        return None
    return best_top - track_top, best_bottom - best_top + 1, track_height


def scrollbar_thumb_ratio(frame, box: Box) -> float | None:
    """滚动条滑块底部相对轨道底部的位置: 0=在最顶, 1=完全到底。

    frame 是整屏 BGR 帧, box 是滚动条条带 (boxes.warehouse_scrollbar)。
    几何读不出时返回 None, 调用方按「读不出」回退固定屏数。
    纯计算无 IO, 合成帧可直接单测。
    """
    geometry = scrollbar_geometry(frame, box)
    if geometry is None:
        return None
    thumb_top, thumb_height, track_height = geometry
    return (thumb_top + thumb_height - 1) / (track_height - 1)


def item_identity(
    center: tuple[int, int], grid: Box, geometry: tuple[int, int, int] | None
) -> tuple[int, float] | None:
    """条目在列表中的绝对位置指纹: (横向桶, 以视口高为单位的纵向绝对位置)。

    纵向按滚动条几何换算: 内容偏移(视口数) = 滑块顶/滑块高, 加上条目在
    视口内的分数位置 —— 同一藏品在任意滚动状态下指纹一致, 是「不再选中
    已保留物品」的依据 (名字 OCR 会抖, 位置指纹不会)。横向滚动不改变,
    按 30px 桶化吸收聚类质心抖动。几何读不出时返回 None, 调用方回落到
    名字 + 屏内坐标去重。
    """
    if geometry is None:
        return None
    thumb_top, thumb_height, _track_height = geometry
    if thumb_height <= 0 or grid.height <= 0:
        return None
    offset_views = thumb_top / thumb_height
    view_frac = (center[1] - grid.y) / grid.height
    return (int(center[0] // RED_IDENTITY_X_BUCKET), round(offset_views + view_frac, 3))


def identity_known(identity: tuple[int, float] | None, known: set) -> bool:
    """指纹是否已在保留集合里; 纵向允许 RED_IDENTITY_TOLERANCE 的量化误差。"""
    if identity is None:
        return False
    x_bucket, position = identity
    return any(
        x_bucket == known_bucket and abs(position - known_position) <= RED_IDENTITY_TOLERANCE
        for known_bucket, known_position in known
    )


def _name_known(name: str, skipped_names: set) -> bool:
    """名字是否已在保留集合里; 双向包含匹配吸收 OCR 丢字/多字的抖动。

    空名直接返回 False: 空串是任何串的子串, 会把所有保留名误判命中。
    """
    if not name:
        return False
    return any(
        known and (name in known or known in name) for known in skipped_names
    )


def should_sell_item(price: int | None, limit: int) -> bool:
    """单件出售判定: 价格读得出且为正且低于上限才卖。

    price 为 None (详情面板价格读不出) 按保留处理 —— 保守方向是少卖,
    少卖可手动补, 误卖贵重红不可逆 (口径同「低于此价才卖红」, 见 auction-notes 7)。
    """
    return price is not None and 0 < price < limit


def normalize_item_name(texts) -> str:
    """把详情面板名称区的 OCR 命中文本拼成规范化名称, 空白全部去掉。

    同名藏品同价, 名称去重即价格决策去重; 读不出名称时返回空串, 调用方
    回落到按屏内坐标跳过。
    """
    joined = "".join(str(text) for text in texts)
    return "".join(joined.split())


def pick_unvisited(
    clusters: list[tuple[int, int]], skipped: list[tuple[int, int]], radius: float
) -> tuple[int, int] | None:
    """取第一个与所有跳过点距离超过 radius 的聚类中心; 全部被跳过时返回 None。"""
    for center in clusters:
        if all(_distance(center, point) > radius for point in skipped):
            return center
    return None


def _distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return float(((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5)


def _scroll_screen(task: AuctionSellOps, passes: int) -> None:
    """把网格滚固定屏数: 传负的 passes 为回顶, 正的为向下翻页。

    固定屏数不做位移检测, 理由见 RED_SCAN_MAX_PASSES 注释; 到顶/到底后的
    空滚只是原地滚轮, 无副作用。
    """
    wheel = RED_SCROLL_WHEEL if passes > 0 else -RED_SCROLL_WHEEL
    for _ in range(abs(passes)):
        auction_interaction.scroll_screen(
            task, *POS_WAREHOUSE_GRID_SCROLL, wheel
        )
        task._bounded_sleep(None, RED_SCROLL_SETTLE)


def _click_point(task: AuctionSellOps, point: tuple[int, int]) -> None:
    """点击屏幕像素坐标: 用 2x2 Box 承载, operate_click 的 Box 分支会点中心。"""
    task.operate_click(Box(point[0], point[1], 2, 2), after_sleep=DETAIL_STEP_SLEEP)


def _read_detail_name(task: AuctionSellOps, boxes: AuctionBoxes) -> str:
    """读详情面板的藏品名, 读不出返回空串 (调用方回落到坐标去重)。"""
    found = task.ocr(box=boxes.item_detail_name, match=None, log=False)
    return normalize_item_name(part.name for part in found)


def _read_item_price(task: AuctionSellOps, boxes: AuctionBoxes) -> int | None:
    """读详情面板的单件价格, 读不出返回 None (按保留处理)。"""
    ops = task._reading_ops()
    return ops.asset_once(ops, boxes.item_detail_price, ITEM_PRICE_TIMEOUT, "藏品单价")


def _find_sell_button(
    task: AuctionSellOps, boxes: AuctionBoxes, timeout: float = DETAIL_POPUP_TIMEOUT
):
    """OCR 找详情面板的「出售」按钮, 返回命中框列表 (空 = 未找到)。"""
    return task.wait_ocr(
        box=boxes.item_sell_btn,
        match=RE_ITEM_SELL,
        time_out=timeout,
        raise_if_not_found=False,
        settle_time=0.3,
    )


def _confirm_sell_popup(task: AuctionSellOps, boxes: AuctionBoxes) -> None:
    """点完出售后的收尾: 面板直接收起(无需确认)就结束, 否则找「确认」或补点出售位。

    参考实现在同一坐标连点两次 (第二次即确认); 这里优先判断面板是否已收起,
    避免面板已经关掉还去补点、误点到压暗的仓库网格上。确认弹窗位置未实测,
    搜索区取出售按钮周边的宽松区域 (BOX_ITEM_CONFIRM)。
    """
    task._bounded_sleep(None, DETAIL_STEP_SLEEP)
    if not _detail_popup_open(task, boxes):
        return  # 面板已收起: 出售直接生效, 没有确认弹窗
    confirm = task.wait_ocr(
        box=boxes.item_confirm,
        match=RE_CONFIRM,
        time_out=DETAIL_POPUP_TIMEOUT,
        raise_if_not_found=False,
        settle_time=0.3,
    )
    if confirm:
        task.operate_click(confirm, after_sleep=DETAIL_STEP_SLEEP)
        return
    # 找不到确认按钮但面板还在: 按参考实现补点一次出售位。
    task.operate_click(task.box_of_screen(*POS_ITEM_SELL), after_sleep=DETAIL_STEP_SLEEP)


def _close_detail_popup(task: AuctionSellOps, boxes: AuctionBoxes) -> None:
    """关闭详情面板 (跳过保留的件): 点面板关闭点, 面板没开也不抛。"""
    task.operate_click(task.box_of_screen(*POS_ITEM_DETAIL_CLOSE), after_sleep=DETAIL_STEP_SLEEP)


def _dismiss_gain_popup(task: AuctionSellOps, boxes: AuctionBoxes) -> None:
    """关掉出售成功后的「获得物品」提示条: 点提示条以外的空白区域。

    与结算界面一键出售后的同款提示 (sell_on_settlement_screen) 同一交互;
    提示条是模态的, 不关掉会挡住后续点击与扫描 (2026-10-28 实测)。等不到
    提示条(游戏没弹)只记日志跳过; 提示条位置与结算界面共用 (仓库视图实测
    一致, 若有偏差按日志微调 BOX_POPUP_CLOSE_HINT)。
    """
    hint = task.wait_ocr(
        box=boxes.popup_close_hint,
        match=RE_POPUP_CLOSE_HINT,
        time_out=GAIN_POPUP_TIMEOUT,
        raise_if_not_found=False,
        settle_time=0.5,
    )
    if not hint:
        task.log_debug("未出现「获得物品」提示, 跳过关闭")
        return
    task.operate_click(boxes.popup_blank, after_sleep=DETAIL_STEP_SLEEP)
    task.log_debug("已点击空白区域关闭「获得物品」提示")


def _detail_popup_open(task: AuctionSellOps, boxes: AuctionBoxes) -> bool:
    """检测详情面板是否还开着: 「出售」按钮文字命中即视为开着。

    用按钮文字而不是名称区: 面板关闭后名称区落在压暗的仓库网格上, OCR 偶尔
    能从网格读出零星文字造成误判; 按钮文字「出售」只在面板上出现, 且与
    仓库自己的出售圆钮(右下角)区域不重叠。
    """
    return bool(task.ocr(box=boxes.item_sell_btn, match=RE_ITEM_SELL, log=False))


def sell_small_red_items(
    task: AuctionSellOps, boxes: AuctionBoxes, price_limit: int
) -> int | None:
    """扫描整个藏品仓库, 出售单价低于 price_limit 的红色藏品, 返回出售件数。

    必须在藏品仓库普通视图 (未点出售圆钮) 调用。判定为保留的件记「位置指纹」
    (滚动条几何换算的列表绝对位置, 见 item_identity) 与名字, 后续屏再次出现时
    直接跳过不点开 (2026-10-30 用户反馈: 只靠名字去重仍会重复点开保留件)。
    滚动条几何读不出时回落到屏内坐标 + 名字去重。流程走完返回出售件数 (0 也算
    走完); 「确认出售后面板未收起」这类结果未知的情况返回 None, 由调用方按
    「结果未知」处理。TaskDisabledException 原样上抛, 其余异常兜住记日志并返回
    None。
    """
    if price_limit <= 0:
        task.log_warning(f"「小红出售价格上限」为 {price_limit}, 跳过本次出售小红扫描")
        return None
    task.log_info(f"开始出售小红扫描: 单件价格低于 {price_limit} 的红色藏品出售")
    budget = clock.timer(timeout=RED_SCAN_BUDGET)
    skip_radius = max(SKIP_RADIUS_MIN, boxes.warehouse_grid.height * SKIP_RADIUS_FRACTION)
    skipped_names: set[str] = set()
    skipped_points: list[tuple[int, int]] = []
    skipped_ids: set[tuple[int, float]] = set()
    sold = 0
    processed = 0
    try:
        _scroll_screen(task, -RED_SCROLL_TOP_PASSES)
        task.next_frame()
        top_ratio = scrollbar_thumb_ratio(task.frame, boxes.warehouse_scrollbar)
        if top_ratio is not None:
            task.log_debug(f"回顶完成, 滑块位置 {top_ratio:.2f} (0=顶)")
        for current_pass in range(1, RED_SCAN_MAX_PASSES + 1):
            if budget.expired():
                task.log_warning(f"出售小红扫描超出 {RED_SCAN_BUDGET} 秒预算, 提前收尾")
                break
            if processed >= RED_SCAN_MAX_ITEMS:
                task.log_warning(f"出售小红扫描处理件数达兜底上限 {RED_SCAN_MAX_ITEMS}, 收尾")
                break
            # 一屏内反复「识别 → 处理一件」, 直到本屏没有未处理的候选
            # (出售回流时 skipped_points 已清空, 会按新坐标重新识别)。
            # 预算与件数护栏放在内层: 一屏内连续出售多件同样受约束。
            while True:
                if budget.expired() or processed >= RED_SCAN_MAX_ITEMS:
                    break
                task.next_frame()
                frame = task.frame
                clusters = red_item_clusters(frame, boxes.warehouse_grid)
                geometry = scrollbar_geometry(frame, boxes.warehouse_scrollbar)
                # 选第一个「屏内坐标未跳过 且 位置指纹未保留」的候选;
                # 指纹命中说明是保留过的物品, 不点开直接略过。
                center = None
                identity = None
                for candidate in clusters:
                    if any(_distance(candidate, point) <= skip_radius for point in skipped_points):
                        continue
                    ident = item_identity(candidate, boxes.warehouse_grid, geometry)
                    if identity_known(ident, skipped_ids):
                        continue
                    center = candidate
                    identity = ident
                    break
                if center is None:
                    break
                processed += 1
                task.log_debug(
                    f"第 {current_pass} 屏识别到 {len(clusters)} 个红色候选, "
                    f"点击 ({center[0]},{center[1]})"
                )
                _click_point(task, center)
                sell_hit = _find_sell_button(task, boxes)
                if not sell_hit:
                    # 颜色误报或点击落空: 详情面板没有弹出来, 按坐标跳过继续。
                    task.log_debug(f"点击 ({center[0]},{center[1]}) 后未出现藏品详情, 按误报跳过")
                    skipped_points.append(center)
                    continue
                name = _read_detail_name(task, boxes)
                if name and _name_known(name, skipped_names):
                    # 名字去重的兜底路径 (指纹读不出时的第二道防线)。
                    if identity is not None:
                        skipped_ids.add(identity)
                    _close_detail_popup(task, boxes)
                    skipped_points.append(center)
                    continue
                price = _read_item_price(task, boxes)
                if should_sell_item(price, price_limit):
                    task.log_info(f"出售红色藏品「{name or '未识别名称'}」单价 {price}")
                    task.operate_click(sell_hit, after_sleep=0)
                    _confirm_sell_popup(task, boxes)
                    if _detail_popup_open(task, boxes):
                        task.log_warning("确认出售后详情面板未收起, 本件是否售出未知")
                        _close_detail_popup(task, boxes)
                        return None
                    sold += 1
                    # 提示条是模态的, 必须先关掉再继续扫 (不关会挡住全部点击)。
                    _dismiss_gain_popup(task, boxes)
                    skipped_points = []  # 列表回流, 当前屏坐标全部作废重扫
                    # 卖出件下方的条目整体上移, 它们的纵向指纹全部失效;
                    # 只保留卖出点之上(不受回流影响)的指纹, 名字集合不受影响。
                    if identity is not None:
                        skipped_ids = {
                            key for key in skipped_ids if key[1] <= identity[1] + 0.001
                        }
                    continue
                task.log_info(
                    f"红色藏品「{name or '未识别名称'}」单价 "
                    f"{price if price is not None else '未读出'}, 不低于 {price_limit}, 保留"
                )
                # 保留: 名字与位置指纹都记, 名字读不出也靠指纹挡住后续重选。
                if name:
                    skipped_names.add(name)
                if identity is not None:
                    skipped_ids.add(identity)
                _close_detail_popup(task, boxes)
                skipped_points.append(center)
            # 本屏处理完先看滚动条: 滑块贴底 = 最后一屏也已处理, 整仓扫完收尾。
            # 滑块读不出(遮挡/界面切换)不拦截, 靠固定屏数兜底继续翻页。
            task.next_frame()
            ratio = scrollbar_thumb_ratio(task.frame, boxes.warehouse_scrollbar)
            if ratio is not None and ratio >= RED_SCROLLBAR_BOTTOM_RATIO:
                task.log_info(f"滚动条滑块已贴底({ratio:.2f}), 整仓扫描完成")
                break
            # 未到底: 向下翻一屏; 固定屏数只作滑块读不出时的兜底。
            if current_pass < RED_SCAN_MAX_PASSES:
                _scroll_screen(task, 1)
                skipped_points = []
        if processed == 0:
            task.log_warning(
                "整仓扫描未识别到任何红色藏品; 若仓库确有红色藏品, 请检查仓库网格区域"
                "或红卡颜色范围是否需要校准"
            )
        # 收尾兜一次: 扫描中途预算耗尽等路径可能残留提示条, 不关掉会挡住
        # 后续的仓库关闭点击。
        _dismiss_gain_popup(task, boxes)
        task.log_info(
            f"出售小红扫描完成, 共处理 {processed} 件, 出售 {sold} 件红色藏品"
        )
        return sold
    except TaskDisabledException:
        raise
    except Exception as e:
        task._log_aux_error("出售小红扫描失败", e)
        return None
