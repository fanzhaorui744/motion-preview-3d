"""Build multi-sequence GT/phone/watch alignment diagnostic pages."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SEQUENCE_ID = "2026_8_14_test_81"
EXTRA_SEQUENCE_IDS = [
    "2026_8_14_test_111",
    "test_11",
    "2026_8_14_test_91",
]
SEQUENCE_IDS = [DEFAULT_SEQUENCE_ID, *EXTRA_SEQUENCE_IDS]
OUTPUT_ROOT = ROOT / "unified_motion_preview_20260916"


def sequence_paths(sequence_id: str) -> dict[str, Path]:
    return {
        "world_csv": (
            ROOT
            / "phone_watch_post_third_jump_minusx_gt_world_csv_20260828_v3"
            / sequence_id
            / "aligned_gt_world.csv"
        ),
        "metadata": (
            ROOT
            / "phone_watch_post_third_jump_minusx_dual_prepared_20260828_v1"
            / "sequences"
            / sequence_id
            / "metadata.json"
        ),
        "raw_npz": (
            ROOT
            / "clock_corrected_npz_20260821"
            / sequence_id
            / f"{sequence_id}.npz"
        ),
    }


def raw_npz_path(sequence_id: str) -> Path:
    path = sequence_paths(sequence_id)["raw_npz"]
    if path.exists():
        return path
    return ROOT / "有效数据集" / "dual_phone_watch" / sequence_id / f"{sequence_id}.npz"


def read_world_csv(sequence_id: str) -> dict[str, np.ndarray]:
    with sequence_paths(sequence_id)["world_csv"].open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))

    def column(name: str) -> np.ndarray:
        return np.asarray(
            [float(row[name]) if row[name] != "" else np.nan for row in rows], dtype=float
        )

    def vector(prefix: str, suffix: str) -> np.ndarray:
        return np.column_stack([column(f"{prefix}_{axis}_{suffix}") for axis in "xyz"])

    result = {
        "time": column("time_s"),
        "gt_position": vector("gt_centroid", "m"),
        "gt_velocity": vector("gt_velocity", "mps"),
        "gt_acceleration": vector("gt_acceleration", "mps2"),
        "phone_acceleration": vector("phone_acc_gt", "mps2"),
        "watch_acceleration": vector("watch_acc_gt", "mps2"),
        "gt_valid": column("gt_valid").astype(bool),
        "phone_valid": column("phone_valid").astype(bool),
        "watch_valid": column("watch_valid").astype(bool),
    }
    # Some exported CSVs mark isolated samples as invalid with blank numerics.
    # Zero-fill those isolated blanks; validity flags remain available for QA.
    for key, values in result.items():
        if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.floating):
            result[key] = np.where(np.isfinite(values), values, 0.0)
    return result


def interpolate(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(len(source_time), -1)
    sampled = np.column_stack(
        [np.interp(target_time, source_time, flat[:, index]) for index in range(flat.shape[1])]
    )
    return sampled.reshape((len(target_time),) + values.shape[1:])


def nearest(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_time, target_time, side="left")
    right = np.clip(right, 0, len(source_time) - 1)
    left = np.clip(right - 1, 0, len(source_time) - 1)
    choose_left = np.abs(target_time - source_time[left]) <= np.abs(source_time[right] - target_time)
    indices = np.where(choose_left, left, right)
    return np.asarray(values)[indices]


def smooth_rms(values: np.ndarray, width: int = 9) -> np.ndarray:
    kernel = np.ones(width, dtype=float) / width
    return np.sqrt(np.convolve(np.asarray(values, dtype=float) ** 2, kernel, mode="same"))


def normalized_energy(acceleration: np.ndarray, remove_gravity: bool) -> np.ndarray:
    magnitude = np.linalg.norm(acceleration, axis=1)
    if remove_gravity:
        magnitude = magnitude - 9.80665
    energy = smooth_rms(magnitude)
    low, high = np.percentile(energy, [5, 99])
    scale = max(float(high - low), 1e-9)
    return np.clip((energy - low) / scale, 0.0, 1.15)


def straight_segments(time: np.ndarray, position: np.ndarray, hz: float) -> list[dict]:
    """Find short, useful straight-walk windows for direction QA."""
    # GT world uses X-Z as the horizontal plane and +Y as vertical.
    horizontal_step = np.diff(position[:, [0, 2]], axis=0)
    step_length = np.linalg.norm(horizontal_step, axis=1)
    heading = np.arctan2(horizontal_step[:, 1], horizontal_step[:, 0])
    window = int(round(3.0 * hz))
    candidates: list[tuple[float, int, int]] = []
    for start in range(0, len(time) - window):
        stop = start + window
        lengths = step_length[start:stop]
        angles = heading[start:stop]
        path_length = float(lengths.sum())
        median_speed = float(np.median(lengths) * hz)
        straightness = float(
            np.hypot(np.cos(angles).mean(), np.sin(angles).mean())
        )
        if path_length < 0.9 or median_speed < 0.2 or straightness < 0.982:
            continue
        quality = straightness * path_length
        candidates.append((quality, start, stop))

    candidates.sort(reverse=True)
    selected: list[tuple[int, int]] = []
    for _, start, stop in candidates:
        if all(max(start, old_start) > min(stop, old_stop) for old_start, old_stop in selected):
            selected.append((start, stop))
        if len(selected) >= 16:
            break

    selected.sort()
    result: list[dict] = []
    for start, stop in selected:
        steps = horizontal_step[start:stop]
        bearing = float(
            np.degrees(np.arctan2(float(steps[:, 1].sum()), float(steps[:, 0].sum())))
        )
        result.append(
            {
                "start_index": int(start),
                "end_index": int(stop),
                "start_s": round(float(time[start]), 2),
                "end_s": round(float(time[stop]), 2),
                "length_m": round(float(step_length[start:stop].sum()), 2),
                "bearing_deg": round(bearing, 1),
            }
        )
    return result


def lag_diagnostic(reference: np.ndarray, device: np.ndarray, hz: float) -> dict:
    reference = np.asarray(reference, dtype=float)
    device = np.asarray(device, dtype=float)
    start = int(round(hz))
    stop = len(reference) - start
    reference = reference[start:stop]
    device = device[start:stop]
    max_lag = int(round(2.0 * hz))
    zero_correlation = float(np.corrcoef(reference, device)[0, 1])
    candidates: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ref, candidate = reference[-lag:], device[:lag]
        elif lag > 0:
            ref, candidate = reference[:-lag], device[lag:]
        else:
            ref, candidate = reference, device
        candidates.append((float(np.corrcoef(ref, candidate)[0, 1]), lag))
    best_correlation, best_lag = max(candidates)
    lag_seconds = float(best_lag / hz)
    review = abs(lag_seconds) > 0.12 or best_correlation < 0.75
    return {
        "candidate_lag_s": round(lag_seconds, 3),
        "best_correlation": round(best_correlation, 3),
        "zero_lag_correlation": round(zero_correlation, 3),
        "interpretation": "needs_review" if review else "consistent_with_alignment",
        "lag_sign": "positive means device signal occurs later than GT",
    }


def build_payload(sequence_id: str) -> dict:
    world = read_world_csv(sequence_id)
    metadata = json.loads(
        sequence_paths(sequence_id)["metadata"].read_text(encoding="utf-8"),
        strict=False,
    )
    time = world["time"]
    hz = float(1.0 / np.median(np.diff(time)))
    crop_start = float(metadata["segmented_rebuild_alignment"]["crop_start_s"])

    with np.load(raw_npz_path(sequence_id)) as raw:
        raw_time = np.asarray(raw["time_s"], dtype=float)
        query_time = time + crop_start
        markers = interpolate(raw["gt_marker_position"], raw_time, query_time)
        marker_valid = nearest(raw["gt_marker_position_valid"], raw_time, query_time).astype(bool)
        raw_centroid = interpolate(raw["gt_centroid_position"], raw_time, query_time)

    correction = world["gt_position"] - raw_centroid
    markers = markers + correction[:, None, :]
    markers = np.where(marker_valid[:, :, None], markers, world["gt_position"][:, None, :])
    origin = world["gt_position"][0]
    position = world["gt_position"] - origin
    markers = markers - origin

    gt_energy = normalized_energy(world["gt_acceleration"], remove_gravity=False)
    phone_energy = normalized_energy(world["phone_acceleration"], remove_gravity=True)
    watch_energy = normalized_energy(world["watch_acceleration"], remove_gravity=True)
    diagnostics = {
        "phone_vs_gt": lag_diagnostic(gt_energy, phone_energy, hz),
        "watch_vs_gt": lag_diagnostic(gt_energy, watch_energy, hz),
    }
    segments = straight_segments(time, position, hz)

    return {
        "schema": "full_alignment_diagnostic_v1",
        "sequence_id": sequence_id,
        "sample_hz": round(hz, 3),
        "sample_count": len(time),
        "duration_s": round(float(time[-1]), 3),
        "coordinate_contract": {
            "frame": "GT/XINGYING world XYZ",
            "vertical_axis": "+Y",
            "imu_transform": "v_GT = R_GT_from_device @ v_device",
            "position_origin": "first GT centroid sample",
            "phone_watch_position_available": False,
        },
        "source_alignment": {
            "sync_status": metadata["sync_status"],
            "quality_class": metadata["quality_class"],
            "valid_fraction": metadata["valid_fraction"],
            "crop_start_s": crop_start,
            "marker_centroid_correction_rmse_m": round(
                float(np.sqrt(np.mean(correction**2))), 6
            ),
        },
        "lag_diagnostics": diagnostics,
        "time_s": np.round(time, 4).tolist(),
        "straight_segments": segments,
        "gt_position_m": np.round(position, 5).tolist(),
        "gt_markers_m": np.round(markers, 5).tolist(),
        "gt_marker_valid": marker_valid.tolist(),
        "gt_acc_mps2": np.round(world["gt_acceleration"], 5).tolist(),
        "phone_acc_mps2": np.round(world["phone_acceleration"], 5).tolist(),
        "watch_acc_mps2": np.round(world["watch_acceleration"], 5).tolist(),
        "gt_energy": np.round(gt_energy, 5).tolist(),
        "phone_energy": np.round(phone_energy, 5).tolist(),
        "watch_energy": np.round(watch_energy, 5).tolist(),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN" data-runtime-status="loading">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GT / Phone / Watch Alignment Diagnostic</title>
  <style>
    :root { --ink:#17212b; --muted:#66747e; --line:#cbd4d8; --page:#f4f6f6; --panel:#fff; --gt:#d1493f; --phone:#087f8c; --watch:#8b5aa3; --ok:#2b7a57; --warn:#ad6418; }
    * { box-sizing:border-box; }
    html,body { margin:0; min-height:100%; }
    body { overflow-x:hidden; background:var(--page); color:var(--ink); font:14px/1.4 Arial,"Microsoft YaHei",sans-serif; }
    header { background:#fff; border-bottom:1px solid var(--line); padding:15px 22px 12px; }
    h1 { margin:0; font-size:20px; font-weight:680; letter-spacing:0; }
    .subtitle { margin-top:4px; color:var(--muted); overflow-wrap:anywhere; }
    .controls { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:12px; }
    button,select,input { font:inherit; }
    button,select { min-height:32px; border:1px solid #aebbc1; border-radius:3px; background:#fff; color:var(--ink); padding:5px 9px; }
    button { cursor:pointer; font-weight:650; }
    button:hover { background:#edf2f3; }
    input[type=range] { flex:1 1 320px; min-width:180px; accent-color:var(--gt); }
    .clock { min-width:150px; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
    .statusbar { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border-bottom:1px solid var(--line); background:#fff; }
    .status { padding:9px 14px; border-right:1px solid var(--line); min-width:0; }
    .status:last-child { border-right:0; }
    .status b { display:block; font-size:11px; color:var(--muted); font-weight:650; }
    .status span { display:block; margin-top:2px; font-variant-numeric:tabular-nums; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .review { color:var(--warn); font-weight:700; }
    main { width:min(1560px,100%); margin:auto; padding:14px 22px 24px; display:grid; grid-template-columns:minmax(0,1.28fr) minmax(0,1fr); gap:14px; }
    section { min-width:0; background:var(--panel); border:1px solid var(--line); }
    h2 { margin:0; padding:9px 11px; border-bottom:1px solid var(--line); font-size:14px; font-weight:680; }
    canvas { display:block; width:100%; }
    #trajectory { height:430px; }
    #vectors { height:220px; }
    #overview { height:150px; }
    #zoom { height:245px; }
    #scene3d { height:540px; cursor:grab; touch-action:none; }
    .scene-toolbar { display:flex; align-items:center; gap:8px 10px; flex-wrap:wrap; padding:9px 11px; border-bottom:1px solid var(--line); background:#fbfcfc; font-size:12px; color:var(--muted); }
    .scene-toolbar select { min-height:30px; }
    .scene-toolbar label { display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }
    .direction-grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px 12px; padding:10px 11px; border-top:1px solid var(--line); font-variant-numeric:tabular-nums; }
    .direction-grid .metric { border-left-width:3px; }
    .wide { grid-column:1 / -1; }
    .note { padding:8px 11px 10px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; padding:8px 11px 0; color:var(--muted); font-size:12px; }
    .swatch { display:inline-block; width:18px; height:3px; margin-right:5px; vertical-align:3px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px 12px; padding:10px 11px 12px; font-variant-numeric:tabular-nums; }
    .metric { border-left:3px solid var(--line); padding-left:7px; min-width:0; }
    .metric b { display:block; color:var(--muted); font-size:11px; }
    .metric span { display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    @media (max-width:900px) { header { padding:13px 14px; } h1 { font-size:18px; } .clock { width:100%; text-align:left; } .statusbar { grid-template-columns:repeat(2,minmax(0,1fr)); } .status:nth-child(2) { border-right:0; } main { padding:12px 14px 20px; grid-template-columns:minmax(0,1fr); } .wide { grid-column:auto; } #trajectory { height:390px; } .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:900px) { #scene3d { height:460px; } .direction-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:520px) { .statusbar { grid-template-columns:1fr; } .status { border-right:0; border-bottom:1px solid var(--line); } .status:last-child { border-bottom:0; } #trajectory { height:350px; } .metrics { grid-template-columns:1fr; } #scene3d { height:400px; } }
  </style>
</head>
<body>
  <header>
    <h1>多序列完整诊断: GT / 手机 / 手表</h1>
    <div class="subtitle" id="subtitle"></div>
    <div class="controls">
      <label for="sequenceSelect">序列</label><select id="sequenceSelect" aria-label="选择序列"></select>
      <button id="play" type="button" title="播放或暂停">Pause</button>
      <button id="reset" type="button" title="回到序列起点">Reset</button>
      <label for="speed">速度</label><select id="speed"><option value="1">1x</option><option value="4">4x</option><option value="16" selected>16x</option></select>
      <label for="timeline">时间</label><input id="timeline" type="range" min="0" value="0" step="1" aria-label="完整序列时间轴">
      <span class="clock" id="clock"></span>
    </div>
  </header>
  <div class="statusbar">
    <div class="status"><b>数据有效率</b><span id="validity"></span></div>
    <div class="status"><b>主时间对齐</b><span>初始三跳事件</span></div>
    <div class="status"><b>残余时延 QA</b><span id="lagQa"></span></div>
    <div class="status"><b>当前判断</b><span class="review">需要结合峰值重合复核</span></div>
  </div>
  <main>
    <section class="wide">
      <h2>3D 方向诊断: GT 轨迹 + 去重力加速度</h2>
      <div class="scene-toolbar">
        <label for="segmentSelect">直走段</label>
        <select id="segmentSelect" aria-label="选择直走段"></select>
        <button id="jumpSegment" type="button">跳到起点</button>
        <button id="topView" type="button">俯视</button>
        <button id="resetView" type="button">复位视角</button>
        <label><input id="horizontalOnly" type="checkbox" checked> 只看水平加速度</label>
        <label>平滑 <select id="smoothWindow"><option value="1">原始</option><option value="5">0.2 s</option><option value="10" selected>0.4 s</option><option value="20">0.8 s</option></select></label>
      </div>
      <canvas id="scene3d" role="img" aria-label="GT 三维轨迹和三端去重力加速度方向"></canvas>
      <div class="legend">
        <span><i class="swatch" style="background:var(--gt)"></i>GT</span>
        <span><i class="swatch" style="background:var(--phone)"></i>Phone</span>
        <span><i class="swatch" style="background:var(--watch)"></i>Watch</span>
        <span>拖动旋转 · 滚轮缩放</span>
      </div>
      <div class="direction-grid" id="directionMetrics"></div>
      <div class="note">三端均使用初始三跳对齐后的 GT 世界系数据，这里不再额外平移时延。箭头是去重力加速度：Phone/Watch 已减去 +Y 重力，再做短期平滑；“只看水平加速度”会把 Y 置为 0。下方数值对当前选中直走段计算，X/Z 相关为正说明该轴方向一致，接近 0 说明只是时间/部位差异，为负说明该轴反着。</div>
    </section>
    <section>
      <h2>GT: 四 marker、质心与完整 X-Z 轨迹</h2>
      <canvas id="trajectory" role="img" aria-label="GT 四 marker 和质心完整轨迹"></canvas>
      <div class="note">位置来自光学 GT。右下角放大当前帧 marker 几何；主图显示完整 307 秒轨迹及已播放部分。</div>
    </section>
    <section>
      <h2>当前帧: GT / Phone / Watch 三轴响应</h2>
      <canvas id="vectors" role="img" aria-label="GT 手机 手表当前三轴加速度向量"></canvas>
      <div class="metrics" id="metrics"></div>
      <div class="note">三组向量均为 GT/XINGYING 世界 XYZ；手机与手表是加速度计读数，包含重力和佩戴部位运动。</div>
    </section>
    <section class="wide">
      <h2>完整 5 分钟动态强度总览</h2>
      <canvas id="overview" role="img" aria-label="完整序列 GT 手机 手表动态强度"></canvas>
      <div class="legend"><span><i class="swatch" style="background:var(--gt)"></i>GT</span><span><i class="swatch" style="background:var(--phone)"></i>Phone</span><span><i class="swatch" style="background:var(--watch)"></i>Watch</span></div>
    </section>
    <section class="wide">
      <h2>当前 12 秒对齐窗口</h2>
      <canvas id="zoom" role="img" aria-label="GT 手机 手表十二秒局部对齐窗口"></canvas>
      <div class="note">判断依据是同一动作峰值是否在竖直播放线附近同步出现。相关性候选时延只作提示；周期步态可能产生多个相关峰。</div>
    </section>
  </main>
  <script>
    window.__runtimeErrors=[];
    window.addEventListener('error',event=>{ window.__runtimeErrors.push(event.message); document.documentElement.dataset.runtimeStatus='error'; document.documentElement.dataset.runtimeError=event.message; });
    const data=__PAYLOAD__; const sequenceOptions=__SEQUENCE_OPTIONS__;
    const colors={gt:'#d1493f',phone:'#087f8c',watch:'#8b5aa3',ink:'#17212b',muted:'#66747e',line:'#cbd4d8',path:'#9aa8ae',marker:['#168aad','#4f7cac','#f4a261','#6a994e']};
    const query=new URLSearchParams(location.search); const requestedTime=Number(query.get('t')); let cursor=Number.isFinite(requestedTime)?Math.max(0,Math.min(data.sample_count-1,Math.round(requestedTime*data.sample_hz))):0; let playing=query.get('autoplay')!=='0'; let previous=null;
    const trajectory=document.querySelector('#trajectory'),tctx=trajectory.getContext('2d'); const vectors=document.querySelector('#vectors'),vctx=vectors.getContext('2d'); const overview=document.querySelector('#overview'),octx=overview.getContext('2d'); const zoom=document.querySelector('#zoom'),zctx=zoom.getContext('2d'); const timeline=document.querySelector('#timeline'); const play=document.querySelector('#play'); const speed=document.querySelector('#speed');
    timeline.max=String(data.sample_count-1); timeline.value=String(Math.floor(cursor)); play.textContent=playing?'Pause':'Play';
    sequenceOptions.forEach(option=>{const element=document.createElement('option');element.value=option.href;element.textContent=option.label;if(option.sequence_id===data.sequence_id)element.selected=true;document.querySelector('#sequenceSelect').appendChild(element);});
    document.querySelector('#sequenceSelect').addEventListener('change',event=>{location.href=event.target.value;});
    document.querySelector('#subtitle').textContent=`${data.sequence_id} | ${data.duration_s.toFixed(2)} s | ${data.sample_hz.toFixed(0)} Hz | ${data.sample_count} frames | ${data.source_alignment.sync_status}`;
    const valid=data.source_alignment.valid_fraction; document.querySelector('#validity').textContent=`GT ${(valid.gt*100).toFixed(1)}% · Phone ${(valid.phone*100).toFixed(1)}% · Watch ${(valid.watch*100).toFixed(1)}%`;
    function lagText(key){const d=data.lag_diagnostics[key];const sign=d.candidate_lag_s>0?'+':'';return `${sign}${d.candidate_lag_s.toFixed(2)} s · corr ${d.best_correlation.toFixed(2)} · 需复核`;}
    document.querySelector('#lagQa').textContent=`Phone ${lagText('phone_vs_gt')} · Watch ${lagText('watch_vs_gt')} · 不参与绘图`;
    function fit(canvas,ctx){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}ctx.setTransform(dpr,0,0,dpr,0,0);return rect;}
    function path(ctx,points,color,width=1,dash=[]){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
    function circle(ctx,p,r,color){ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
    function frameIndex(){return Math.max(0,Math.min(data.sample_count-1,Math.floor(cursor)));}
    function timeLabel(seconds){const minutes=Math.floor(seconds/60);return `${minutes}:${(seconds-minutes*60).toFixed(1).padStart(4,'0')}`;}
    function vectorText(values){return values.map(v=>Number(v).toFixed(2)).join(', ');}
    function drawTrajectory(){const rect=fit(trajectory,tctx),index=frameIndex(),positions=data.gt_position_m,pad=34,xValues=positions.map(p=>p[0]),zValues=positions.map(p=>p[2]),xmin=Math.min(...xValues),xmax=Math.max(...xValues),zmin=Math.min(...zValues),zmax=Math.max(...zValues),spanX=Math.max(xmax-xmin,.5),spanZ=Math.max(zmax-zmin,.5),scale=Math.min((rect.width-pad*2)/spanX,(rect.height-pad*2)/spanZ),map=p=>[pad+(p[0]-xmin)*scale,rect.height-pad-(p[2]-zmin)*scale];tctx.clearRect(0,0,rect.width,rect.height);path(tctx,positions.map(map),colors.path,1);path(tctx,positions.slice(0,index+1).map(map),colors.gt,2.2);const current=positions[index],currentPoint=map(current),markers=data.gt_markers_m[index];markers.forEach((marker,i)=>{if(!data.gt_marker_valid[index][i])return;circle(tctx,map(marker),4,colors.marker[i]);});circle(tctx,currentPoint,6,colors.gt);tctx.fillStyle=colors.muted;tctx.font='11px Arial';tctx.fillText('X',rect.width-17,rect.height-12);tctx.fillText('Z',10,15);
      const box={x:rect.width-187,y:rect.height-165,w:169,h:145};tctx.fillStyle='rgba(255,255,255,.95)';tctx.fillRect(box.x,box.y,box.w,box.h);tctx.strokeStyle=colors.line;tctx.strokeRect(box.x,box.y,box.w,box.h);tctx.fillStyle=colors.ink;tctx.fillText('Current marker geometry',box.x+9,box.y+16);const center=[box.x+box.w/2,box.y+box.h/2+12],localScale=330;markers.forEach((marker,i)=>{if(!data.gt_marker_valid[index][i])return;const dx=marker[0]-current[0],dz=marker[2]-current[2],point=[center[0]+dx*localScale,center[1]-dz*localScale];path(tctx,[center,point],colors.marker[i],1);circle(tctx,point,5,colors.marker[i]);});circle(tctx,center,6,colors.gt);
    }
    function drawArrow(ctx,origin,values,color,scale){const endpoint=[origin[0]+(values[0]-.45*values[2])*scale,origin[1]-(values[1]+.18*values[0]+.18*values[2])*scale];path(ctx,[origin,endpoint],color,2.4);const angle=Math.atan2(endpoint[1]-origin[1],endpoint[0]-origin[0]),head=7;ctx.beginPath();ctx.moveTo(endpoint[0],endpoint[1]);ctx.lineTo(endpoint[0]-head*Math.cos(angle-.5),endpoint[1]-head*Math.sin(angle-.5));ctx.lineTo(endpoint[0]-head*Math.cos(angle+.5),endpoint[1]-head*Math.sin(angle+.5));ctx.closePath();ctx.fillStyle=color;ctx.fill();}
    function drawVectors(){const rect=fit(vectors,vctx),index=frameIndex(),series=[['GT',data.gt_acc_mps2[index],colors.gt],['Phone',data.phone_acc_mps2[index],colors.phone],['Watch',data.watch_acc_mps2[index],colors.watch]],column=rect.width/3,maxValue=16,scale=Math.min(column*.27,rect.height*.22)/maxValue;vctx.clearRect(0,0,rect.width,rect.height);series.forEach(([label,values,color],i)=>{const origin=[column*(i+.5),rect.height*.61];path(vctx,[[origin[0]-column*.27,origin[1]],[origin[0]+column*.27,origin[1]]],colors.line,1);path(vctx,[[origin[0],origin[1]+45],[origin[0],origin[1]-65]],colors.line,1);circle(vctx,origin,4,colors.ink);drawArrow(vctx,origin,values,color,scale);vctx.fillStyle=color;vctx.font='bold 13px Arial';vctx.textAlign='center';vctx.fillText(label,origin[0],22);vctx.fillStyle=colors.muted;vctx.font='11px Arial';vctx.fillText(`${Math.hypot(...values).toFixed(2)} m/s²`,origin[0],rect.height-13);});vctx.textAlign='left';}
    function drawSignalPlot(canvas,ctx,start,end,showWindow){const rect=fit(canvas,ctx),pad={left:42,right:15,top:16,bottom:24},width=rect.width-pad.left-pad.right,height=rect.height-pad.top-pad.bottom,x=i=>pad.left+((i-start)/Math.max(end-start,1))*width,y=v=>pad.top+(1-Math.min(v,1.15)/1.15)*height;ctx.clearRect(0,0,rect.width,rect.height);[0,.5,1].forEach(value=>{const py=y(value);path(ctx,[[pad.left,py],[rect.width-pad.right,py]],colors.line,1);});const stride=Math.max(1,Math.floor((end-start)/Math.max(width,1)));[['gt_energy',colors.gt],['phone_energy',colors.phone],['watch_energy',colors.watch]].forEach(([key,color])=>{const points=[];for(let i=start;i<=end;i+=stride)points.push([x(i),y(data[key][i])]);path(ctx,points,color,1.35);});const index=frameIndex();if(showWindow){const half=Math.round(6*data.sample_hz),left=Math.max(start,index-half),right=Math.min(end,index+half);ctx.fillStyle='rgba(23,33,43,.08)';ctx.fillRect(x(left),pad.top,Math.max(2,x(right)-x(left)),height);}const playX=x(index);path(ctx,[[playX,pad.top],[playX,rect.height-pad.bottom]],colors.ink,1,[3,3]);ctx.fillStyle=colors.muted;ctx.font='11px Arial';ctx.fillText(timeLabel(data.time_s[start]),pad.left,rect.height-6);const endLabel=timeLabel(data.time_s[end]);ctx.fillText(endLabel,rect.width-pad.right-39,rect.height-6);}
    function drawOverview(){drawSignalPlot(overview,octx,0,data.sample_count-1,true);}
    function drawZoom(){const index=frameIndex(),half=Math.round(6*data.sample_hz),start=Math.max(0,index-half),end=Math.min(data.sample_count-1,index+half);drawSignalPlot(zoom,zctx,start,end,false);}
    function updateText(){const i=frameIndex(),time=data.time_s[i],position=data.gt_position_m[i],metrics=[['GT position XYZ (m)',vectorText(position)],['GT acc XYZ (m/s²)',vectorText(data.gt_acc_mps2[i])],['Phone acc XYZ (m/s²)',vectorText(data.phone_acc_mps2[i])],['Watch acc XYZ (m/s²)',vectorText(data.watch_acc_mps2[i])],['Visible markers',String(data.gt_marker_valid[i].filter(Boolean).length)],['Frame',`${i+1} / ${data.sample_count}`],['Time',timeLabel(time)],['Path displacement',`${Math.hypot(position[0],position[2]).toFixed(2)} m`]];document.querySelector('#metrics').innerHTML=metrics.map(([key,value])=>`<div class="metric"><b>${key}</b><span>${value}</span></div>`).join('');document.querySelector('#clock').textContent=`${timeLabel(time)} / ${timeLabel(data.duration_s)}`;timeline.value=String(i);}
    function render(){drawTrajectory();drawVectors();drawOverview();drawZoom();updateText();document.documentElement.dataset.runtimeStatus='ok';}
    function loop(now){if(previous===null)previous=now;const elapsed=(now-previous)/1000;previous=now;if(playing){cursor+=elapsed*data.sample_hz*Number(speed.value);if(cursor>=data.sample_count)cursor=0;}render();requestAnimationFrame(loop);}
    play.addEventListener('click',()=>{playing=!playing;play.textContent=playing?'Pause':'Play';});document.querySelector('#reset').addEventListener('click',()=>{cursor=0;render();});timeline.addEventListener('input',()=>{cursor=Number(timeline.value);render();});window.addEventListener('resize',render);render();requestAnimationFrame(loop);
    function drawScene3d(){const rect=fit(scene,sctx),info=sceneCenter(),index=frameIndex(),positions=data.gt_position_m,current=positions[index],markers=data.gt_markers_m[index];sctx.clearRect(0,0,rect.width,rect.height);sctx.fillStyle='#fff';sctx.fillRect(0,0,rect.width,rect.height);
      const xValues=positions.map(point=>point[0]),zValues=positions.map(point=>point[2]);const xMin=Math.floor(Math.min(...xValues)-.5),xMax=Math.ceil(Math.max(...xValues)+.5),zMin=Math.floor(Math.min(...zValues)-.5),zMax=Math.ceil(Math.max(...zValues)+.5);for(let x=xMin;x<=xMax;x++)path3d([[x,0,zMin],[x,0,zMax]],colors.line,1);for(let z=zMin;z<=zMax;z++)path3d([[xMin,0,z],[xMax,0,z]],colors.line,1);
      path3d([[0,0,0],[1.2,0,0]],'#c43c31',4);path3d([[0,0,0],[0,1.2,0]],'#33824a',4);path3d([[0,0,0],[0,0,1.2]],'#33608c',4);sctx.font='bold 13px Arial';let label=project3d([1.32,0,0],rect,info);sctx.fillStyle='#c43c31';sctx.fillText('X',label[0],label[1]);label=project3d([0,1.32,0],rect,info);sctx.fillStyle='#33824a';sctx.fillText('Y',label[0],label[1]);label=project3d([0,0,1.32],rect,info);sctx.fillStyle='#33608c';sctx.fillText('Z',label[0],label[1]);
      path3d(positions,colors.path,1);const selected=motionSegments[activeSegment];if(selected)path3d(positions.slice(selected.start_index,selected.end_index+1),'#f4a261',4);path3d(positions.slice(0,index+1),colors.gt,2.4);markers.forEach((marker,markerIndex)=>{if(!data.gt_marker_valid[index][markerIndex])return;path3d([current,marker],colors.marker[markerIndex],1.4);circle(sctx,project3d(marker,rect,info),5,colors.marker[markerIndex]);});
      const horizontalOnly=document.querySelector('#horizontalOnly').checked,width=Number(document.querySelector('#smoothWindow').value);const gtVector=linearAcceleration('gt',width)[index],phoneVector=linearAcceleration('phone',width)[index],watchVector=linearAcceleration('watch',width)[index];const arrowScale=horizontalOnly?.20:.15;[[gtVector,colors.gt,5],[phoneVector,colors.phone,4],[watchVector,colors.watch,4]].forEach(([vector,color,lineWidth])=>{const shown=horizontalOnly?[vector[0],0,vector[2]]:vector;arrow3d(current,shown.map(value=>value*arrowScale),color,lineWidth);});
      if(horizontalOnly){sctx.setLineDash([4,4]);path3d([[current[0]-.62,current[1],current[2]-.62],[current[0]+.62,current[1],current[2]-.62],[current[0]+.62,current[1],current[2]+.62],[current[0]-.62,current[1],current[2]+.62],[current[0]-.62,current[1],current[2]-.62]],'#8c979d',1);sctx.setLineDash([]);}
      circle(sctx,project3d(current,rect,info),6,colors.gt);sctx.fillStyle=colors.muted;sctx.font='11px Arial';sctx.fillText(`camera yaw ${(camera.yaw*180/Math.PI).toFixed(0)}° · pitch ${(camera.pitch*180/Math.PI).toFixed(0)}° · drag rotate · wheel zoom`,12,rect.height-12);}
requestAnimationFrame(()=>{
    window.updateDirectionMetrics=function(){const segment=motionSegments[activeSegment],metric=directionMetric();if(!segment){document.querySelector('#directionMetrics').innerHTML='<div class=metric><b>直走段</b><span>未发现</span></div>';return;}const verdict=metric.phone.agreement>.35&&metric.watch.agreement>.35?'两设备箭头趋向 GT 方向':metric.phone.agreement>.35?'Phone 趋向一致，Watch 不稳定':'本段方向证据不足，逐帧看箭头';const items=[['直走段',`${segment.start_s.toFixed(1)}-${segment.end_s.toFixed(1)} s · GT 走向 ${metric.travelBearing.toFixed(0)}°`],['Phone X/Z 相关',`${metric.phone.corrX.toFixed(2)} / ${metric.phone.corrZ.toFixed(2)}`],['Phone 方向一致度',`${(metric.phone.agreement*100).toFixed(0)}%`],['Watch X/Z 相关',`${metric.watch.corrX.toFixed(2)} / ${metric.watch.corrZ.toFixed(2)}`],['Watch 方向一致度',`${(metric.watch.agreement*100).toFixed(0)}%`],['读图',verdict]];document.querySelector('#directionMetrics').innerHTML=items.map(([key,value])=>`<div class="metric"><b>${key}</b><span>${value}</span></div>`).join('');}
    let sceneDragging=false,sceneLast=null;scene.addEventListener('pointerdown',event=>{sceneDragging=true;sceneLast=[event.clientX,event.clientY];scene.setPointerCapture(event.pointerId);scene.style.cursor='grabbing';});scene.addEventListener('pointermove',event=>{if(!sceneDragging)return;camera.yaw+=(event.clientX-sceneLast[0])*.008;camera.pitch=Math.max(-1.48,Math.min(1.48,camera.pitch+(event.clientY-sceneLast[1])*.006));sceneLast=[event.clientX,event.clientY];});scene.addEventListener('pointerup',event=>{sceneDragging=false;scene.style.cursor='grab';scene.releasePointerCapture(event.pointerId);});scene.addEventListener('wheel',event=>{event.preventDefault();camera.zoom=Math.max(.45,Math.min(3.2,camera.zoom*(event.deltaY>0?.92:1.08)));},{passive:false});
    document.querySelector('#segmentSelect').addEventListener('change',event=>{activeSegment=Number(event.target.value);directionCache={key:'',value:null};});document.querySelector('#jumpSegment').addEventListener('click',()=>{if(motionSegments[activeSegment]){cursor=motionSegments[activeSegment].start_index;playing=false;document.querySelector('#play').textContent='Play';}});document.querySelector('#topView').addEventListener('click',()=>{camera.pitch=1.28;});document.querySelector('#resetView').addEventListener('click',()=>{camera.yaw=-.48;camera.pitch=.53;camera.zoom=1;});document.querySelector('#smoothWindow').addEventListener('change',()=>{accelerationCache.clear();directionCache={key:'',value:null};});document.querySelector('#horizontalOnly').addEventListener('change',()=>{directionCache={key:'',value:null};});
});
    function sceneLoop(){drawScene3d();try{updateDirectionMetrics();}catch(error){document.querySelector('#directionMetrics').innerHTML='<div class=metric><b>Direction QA error</b><span>'+error.message+'</span></div>';}requestAnimationFrame(sceneLoop);}requestAnimationFrame(sceneLoop);
    // ==== Interactive 3D direction diagnostics ====
    const scene=document.querySelector('#scene3d'),sctx=scene.getContext('2d');
    const motionSegments=data.straight_segments||[];let activeSegment=0;const requestedSegment=Number(query.get('seg'));
    if(Number.isInteger(requestedSegment)&&requestedSegment>=0&&requestedSegment<motionSegments.length)activeSegment=requestedSegment;
    const camera={yaw:-0.48,pitch:0.53,zoom:1};let sceneInfo=null;const accelerationCache=new Map();let directionCache={key:'',value:null};
    motionSegments.forEach((segment,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=`${segment.start_s.toFixed(1)}-${segment.end_s.toFixed(1)} s · ${segment.length_m.toFixed(1)} m · ${segment.bearing_deg.toFixed(0)}°`;document.querySelector('#segmentSelect').appendChild(option);});
    document.querySelector('#segmentSelect').value=String(activeSegment);
    function pearson(left,right){const n=Math.min(left.length,right.length);let sl=0,sr=0,sll=0,srr=0,slr=0,nUsed=0;for(let i=0;i<n;i++){if(!Number.isFinite(left[i])||!Number.isFinite(right[i]))continue;sl+=left[i];sr+=right[i];sll+=left[i]*left[i];srr+=right[i]*right[i];slr+=left[i]*right[i];nUsed++;}if(nUsed<8)return 0;const covariance=slr-sl*sr/nUsed,variance=Math.max((sll-sl*sl/nUsed)*(srr-sr*sr/nUsed),1e-12);return Math.max(-1,Math.min(1,covariance/Math.sqrt(variance)));}
    function sceneCenter(){if(!sceneInfo){const positions=data.gt_position_m;const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];positions.forEach(point=>{for(let axis=0;axis<3;axis++){min[axis]=Math.min(min[axis],point[axis]);max[axis]=Math.max(max[axis],point[axis]);}});const center=[(min[0]+max[0])/2,(min[1]+max[1])/2,(min[2]+max[2])/2];const radius=Math.max(.8,...positions.map(point=>Math.hypot(point[0]-center[0],point[1]-center[1],point[2]-center[2])));sceneInfo={center,radius};}return sceneInfo;}
    function project3d(point,rect,info){const x=point[0]-info.center[0],y=point[1]-info.center[1],z=point[2]-info.center[2];const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw);const rx=cy*x+sy*z,rz=-sy*x+cy*z;const cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);const vertical=cp*y-sp*rz,depth=sp*y+cp*rz;const scale=Math.min(rect.width/(info.radius*3.1),rect.height/(info.radius*2.2))*camera.zoom;return [rect.width/2+rx*scale,rect.height*.60-vertical*scale,depth];}
    function path3d(points,color,width=1,dash=[]){if(!points.length)return;const rect=scene.getBoundingClientRect(),info=sceneCenter();sctx.beginPath();points.forEach((point,index)=>{const mapped=project3d(point,rect,info);if(index)sctx.lineTo(mapped[0],mapped[1]);else sctx.moveTo(mapped[0],mapped[1]);});sctx.strokeStyle=color;sctx.lineWidth=width;sctx.setLineDash(dash);sctx.stroke();sctx.setLineDash([]);}
    function arrow3d(origin,scaledVector,color,width){const magnitude=Math.hypot(...scaledVector);if(magnitude<.045)return;const rect=scene.getBoundingClientRect(),info=sceneCenter();const start=project3d(origin,rect,info),end=project3d([origin[0]+scaledVector[0],origin[1]+scaledVector[1],origin[2]+scaledVector[2]],rect,info);sctx.beginPath();sctx.moveTo(start[0],start[1]);sctx.lineTo(end[0],end[1]);sctx.strokeStyle=color;sctx.lineWidth=width;sctx.lineCap='round';sctx.stroke();const angle=Math.atan2(end[1]-start[1],end[0]-start[0]),head=Math.max(8,width*3);sctx.beginPath();sctx.moveTo(end[0],end[1]);sctx.lineTo(end[0]-head*Math.cos(angle-.42),end[1]-head*Math.sin(angle-.42));sctx.lineTo(end[0]-head*Math.cos(angle+.42),end[1]-head*Math.sin(angle+.42));sctx.closePath();sctx.fillStyle=color;sctx.fill();}
    function linearAcceleration(device,width){const cacheKey=[device,width].join(':');if(accelerationCache.has(cacheKey))return accelerationCache.get(cacheKey);const sourceKey=device==='gt'?'gt_acc_mps2':device==='phone'?'phone_acc_mps2':'watch_acc_mps2';const gravity=device==='gt'?[0,0,0]:[0,9.80665,0];const source=data[sourceKey],count=source.length,removed=Array.from({length:count},()=>[0,0,0]);
      for(let axis=0;axis<3;axis++){for(let i=0;i<count;i++){let sum=0,used=0;for(let j=Math.max(0,i-25);j<=Math.min(count-1,i+25);j++){sum+=source[j][axis]-gravity[axis];used++;}removed[i][axis]=source[i][axis]-gravity[axis]-sum/Math.max(used,1);}}
      let linear=removed;if(width>1){linear=Array.from({length:count},()=>[0,0,0]);const weights=Array.from({length:width},()=>1/width);for(let axis=0;axis<3;axis++)for(let i=0;i<count;i++){let value=0;for(let j=0;j<width;j++){const index=Math.max(0,Math.min(count-1,i+j-Math.floor(width/2)));value+=removed[index][axis]*weights[j];}linear[i][axis]=value;}}
      accelerationCache.set(cacheKey,linear);return linear;}
    function directionMetric(){const segment=motionSegments[activeSegment];if(!segment)return null;const width=Number(document.querySelector('#smoothWindow').value);const key=[activeSegment,width].join(':');if(directionCache.key===key)return directionCache.value;const gt=linearAcceleration('gt',width),phone=linearAcceleration('phone',width),watch=linearAcceleration('watch',width);const phonePoints=phone.slice(segment.start_index,segment.end_index+1),watchPoints=watch.slice(segment.start_index,segment.end_index+1);const correlations={phone:{x:[],z:[]},watch:{x:[],z:[]}},agreements={phone:[],watch:[]};
      for(let i=segment.start_index;i<=segment.end_index;i++){['phone','watch'].forEach(device=>{const series=device==='phone'?phone:watch;correlations[device].x.push(gt[i][0]);correlations[device].z.push(gt[i][2]);const gtHorizontal=Math.hypot(gt[i][0],gt[i][2]),deviceHorizontal=Math.hypot(series[i][0],series[i][2]);if(gtHorizontal>.45&&deviceHorizontal>.45)agreements[device].push((gt[i][0]*series[i][0]+gt[i][2]*series[i][2])/(gtHorizontal*deviceHorizontal));});}
      const mean=values=>values.length?values.reduce((sum,value)=>sum+value,0)/values.length:0;const dx=data.gt_position_m[segment.end_index][0]-data.gt_position_m[segment.start_index][0],dz=data.gt_position_m[segment.end_index][2]-data.gt_position_m[segment.start_index][2];const value={travelBearing:(180*Math.atan2(dz,dx)/Math.PI),phone:{corrX:pearson(correlations.phone.x,phonePoints.map(point=>point[0])),corrZ:pearson(correlations.phone.z,phonePoints.map(point=>point[2])),agreement:mean(agreements.phone)},watch:{corrX:pearson(correlations.watch.x,watchPoints.map(point=>point[0])),corrZ:pearson(correlations.watch.z,watchPoints.map(point=>point[2])),agreement:mean(agreements.watch)}};directionCache={key,value};return value;}
  </script>
</body>
</html>"""


def main() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    payloads = [build_payload(sequence_id) for sequence_id in SEQUENCE_IDS]

    def safe_name(sequence_id: str) -> str:
        return "".join(character if character.isalnum() else "_" for character in sequence_id)

    page_names = {
        DEFAULT_SEQUENCE_ID: "full_alignment.html",
        **{
            sequence_id: f"full_alignment__{safe_name(sequence_id)}.html"
            for sequence_id in EXTRA_SEQUENCE_IDS
        },
    }
    sequence_options = [
        {
            "sequence_id": payload["sequence_id"],
            "href": page_names[payload["sequence_id"]],
            "label": (
                f"{payload['sequence_id']} · "
                f"{payload['duration_s'] / 60:.1f} min · "
                f"{payload['source_alignment']['quality_class']}"
            ),
        }
        for payload in payloads
    ]

    def render(payload: dict) -> str:
        return (
            HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
            .replace("__SEQUENCE_OPTIONS__", json.dumps(sequence_options, ensure_ascii=False))
        )

    payloads_by_id = {payload["sequence_id"]: payload for payload in payloads}
    default_payload = payloads_by_id[DEFAULT_SEQUENCE_ID]
    for payload in payloads:
        sequence_id = payload["sequence_id"]
        data_name = (
            "full_alignment_data.json"
            if sequence_id == DEFAULT_SEQUENCE_ID
            else f"full_alignment_data__{safe_name(sequence_id)}.json"
        )
        data_path = OUTPUT_ROOT / data_name
        data_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        page_path = OUTPUT_ROOT / page_names[sequence_id]
        page_path.write_text(render(payload), encoding="utf-8")
        print(f"Wrote {data_path}")
        print(f"Wrote {page_path}")

    index_path = OUTPUT_ROOT / "index.html"
    index_path.write_text(render(default_payload), encoding="utf-8")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
