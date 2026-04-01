"""
批次人聲偵測腳本 - 使用 Silero VAD + 多核心平行處理
用途：掃描 train_audio/ 下所有 .ogg 檔，偵測人聲並輸出 CSV 報告
使用方式：
    conda run -n pytorch python script/vad/batch_detect_voice.py
輸出：
    voice_detection_report.csv
"""

import os
import sys
import csv
import torch
import torchaudio
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings

# ── 設定 ──────────────────────────────────────────────
AUDIO_DIR   = Path("train_audio")
OUTPUT_CSV  = Path("voice_detection_report.csv")
SILERO_SR   = 16000
THRESHOLD   = 0.5       # 人聲判定閾值
MERGE_GAP   = 0.3       # 合併相鄰片段的間距 (秒)
NUM_WORKERS = 8         # 平行 worker 數 (12 核心留 4 給系統)
# ──────────────────────────────────────────────────────


def load_audio(file_path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(file_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SILERO_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SILERO_SR)
        waveform = resampler(waveform)
    return waveform.squeeze(0)


def merge_segments(segments: list, gap_sec: float) -> list:
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg['start'] - merged[-1]['end'] <= gap_sec:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg.copy())
    return merged


def process_file(file_path: Path) -> dict:
    """單一檔案的偵測邏輯，在 worker process 中執行"""
    warnings.filterwarnings('ignore')

    rel_path = file_path.relative_to(AUDIO_DIR.parent)  # 相對於專案根目錄

    try:
        waveform = load_audio(file_path)
        total_sec = len(waveform) / SILERO_SR

        # 每個 worker 各自載入模型 (torch.hub 有本地快取，不會重複下載)
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
            verbose=False
        )
        get_speech_timestamps = utils[0]

        speech_timestamps = get_speech_timestamps(
            waveform,
            model,
            threshold=THRESHOLD,
            sampling_rate=SILERO_SR,
            return_seconds=True
        )

        merged = merge_segments(speech_timestamps, gap_sec=MERGE_GAP)

        if not merged:
            return {
                'filename': str(rel_path),
                'total_sec': round(total_sec, 2),
                'has_voice': False,
                'voice_segments_count': 0,
                'total_voice_sec': 0.0,
                'voice_ratio_pct': 0.0,
                'segments_detail': '',
                'error': ''
            }
        else:
            total_voice = sum(s['end'] - s['start'] for s in merged)
            # 格式：[開始-結束, 開始-結束, ...]  例：[210.90-212.30, 212.90-216.50]
            detail = ' | '.join(
                f"{s['start']:.2f}-{s['end']:.2f}"
                for s in merged
            )
            # 同時換算成 分:秒 格式方便閱讀
            detail_mmss = ' | '.join(
                f"{int(s['start'])//60}:{s['start']%60:05.2f}-{int(s['end'])//60}:{s['end']%60:05.2f}"
                for s in merged
            )
            return {
                'filename': str(rel_path),
                'total_sec': round(total_sec, 2),
                'has_voice': True,
                'voice_segments_count': len(merged),
                'total_voice_sec': round(total_voice, 2),
                'voice_ratio_pct': round(total_voice / total_sec * 100, 1),
                'segments_detail': detail,
                'segments_mmss': detail_mmss,
                'error': ''
            }

    except Exception as e:
        return {
            'filename': str(rel_path),
            'total_sec': 0.0,
            'has_voice': False,
            'voice_segments_count': 0,
            'total_voice_sec': 0.0,
            'voice_ratio_pct': 0.0,
            'segments_detail': '',
            'segments_mmss': '',
            'error': str(e)
        }


def main():
    audio_files = sorted(AUDIO_DIR.rglob('*.ogg'))
    total = len(audio_files)
    print(f"共找到 {total} 個音訊檔案，使用 {NUM_WORKERS} 個 worker 開始處理...")

    fieldnames = [
        'filename', 'total_sec', 'has_voice',
        'voice_segments_count', 'total_voice_sec', 'voice_ratio_pct',
        'segments_detail', 'segments_mmss', 'error'
    ]

    completed = 0
    voice_count = 0

    # 使用 utf-8-sig 讓 Excel 開啟時不會亂碼
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with Pool(processes=NUM_WORKERS) as pool:
            for result in pool.imap_unordered(process_file, audio_files, chunksize=4):
                completed += 1
                if result.get('has_voice'):
                    voice_count += 1

                # 補上缺少的欄位 (避免 DictWriter 報錯)
                for field in fieldnames:
                    result.setdefault(field, '')

                writer.writerow(result)
                f.flush()  # 即時寫入，避免中途中斷遺失資料

                # 進度顯示
                status = "有人聲" if result['has_voice'] else "乾淨"
                print(f"[{completed}/{total}] {status} | {result['filename']}", flush=True)

    print(f"\n完成！共 {total} 筆，其中 {voice_count} 筆含有人聲。")
    print(f"報告已儲存至：{OUTPUT_CSV.resolve()}")


if __name__ == '__main__':
    # Windows multiprocessing 必須在 main guard 內啟動
    main()
