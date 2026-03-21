"""
人聲過濾工具模組
從 voice_detection_report.csv 載入人聲時間段，
提供函式判斷某個時間窗口是否與人聲重疊。
"""

import pandas as pd
from pathlib import Path


def load_voice_segments(csv_path: str) -> dict:
    """
    讀取 voice_detection_report.csv，回傳人聲時間段字典。

    :return: { 'species/filename.ogg': [(start_sec, end_sec), ...], ... }
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df[df['has_voice'] == True]

    voice_map = {}
    for _, row in df.iterrows():
        # CSV 路徑是 train_audio\species\file.ogg，統一轉成 species/file.ogg 作為 key
        raw_path = str(row['filename']).replace('\\', '/')
        # 去掉開頭的 train_audio/
        key = '/'.join(raw_path.split('/')[-2:])

        segments = []
        if pd.notna(row['segments_detail']) and str(row['segments_detail']).strip():
            for seg in str(row['segments_detail']).split('|'):
                seg = seg.strip()
                if '-' in seg:
                    parts = seg.split('-')
                    try:
                        segments.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        voice_map[key] = segments

    return voice_map


def has_voice_overlap(voice_segments: list, start_sec: float, end_sec: float,
                      tolerance: float = 0.5) -> bool:
    """
    判斷 [start_sec, end_sec] 這個窗口是否與任何人聲片段重疊。

    :param voice_segments: [(v_start, v_end), ...] 該音訊的人聲時間段
    :param start_sec:      窗口起始秒數
    :param end_sec:        窗口結束秒數
    :param tolerance:      容忍重疊秒數，小於此值視為不重疊（預設 0.5 秒）
    :return: True 表示有重疊，應跳過此片段
    """
    for v_start, v_end in voice_segments:
        overlap_start = max(start_sec, v_start)
        overlap_end = min(end_sec, v_end)
        if overlap_end - overlap_start > tolerance:
            return True
    return False
