import streamlit as st
import cv2
import tempfile
import os
from Taekwondo import BatchTaekwondoDetector

st.set_page_config(page_title="跆拳道卡腳分析系統", layout="wide")
st.title("跆拳道卡腳與重心穩定度分析系統")

st.sidebar.header("參數設定")
skip_frames = st.sidebar.slider("跳幀數", 0, 10, 2, help="設定每隔多少幀分析一次，越高處理越快")
resize_width = st.sidebar.number_input("處理寬度", 240, 1920, 480, help="影片處理的寬度，越小處理越快")
resize_height = st.sidebar.number_input("處理高度", 180, 1080, 360, help="影片處理的高度，越小處理越快")

uploaded_file = st.file_uploader("上傳跆拳道比賽影片", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 保存上傳的檔案
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    
    # 設定輸出路徑
    output_path = os.path.join(tempfile.gettempdir(), "analyzed_result.mp4")
    
    col1, col2 = st.columns(2)
    with col1:
        st.video(input_path)
        st.caption("原始影片")
    
    # 處理按鈕
    if st.button("開始分析"):
        with st.spinner("正在分析影片...這可能需要幾分鐘時間"):
            # 初始化並處理
            detector = BatchTaekwondoDetector()
            detector.process_video_batch(
                input_path,
                output_path,
                skip_frames=skip_frames,
                resize_width=resize_width,
                resize_height=resize_height,
                show_progress=False
            )
            
            # 顯示統計結果
            st.subheader("卡腳統計結果")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("總卡腳次數", f"{detector.block_count}次")
            
            if detector.block_durations:
                durations = [block['duration'] for block in detector.block_durations]
                with col2:
                    st.metric("平均持續時間", f"{sum(durations)/len(durations):.2f}秒")
                with col3:
                    st.metric("最長持續時間", f"{max(durations):.2f}秒")
            
            # 顯示處理後的影片
            with col2:
                st.video(output_path)
                st.caption("分析結果")
            
            # 詳細分析結果
            st.subheader("詳細卡腳資訊")
            for i, block in enumerate(detector.block_durations, 1):
                with st.expander(f"卡腳 #{i} - 發生於 {block['start_time']:.2f}秒"):
                    block_col1, block_col2 = st.columns(2)
                    with block_col1:
                        st.write(f"持續時間: {block['duration']:.2f}秒")
                        st.write(f"紅方支撐腳移動: {'是' if block.get('red_moved', False) else '否'}")
                        st.write(f"藍方支撐腳移動: {'是' if block.get('blue_moved', False) else '否'}")
                    with block_col2:
                        st.write(f"紅方重心穩定度: {block.get('red_stability', 0):.2f}%")
                        st.write(f"藍方重心穩定度: {block.get('blue_stability', 0):.2f}%")
                
    # 清理臨時文件
    os.unlink(input_path)