"""
跆拳道卡腳檢測系統使用示例

這個示例展示了如何使用 BatchTaekwondoDetector 類來檢測跆拳道比賽中的卡腳情況。
"""
import os
import argparse
from taekwondo import BatchTaekwondoDetector


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="跆拳道卡腳檢測示例")
    
    parser.add_argument("--video", "-v", type=str, required=True,
                        help="輸入視頻文件路徑")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="輸出視頻文件路徑 (默認為輸入文件名_result.mp4)")
    parser.add_argument("--skip", "-s", type=int, default=2,
                        help="處理時每隔多少幀處理一次 (默認: 2)")
    parser.add_argument("--width", "-w", type=int, default=480,
                        help="處理時調整的寬度 (默認: 480)")
    parser.add_argument("--height", "-ht", type=int, default=360,
                        help="處理時調整的高度 (默認: 360)")
    parser.add_argument("--api-key", "-k", type=str, default=None,
                        help="Roboflow API金鑰 (默認使用環境變數或config.py)")
    parser.add_argument("--play", "-p", action="store_true",
                        help="處理完後播放結果")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="播放速度 (默認: 1.0)")
    
    return parser.parse_args()


def main():
    """主函數"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 設置默認輸出路徑
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base_name}_result.mp4"
    
    print(f"處理視頻: {args.video}")
    print(f"輸出路徑: {args.output}")
    print(f"處理參數: 跳過幀={args.skip}, 處理尺寸={args.width}x{args.height}")
    
    # 創建檢測器
    detector = BatchTaekwondoDetector(api_key=args.api_key)
    
    try:
        # 處理視頻
        detector.process_video_batch(
            video_path=args.video,
            output_path=args.output,
            skip_frames=args.skip,
            resize_width=args.width,
            resize_height=args.height,
            show_progress=True
        )
        
        # 如果設置了播放選項，則播放結果
        if args.play and os.path.exists(args.output):
            print(f"\n正在以 {args.speed}x 速度播放結果...")
            detector.play_result(args.output, speed=args.speed)
    
    except Exception as e:
        print(f"處理時發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()