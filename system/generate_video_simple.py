"""Given a folder of pngs, this script generates a mp4 video."""
import argparse
import os


def main(args: argparse.Namespace):
    # Collect the png filenames, in sorted order.
    # paths = [os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir)]
    # paths = sorted(paths)

    # Run the ffmpeg command to generate the video.
    fps = 40
    # command = f"ffmpeg -r {fps} -i {args.src_dir}/%06d.png -vb 20M -vcodec mpeg4 -y {args.temp_path}"
    # os.system(command)
    # command = f'ffmpeg -i {args.temp_path} -filter:v "setpts=0.05*PTS" -y {args.dst_path}'
    # os.system(command)

    render_frequency = 6
    speed_up_factor = 40./25

    command = f'ffmpeg -i {args.src_dir}/%06d.png -frames:v 25000 -filter:v "setpts=PTS/{speed_up_factor},fps={fps}" -vb 20M -vcodec libx264 -preset veryslow -y {args.dst_path}'
    # command = f'ffmpeg -i {args.src_dir}/%06d.png -frames:v 50000 -vf "fps={fps}" -vb 20M -vcodec libx264 -preset veryslow -y {args.dst_path}'

    # command = f'ffmpeg -i {args.src_dir}/%06d.png -frames:v 50000 -filter:v -vb 20M -vcodec libx264 -preset veryslow -y {args.dst_path}'


    # command = f'ffmpeg -i {args.src_dir}/%06d.png -filter:v "setpts=PTS/{speed_up_factor},fps={fps}" -vb 20M -vcodec mpeg4 -y {args.dst_path}'

    # For evaluation.
    # command = f"ffmpeg -framerate 5 -i {args.src_dir}/%06d.png -vb 20M -vcodec mpeg4 -y {args.dst_path}"
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/home/yifengj/Downloads/LinuxBuildLocalhost0512/Captures/temp/third/rgb",
        # default="/Users/michelleguo/Desktop/dash/demo_video/vision/0507",
        help="The directory containing the png images to convert into video format.",
    )
    parser.add_argument(
        "--temp_path",
        type=str,
        default="/Users/michelleguo/Desktop/tmp.mp4",
        help="The destination path to write the mp4 video to.",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        default="/home/yifengj/Downloads/test-4.mp4",
        help="The destination path to write the mp4 video to.",
    )
    args = parser.parse_args()
    main(args=args)
