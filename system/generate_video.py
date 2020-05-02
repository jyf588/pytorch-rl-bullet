"""Given a folder of pngs, this script generates a mp4 video."""
import argparse
import os


def main(args: argparse.Namespace):
    # Collect the png filenames, in sorted order.
    # paths = [os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir)]
    # paths = sorted(paths)

    # Run the ffmpeg command to generate the video.
    fps = 25
    # command = f"ffmpeg -r {fps} -i {args.src_dir}/%06d.png -vb 20M -vcodec mpeg4 -y {args.temp_path}"
    # os.system(command)
    # command = f'ffmpeg -i {args.temp_path} -filter:v "setpts=0.05*PTS" -y {args.dst_path}'
    # os.system(command)

    # command = f'ffmpeg -i {args.src_dir}/%06d.png -filter:v "setpts=PTS/8,fps=30" -vb 20M -vcodec mpeg4 -y {args.dst_path}'
    command = f'ffmpeg -i {args.src_dir}/%06d.png -filter:v "setpts=PTS/1,fps=20" -vb 20M -vcodec mpeg4 -y {args.dst_path}'
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/Users/michelleguo/workspace/lucas/unity/Captures/temp/third/rgb",
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
        default="/Users/michelleguo/Desktop/test.mp4",
        help="The destination path to write the mp4 video to.",
    )
    args = parser.parse_args()
    main(args=args)
